#!/usr/bin/env python3
"""ColBERTv2 zero-shot FULL-CORPUS retrieval on BEIR SciDocs, no reranking shortcut."""
import argparse, csv, json, os, time
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/scidocs')
    ap.add_argument('--output', default='results/colbert')
    ap.add_argument('--checkpoint', default='colbert-ir/colbertv2.0')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--doc-maxlen', type=int, default=180)
    ap.add_argument('--query-maxlen', type=int, default=32)
    ap.add_argument('--nbits', type=int, default=2)
    ap.add_argument('--index-name', default='scidocs_colbertv2_2bit')
    ap.add_argument('--index-root', default='colbert_indexes')
    ap.add_argument('--skip-index', action='store_true')
    args = ap.parse_args()

    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Collection

    data = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    corpus_file, queries_file, qrels_file = data/'corpus.jsonl', data/'queries.jsonl', data/'qrels'/'test.tsv'
    for p in (corpus_file, queries_file, qrels_file):
        if not p.is_file():
            raise FileNotFoundError(f'Missing {p}. Copy the SAME BEIR SciDocs data folder used for BM25/DPR/SPLADE.')

    # Stable mapping: ColBERT internal pid (0..N-1) -> original SciDocs document ID.
    mapping_file = out/'doc_ids.json'
    collection_file = out/'collection.tsv'
    if not args.skip_index:
        docids = []
        with open(collection_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            for pid, row in enumerate(load_jsonl(corpus_file)):
                docids.append(str(row['_id']))
                # Clean tabs/newlines to preserve tsv format; use title + abstract.
                title = ' '.join(str(row.get('title') or '').split())
                body = ' '.join(str(row.get('text') or '').split())
                passage = (title + ' ' + body).strip()
                # ColBERT's TSV parser calls strip() before split('\t').
                # An empty passage would become an ID-only row and crash indexing.
                if not passage:
                    passage = '[EMPTY DOCUMENT]'
                writer.writerow([pid, passage])
        mapping_file.write_text(json.dumps(docids), encoding='utf-8')
        print(f'Corpus={len(docids)} docs; collection created at {collection_file}', flush=True)
    else:
        if not mapping_file.is_file() or not collection_file.is_file():
            raise FileNotFoundError('Cannot --skip-index without prior doc_ids.json and collection.tsv')
        docids = json.loads(mapping_file.read_text(encoding='utf-8'))

    query_rows = list(load_jsonl(queries_file))
    # Evaluate only queries with qrels in SciDocs test set.
    qrels = defaultdict(dict)
    with open(qrels_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for fields in reader:
            if len(fields) < 3 or fields[0] == 'query-id':
                continue
            qid, docid, rel = fields[:3]
            qrels[qid][docid] = int(rel)
    query_lookup = {str(x['_id']): x.get('text', '') for x in query_rows}
    qids = sorted(set(query_lookup) & set(qrels))
    if not qids:
        raise RuntimeError('No test queries matched queries.jsonl and qrels/test.tsv')
    print(f'Test queries={len(qids)}; corpus={len(docids)}', flush=True)
    print('Checkpoint:', args.checkpoint, '\nThis is ColBERTv2, not original SIGIR 2020 ColBERTv1.', flush=True)

    run_cfg = RunConfig(nranks=1, experiment='scidocs_exp03', root='colbert_experiments')
    config = ColBERTConfig(nbits=args.nbits, doc_maxlen=args.doc_maxlen,
                           query_maxlen=args.query_maxlen, bsize=args.batch_size,
                           kmeans_niters=4, index_root=str(Path(args.index_root).absolute()))
    # Resume by calling --skip-index only when a complete index already exists.
    with Run().context(run_cfg):
        if not args.skip_index:
            start = time.perf_counter()
            indexer = Indexer(checkpoint=args.checkpoint, config=config)
            indexer.index(name=args.index_name, collection=str(collection_file), overwrite='reuse')
            index_seconds = time.perf_counter() - start
        else:
            index_seconds = None

        searcher = Searcher(index=args.index_name, checkpoint=args.checkpoint,
                            collection=str(collection_file), config=config,
                            index_root=str(Path(args.index_root).absolute()))
        results = {}
        timings = []
        trec_path = out/'run.trec'
        ranked_path = out/'ranked_papers.csv'
        with open(trec_path, 'w', encoding='utf-8') as tf, open(ranked_path, 'w', encoding='utf-8', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['query_id', 'doc_id', 'rank', 'score'])
            from tqdm import tqdm
            for qid in tqdm(qids, desc='Searching SciDocs'):
                started = time.perf_counter()
                pids, ranks, scores = searcher.search(query_lookup[qid], k=10)
                timings.append((time.perf_counter()-started)*1000)
                scores_dict = {}
                for pid, rank, score in zip(pids, ranks, scores):
                    docid = docids[int(pid)]
                    scores_dict[docid] = float(score)
                    tf.write(f'{qid} Q0 {docid} {rank} {float(score):.10f} ColBERTv2\n')
                    writer.writerow([qid, docid, rank, float(score)])
                results[qid] = scores_dict

    # Official BEIR evaluation is performed externally using verify_colbert.py,
    # which avoids coupling the ColBERT GPU environment to BEIR dependencies.
    import statistics
    metrics = {'method': 'ColBERTv2 MS MARCO pretrained zero-shot PLAID retrieval',
               'dataset':'BEIR SciDocs', 'checkpoint':args.checkpoint,
               'corpus_size':len(docids), 'evaluated_queries':len(qids),
               'top_k':10, 'doc_maxlen':args.doc_maxlen, 'query_maxlen':args.query_maxlen,
               'nbits':args.nbits, 'index_seconds_this_run':index_seconds,
               'mean_query_latency_ms':statistics.mean(timings),
               'median_query_latency_ms':statistics.median(timings),
               'note':'Latency includes query encoding and index retrieval, not corpus indexing. Official BEIR metrics: run verify_colbert.py in BEIR environment.'}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Saved', trec_path, ranked_path, out/'metrics.json', flush=True)
    print('Now run verify_colbert.py in your DPR (BEIR) environment.', flush=True)

if __name__ == '__main__':
    main()

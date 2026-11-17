import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SparseEncoder

MODEL = 'naver/splade-cocondenser-ensembledistil'

def scipy_from_batch(x):
    # SparseEncoder may return a torch sparse tensor or dense tensor depending on its version.
    if isinstance(x, torch.Tensor):
        if x.layout != torch.strided:
            x = x.to_dense()
        x = x.detach().cpu().numpy()
    if sp.issparse(x):
        return x.tocsr().astype(np.float32)
    if isinstance(x, list):
        x = np.asarray(x)
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/scidocs')
    p.add_argument('--out', default='results/splade')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Device={a.device}; torch={torch.__version__}; model={MODEL}', flush=True)
    corpus, queries, qrels = GenericDataLoader(data_folder=a.data).load(split='test')
    doc_ids = sorted(corpus)
    query_ids = sorted(qrels)
    query_ids = [qid for qid in query_ids if qid in queries]
    texts = [(corpus[i].get('title', '') + ' ' + corpus[i].get('text', '')).strip() for i in doc_ids]
    print(f'Corpus={len(doc_ids)}, evaluated queries={len(query_ids)}', flush=True)
    model = SparseEncoder(MODEL, device=a.device)
    model.max_seq_length = a.max_length

    cache = folder / f'doc_vectors_{len(doc_ids)}_maxlen{a.max_length}.npz'
    ids_cache = folder / 'doc_ids.json'
    if cache.exists() and ids_cache.exists() and json.loads(ids_cache.read_text()) == doc_ids:
        doc_vectors = sp.load_npz(cache).tocsr()
        encode_time = 0.0
        cached = True
        print('Reusing cached document vectors', flush=True)
    else:
        started = time.perf_counter()
        pieces = []
        for start in tqdm(range(0, len(texts), a.batch_size), desc='Encoding corpus'):
            batch = texts[start:start + a.batch_size]
            encoded = model.encode_document(batch, batch_size=a.batch_size, show_progress_bar=False)
            pieces.append(scipy_from_batch(encoded))
        doc_vectors = sp.vstack(pieces, format='csr')
        del pieces
        encode_time = time.perf_counter() - started
        sp.save_npz(cache, doc_vectors)
        ids_cache.write_text(json.dumps(doc_ids))
        cached = False
    print(f'Sparse index: shape={doc_vectors.shape}, nnz={doc_vectors.nnz}', flush=True)
    if doc_vectors.shape[0] != len(doc_ids):
        raise RuntimeError('Document cache shape mismatch')

    results = {}
    times = []
    with (folder / 'run.trec').open('w') as f_run, (folder / 'ranked_papers.csv').open('w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['query_id','rank','document_id','score','query','document_title'])
        for qid in tqdm(query_ids, desc='Retrieving queries'):
            tick = time.perf_counter()
            encoded = model.encode_query([queries[qid]], batch_size=1, show_progress_bar=False)
            qvec = scipy_from_batch(encoded)
            scores = (doc_vectors @ qvec.T).toarray().ravel()
            # Stable ordering makes tied zero-score documents reproducible.
            top = np.argsort(-scores, kind='stable')[:10]
            times.append((time.perf_counter() - tick) * 1000)
            results[qid] = {}
            for rank, idx in enumerate(top, start=1):
                did = doc_ids[int(idx)]
                score = float(scores[idx])
                results[qid][did] = score
                f_run.write(f'{qid} Q0 {did} {rank} {score:.10f} SPLADE\n')
                writer.writerow([qid, rank, did, score, queries[qid], corpus[did].get('title', '')])

    # BEIR evaluates only the qrels queries and uses its reference metric definitions.
    ndcg, maps, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [10])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [10], metric='mrr')
    metrics = {
        'precision_at_10': precision['P@10'],
        'recall_at_10': recall['Recall@10'],
        'mrr_at_10': mrr['MRR@10'],
        'ndcg_at_10': ndcg['NDCG@10'],
        'map_at_10': maps['MAP@10'],
        'evaluated_queries': len(query_ids),
        'dataset': 'BEIR SciDocs',
        'method': 'SPLADE++ CoCondenser EnsembleDistil pretrained, sparse dot product',
        'checkpoint': MODEL,
        'corpus_size': len(doc_ids),
        'top_k': 10,
        'device': a.device,
        'batch_size': a.batch_size,
        'max_length': a.max_length,
        'corpus_encoding_seconds_this_run': round(encode_time, 3),
        'reused_cached_document_vectors': cached,
        'mean_query_latency_ms': round(float(np.mean(times)), 3),
        'median_query_latency_ms': round(float(np.median(times)), 3),
        'p95_query_latency_ms': round(float(np.percentile(times, 95)), 3),
        'index_nonzero_entries': int(doc_vectors.nnz),
        'note': 'Zero-shot pretrained checkpoint; no SciDocs training. Query latency includes query encoding and exact sparse dot product; corpus encoding excluded.'
    }
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print('RESULTS', json.dumps(metrics, indent=2), sep='\n', flush=True)

if __name__ == '__main__':
    main()

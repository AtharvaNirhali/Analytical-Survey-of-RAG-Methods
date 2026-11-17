#!/usr/bin/env python3
"""EXP-04: BGE-M3 dense-only, zero-shot, exact dot-product on BEIR SciDocs."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

MODEL = 'BAAI/bge-m3'


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/scidocs')
    p.add_argument('--output-dir', default='results/bge_m3')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--doc-maxlen', type=int, default=256)
    p.add_argument('--query-maxlen', type=int, default=64)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--reencode', action='store_true', help='Ignore saved document embeddings')
    return p.parse_args()


def get_embeddings(model, tokenizer, texts, batch_size, maxlen, device, description):
    arrays = []
    for offset in tqdm(range(0, len(texts), batch_size), desc=description, unit='batch'):
        items = texts[offset:offset+batch_size]
        tokens = tokenizer(items, padding=True, truncation=True, max_length=maxlen,
                           return_tensors='pt').to(device)
        with torch.inference_mode():
            if device.type == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(**tokens)
            else:
                output = model(**tokens)
            # BGE-M3 dense retrieval uses normalized CLS representation.
            vecs = F.normalize(output.last_hidden_state[:, 0, :].float(), p=2, dim=-1)
        arrays.append(vecs.cpu().numpy())
    return np.concatenate(arrays, axis=0).astype(np.float32)


def main():
    a = args_parser()
    data = Path(a.data_dir)
    dest = Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not (data/'corpus.jsonl').exists() or not (data/'queries.jsonl').exists() or not (data/'qrels'/'test.tsv').exists():
        raise FileNotFoundError(f'BEIR SciDocs not found at {data}. Need corpus.jsonl, queries.jsonl, qrels/test.tsv')
    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) or a.device == 'cuda' else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; use --device cpu.')
    print(f'Device: {device}; torch={torch.__version__}', flush=True)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data)).load(split='test')
    # Restrict to queries with relevance judgments, matching prior BEIR runs.
    qids = [qid for qid in queries if qid in qrels]
    docids = sorted(corpus.keys())
    texts = [(' '.join(filter(None, [(corpus[did].get('title') or '').strip(),
                                      (corpus[did].get('text') or '').strip()])).strip() or '[EMPTY DOCUMENT]') for did in docids]
    print(f'Corpus={len(docids)}; evaluated queries={len(qids)}', flush=True)

    cache = dest / f'doc_vectors_{len(docids)}_maxlen{a.doc_maxlen}.npy'
    ids_cache = dest / 'doc_ids.json'
    reused = False
    corpus_seconds = 0.0
    tokenizer = None
    model = None
    if cache.exists() and ids_cache.exists() and not a.reencode:
        old_ids = json.loads(ids_cache.read_text())
        if old_ids == docids:
            doc_vecs = np.load(cache)
            if doc_vecs.shape == (len(docids), 1024):
                reused = True
                print('Reusing cached BGE-M3 document embeddings.', flush=True)
    if not reused:
        print('Loading BGE-M3 checkpoint; first run downloads ~2.3GB model weights.', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(device).eval()
        start = time.perf_counter()
        doc_vecs = get_embeddings(model, tokenizer, texts, a.batch_size, a.doc_maxlen, device, 'Encoding corpus')
        corpus_seconds = time.perf_counter() - start
        np.save(cache, doc_vecs)
        ids_cache.write_text(json.dumps(docids))
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(device).eval()
    print(f'Corpus vectors={doc_vecs.shape}; mode=BGE-M3 dense-only, zero-shot', flush=True)

    # On CUDA 4GB, ~100 MB matrix. If memory is tight, rerun with --device cpu.
    doc_matrix = torch.from_numpy(doc_vecs).to(device)
    rankings = {}
    timings = []
    with (dest/'run.trec').open('w') as run_f, (dest/'ranked_papers.csv').open('w', newline='') as papers_f:
        writer = csv.writer(papers_f)
        writer.writerow(['query_id', 'query', 'rank', 'doc_id', 'score', 'title'])
        for qid in tqdm(qids, desc='Retrieving queries'):
            query = queries[qid]
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()
            qvec = get_query(model, tokenizer, query, a.query_maxlen, device)
            scores = torch.mv(doc_matrix, qvec)
            values, indices = torch.topk(scores, k=10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
            values, indices = values.cpu().tolist(), indices.cpu().tolist()
            rankings[qid] = {}
            for rank, (score, idx) in enumerate(zip(values, indices), 1):
                did = docids[idx]
                rankings[qid][did] = float(score)
                run_f.write(f'{qid} Q0 {did} {rank} {score:.10f} BGE-M3-Dense\n')
                writer.writerow([qid, query, rank, did, f'{score:.10f}', corpus[did].get('title', '')])

    ndcg, ap, recall, prec = EvaluateRetrieval.evaluate(qrels, rankings, [10])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, rankings, [10], metric='mrr')
    metrics = {
        'precision_at_10': prec['P@10'], 'recall_at_10': recall['Recall@10'],
        'mrr_at_10': mrr['MRR@10'], 'ndcg_at_10': ndcg['NDCG@10'], 'map_at_10': ap['MAP@10'],
        'evaluated_queries': len(qids), 'dataset': 'BEIR SciDocs',
        'method': 'BGE-M3 pretrained zero-shot dense-only normalized CLS, exact dot product',
        'checkpoint': MODEL, 'corpus_size': len(docids), 'top_k': 10,
        'device': str(device), 'batch_size': a.batch_size,
        'doc_maxlen': a.doc_maxlen, 'query_maxlen': a.query_maxlen,
        'corpus_encoding_seconds_this_run': round(corpus_seconds, 3),
        'reused_cached_document_vectors': reused,
        'mean_query_latency_ms': round(float(np.mean(timings)), 3),
        'median_query_latency_ms': round(float(np.median(timings)), 3),
        'p95_query_latency_ms': round(float(np.percentile(timings, 95)), 3),
        'evaluation': 'Official BEIR EvaluateRetrieval (in-process)',
        'note': 'Only dense retrieval is tested, not BGE-M3 sparse or multi-vector fusion. Query timing includes query encoding + exact vector search but excludes corpus encoding.'
    }
    (dest/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (dest/'beir_verified.json').write_text(json.dumps({k:metrics[k] for k in
        ('precision_at_10','recall_at_10','mrr_at_10','ndcg_at_10','map_at_10')}, indent=2))
    print('\nOFFICIAL BEIR RESULTS')
    print(json.dumps(metrics, indent=2))


def get_query(model, tokenizer, query, maxlen, device):
    tokens = tokenizer([query], padding=True, truncation=True, max_length=maxlen,
                       return_tensors='pt').to(device)
    with torch.inference_mode():
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(**tokens)
        else:
            out = model(**tokens)
        return F.normalize(out.last_hidden_state[:, 0, :].float(), p=2, dim=-1)[0]


if __name__ == '__main__':
    main()

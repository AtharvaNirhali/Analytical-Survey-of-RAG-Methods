#!/usr/bin/env python3
"""EXP-01: Pretrained DPR retrieval on official BEIR SciDocs (test split).

Uses original Facebook DPR question/context encoder pair and dot-product scoring.
No training or test-set tuning. Saves TREC ranking, per-query rankings and metrics.
"""
import argparse
import json
import os
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
)

DATA_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip'
CTX = 'facebook/dpr-ctx_encoder-single-nq-base'
QUESTION = 'facebook/dpr-question_encoder-single-nq-base'


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', default='results/dpr')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--limit-queries', type=int, default=0, help='Smoke test only (not comparable)')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--no-cache', action='store_true')
    return p.parse_args()


def prepare_data(folder):
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / 'scidocs'
    if (base / 'corpus.jsonl').exists() and (base / 'queries.jsonl').exists():
        return base
    zip_path = folder / 'scidocs.zip'
    if not zip_path.exists():
        print('Downloading official BEIR SciDocs (one time)...')
        urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (folder / member.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise RuntimeError('Unsafe ZIP member path')
        archive.extractall(folder)
    if not (base / 'corpus.jsonl').exists():
        raise RuntimeError('SciDocs files missing after extraction')
    return base


def load_jsonl(filepath):
    with filepath.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(filepath):
    df = pd.read_csv(filepath, sep='\t', dtype={'query-id': str, 'corpus-id': str})
    return {str(q): {str(d): int(s) for d, s in zip(group['corpus-id'], group['score'])}
            for q, group in df.groupby('query-id', sort=False)}


def encode_contexts(rows, device, batch_size, max_length):
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(CTX)
    model = DPRContextEncoder.from_pretrained(CTX).to(device).eval()
    vectors = []
    started = time.perf_counter()
    for i in tqdm(range(0, len(rows), batch_size), desc='Encoding corpus'):
        chunk = rows[i:i + batch_size]
        # Native DPR context format: title [SEP] passage.
        enc = tokenizer(
            [r.get('title', '') or '' for r in chunk],
            [r.get('text', '') or '' for r in chunk],
            truncation=True, max_length=max_length, padding=True, return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            vec = model(**enc).pooler_output
        vectors.append(vec.float().cpu().numpy())
    seconds = time.perf_counter() - started
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0), seconds


def search(qvec, vectors, k):
    scores = vectors @ qvec
    n = min(k, len(scores))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top], kind='stable')]
    return top, scores[top]


def metric_for_query(ranked, qrel, k=10):
    relevant = {d for d, value in qrel.items() if value > 0}
    retrieved = ranked[:k]
    hits = [1 if d in relevant else 0 for d in retrieved]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum((2**qrel.get(docid, 0) - 1) / np.log2(rank + 2)
              for rank, docid in enumerate(retrieved))
    ideal = sorted((max(0, score) for score in qrel.values()), reverse=True)[:k]
    idcg = sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(ideal))
    ap = sum(sum(hits[:i + 1]) / (i + 1) for i, h in enumerate(hits) if h)
    # BEIR/trec_eval map_cut@k uses min(k, number of relevant docs) as denominator.
    ap /= min(k, len(relevant)) if relevant else 1
    return dict(precision_at_10=precision, recall_at_10=recall,
                mrr_at_10=mrr, ndcg_at_10=dcg / idcg if idcg else 0.0,
                map_at_10=ap)


def main():
    a = args_parse()
    if a.top_k != 10:
        raise ValueError('This baseline report uses top_k=10 to match EXP-00.')
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else
                          a.device if a.device != 'auto' else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu.')
    print(f'Device: {device}; torch={torch.__version__}')
    base = prepare_data(Path(a.data_dir))
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus = load_jsonl(base / 'corpus.jsonl')
    all_queries = load_jsonl(base / 'queries.jsonl')
    qrels = load_qrels(base / 'qrels' / 'test.tsv')
    queries = [q for q in all_queries if str(q['_id']) in qrels]
    if a.limit_queries > 0:
        queries = queries[:a.limit_queries]
    ids = [str(doc['_id']) for doc in corpus]
    print(f'Corpus={len(ids)}, evaluated queries={len(queries)}')
    if len(ids) != 25657:
        print('WARNING: corpus size differs from official SciDocs 25,657')
    cache = output / f'ctx_vectors_{len(ids)}_maxlen{a.max_length}.npy'
    start_index = time.perf_counter()
    if cache.exists() and not a.no_cache:
        vectors = np.load(cache, mmap_mode='r')
        corpus_encoding_seconds = 0.0
        reused_cache = True
        print('Loaded cached document embeddings')
    else:
        vectors, corpus_encoding_seconds = encode_contexts(corpus, device, a.batch_size, a.max_length)
        np.save(cache, vectors)
        reused_cache = False
    index_elapsed = time.perf_counter() - start_index

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION)
    model = DPRQuestionEncoder.from_pretrained(QUESTION).to(device).eval()
    rows = []
    trec = []
    per_query = []
    times = []
    for q in tqdm(queries, desc='Searching'):
        qid = str(q['_id'])
        t0 = time.perf_counter()
        inputs = tokenizer(q['text'], return_tensors='pt', truncation=True,
                           max_length=a.max_length).to(device)
        with torch.inference_mode():
            qvector = model(**inputs).pooler_output[0].float().cpu().numpy()
        indices, scores = search(qvector, vectors, a.top_k)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        ranked = []
        for rank, (index, score) in enumerate(zip(indices, scores), 1):
            doc = corpus[int(index)]
            did = ids[int(index)]
            ranked.append(did)
            rows.append({'query_id': qid, 'query_text': q['text'], 'document_id': did,
                         'rank': rank, 'score': float(score), 'paper_title': doc.get('title', '')})
            trec.append(f'{qid} Q0 {did} {rank} {float(score):.8f} DPR')
        per_query.append(metric_for_query(ranked, qrels[qid]))
    df = pd.DataFrame(rows)
    df.to_csv(output / 'ranked_papers.csv', index=False)
    (output / 'run.trec').write_text('\n'.join(trec) + '\n', encoding='utf-8')
    pd.DataFrame(per_query).assign(query_id=[str(q['_id']) for q in queries]).to_csv(
        output / 'per_query_metrics.csv', index=False)
    mean_metrics = {name: float(pd.DataFrame(per_query)[name].mean()) for name in per_query[0]}
    report = {**mean_metrics, 'evaluated_queries': len(queries), 'dataset': 'BEIR SciDocs',
              'method': 'DPR NQ pretrained, dot product', 'top_k': a.top_k,
              'corpus_size': len(ids), 'device': str(device), 'batch_size': a.batch_size,
              'max_length': a.max_length, 'context_model': CTX, 'question_model': QUESTION,
              'index_seconds_this_run': round(index_elapsed, 3),
              'corpus_encoding_seconds_this_run': round(corpus_encoding_seconds, 3),
              'reused_cached_context_vectors': reused_cache,
              'mean_query_latency_ms': round(float(np.mean(times)), 3),
              'median_query_latency_ms': round(float(np.median(times)), 3),
              'p95_query_latency_ms': round(float(np.percentile(times, 95)), 3),
              'process_rss_mb_at_end': round(psutil.Process().memory_info().rss / 1024**2, 2),
              'partial_smoke_test': bool(a.limit_queries),
              'notes': 'Uses pretrained NQ DPR, no SciDocs fine-tuning. Recall/MAP based on SciDocs positive qrels. '
                       'Question query encoding + exact dot-product search timed; excludes corpus encoding.'}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Outputs saved to: {output.resolve()}')


if __name__ == '__main__':
    main()

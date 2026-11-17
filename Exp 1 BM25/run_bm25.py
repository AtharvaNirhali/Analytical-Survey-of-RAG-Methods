#!/usr/bin/env python3
"""EXP-00: BM25 retrieval baseline on the official BEIR SciDocs test split."""
import argparse
import csv
import json
import os
import re
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def download_dataset(root):
    data_dir = root / "scidocs"
    if (data_dir / "corpus.jsonl").exists() and (data_dir / "queries.jsonl").exists():
        print(f"Using existing dataset: {data_dir}")
        return data_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "scidocs.zip"
    if not archive.exists():
        print(f"Downloading dataset from {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive)
    print("Extracting SciDocs...")
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Unexpected ZIP path")
        zf.extractall(root)
    if not (data_dir / "corpus.jsonl").exists():
        raise RuntimeError("Dataset extraction did not produce scidocs/corpus.jsonl")
    return data_dir


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        first = next(reader)
        # BEIR qrels: query-id, corpus-id, score; many copies contain header.
        if first[0].lower() not in {"query-id", "query_id"}:
            rows = [first]
        else:
            rows = []
        for row in list(rows) + list(reader):
            if len(row) >= 3:
                qrels.setdefault(row[0], {})[row[1]] = int(row[2])
    return qrels


def evaluate(ranked, qrels, k=10):
    totals = {"precision_at_10": 0., "recall_at_10": 0., "mrr_at_10": 0., "ndcg_at_10": 0., "map_at_10": 0.}
    count = 0
    for qid, relevance in qrels.items():
        if qid not in ranked:
            continue
        relevant = {d for d, grade in relevance.items() if grade > 0}
        if not relevant:
            continue
        docs = ranked[qid][:k]
        hits = [1 if docid in relevant else 0 for docid, _ in docs]
        totals["precision_at_10"] += sum(hits) / k
        totals["recall_at_10"] += sum(hits) / len(relevant)
        totals["mrr_at_10"] += next((1. / i for i, hit in enumerate(hits, 1) if hit), 0.)
        totals["map_at_10"] += sum(sum(hits[:i]) / i for i, hit in enumerate(hits, 1) if hit) / min(len(relevant), k)
        dcg = sum((2**relevance.get(docid, 0) - 1) / np.log2(i + 1) for i, (docid, _) in enumerate(docs, 1))
        ideal = sorted((grade for grade in relevance.values() if grade > 0), reverse=True)[:k]
        idcg = sum((2**grade - 1) / np.log2(i + 1) for i, grade in enumerate(ideal, 1))
        totals["ndcg_at_10"] += dcg / idcg if idcg else 0.
        count += 1
    if not count:
        raise RuntimeError("No evaluated queries; check qrels/query IDs")
    return {**{key: round(val / count, 6) for key, val in totals.items()}, "evaluated_queries": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="results/bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=0, help="Smoke test only; use 0 for full benchmark")
    args = parser.parse_args()
    data_dir = download_dataset(Path(args.data_root))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    corpus = list(read_jsonl(data_dir / "corpus.jsonl"))
    queries = list(read_jsonl(data_dir / "queries.jsonl"))
    qrels = load_qrels(data_dir / "qrels" / "test.tsv")
    queries = [q for q in queries if q["_id"] in qrels]
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    print(f"Documents: {len(corpus)} | Queries: {len(queries)} | Qrels: {len(qrels)}")
    document_ids = [str(item["_id"]) for item in corpus]
    texts = [(item.get("title", "") + " " + item.get("text", "")).strip() for item in corpus]
    t0 = time.perf_counter()
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    index_seconds = time.perf_counter() - t0
    print(f"BM25 index built in {index_seconds:.2f} sec")
    ranked = {}
    per_query_ms = []
    output_rows = []
    for j, query in enumerate(queries, 1):
        qid = str(query["_id"])
        qtext = query.get("text", "")
        t1 = time.perf_counter()
        scores = bm25.get_scores(tokenize(qtext))
        k = min(args.top_k, len(scores))
        best = np.argpartition(scores, -k)[-k:]
        best = sorted(best, key=lambda x: (-scores[x], document_ids[x]))
        per_query_ms.append((time.perf_counter() - t1) * 1000)
        ranked[qid] = [(document_ids[i], float(scores[i])) for i in best]
        for rank, i in enumerate(best, 1):
            output_rows.append((qid, document_ids[i], rank, float(scores[i]), corpus[i].get("title", "")))
        if j % 100 == 0 or j == len(queries):
            print(f"Processed {j}/{len(queries)} queries", flush=True)
    metrics = evaluate(ranked, qrels)
    metrics.update({
        "dataset": "BEIR SciDocs", "method": "BM25Okapi (rank_bm25)",
        "top_k": args.top_k, "corpus_size": len(corpus),
        "index_seconds": round(index_seconds, 3),
        "mean_query_latency_ms": round(float(np.mean(per_query_ms)), 3),
        "median_query_latency_ms": round(float(np.median(per_query_ms)), 3),
        "p95_query_latency_ms": round(float(np.percentile(per_query_ms, 95)), 3),
        "note": "Custom unstemmed tokenization; scores are not directly comparable to default Lucene/Elasticsearch BM25.",
        "partial_smoke_test": bool(args.limit_queries),
    })
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "ranked_papers.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "document_id", "rank", "bm25_score", "paper_title"])
        writer.writerows(output_rows)
    with open(out / "run.trec", "w", encoding="utf-8") as f:
        for qid, entries in ranked.items():
            for rank, (docid, score) in enumerate(entries, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.8f} BM25\n")
    print("\nFINAL METRICS\n", json.dumps(metrics, indent=2))
    print(f"\nSaved to: {out.resolve()}")

if __name__ == "__main__":
    main()

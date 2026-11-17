# Experiment 1 — BM25 Baseline on BEIR SciDocs

## Overview
This experiment evaluates **BM25Okapi**, a traditional lexical retrieval baseline, on the official **BEIR SciDocs test split**. It uses lowercased, unstemmed word tokens and combines each document's title and body text. No training or LLM is required.

**Script:** `run_bm25.py`  
**Dataset:** BEIR SciDocs (25,657 documents; 1,000 test queries)  
**Metrics:** Precision@10, Recall@10, MRR@10, nDCG@10, MAP@10.

## Suggested directory layout
```text
Exp 1 BM25/
├── README.md
├── requirements.txt
├── run_bm25.py
├── data/                 # generated/downloaded;
│   └── scidocs/
└── results/              # generated; 
    └── bm25/
```

## Setup (Ubuntu / conda)
```bash
conda create -n bm25 python=3.11 -y
conda activate bm25
python -m pip install -r requirements.txt
```
The supplied `requirements.txt` includes NumPy and `rank-bm25`.

## Run
The script automatically downloads and extracts the SciDocs dataset from the BEIR host on its first run.
```bash
# Quick smoke test only (not the full benchmark)
python run_bm25.py --limit-queries 10

# Complete SciDocs test set
python run_bm25.py
```
Optional output/dataset paths:
```bash
python run_bm25.py --data-root data --output results/bm25 --top-k 10
```
The reported metrics are specifically evaluated at 10, so use `--top-k 10` for the published comparison.

## Retrieval protocol
- Tokenizer: `\b\w+\b` on lowercase text (not Lucene/Elasticsearch tokenization).
- Model: `BM25Okapi`, `k1=1.5`, `b=0.75`.
- Documents: title + text; no stemming or model fine-tuning.
- For each query, retrieve the ten highest-scoring documents.
- Evaluate against `data/scidocs/qrels/test.tsv`.

## Outputs
Generated in `results/bm25/`:
- `metrics.json` — all five metrics, evaluated query count, index time, latency, and configuration.
- `ranked_papers.csv` — query ID, document ID, rank, BM25 score, title.
- `run.trec` — six-column TREC run; can be reused by Experiment 6 (BlendedRAG).

## Reproducibility notes
The script uses its own metric implementation rather than the `beir` evaluator; check metric conventions if combining results from other experiments. A limited-query run is marked `partial_smoke_test` and **must not** be presented as the full SciDocs benchmark. The first run needs an internet connection for SciDocs download.

## Upstream references
- BEIR benchmark: https://github.com/beir-cellar/beir
- `rank-bm25`: https://pypi.org/project/rank-bm25/

# EXP-00: BM25 / BEIR SciDocs

A CPU-only BM25 baseline for the AdHR-RAG **analytical survey**. This is **not** the authors' Elasticsearch BM25 configuration; it uses `rank-bm25` and explicitly reports that difference.

## Setup (Ubuntu)

```bash
cd bm25_scidocs_experiment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

```bash
python run_bm25.py --limit-queries 10
```

## Full official test split

```bash
python run_bm25.py
```

The script downloads the official BEIR SciDocs ZIP (around 142 MB), extracts it to `data/scidocs`, and runs BM25 over **all** corpus documents on the **same** test queries and qrels. Results are stored under `results/bm25`.

Outputs:
- `metrics.json`: retrieval-quality metrics and indexing/search timing
- `ranked_papers.csv`: per-query top-10 documents with titles/scores
- `run.trec`: standard TREC run file for additional evaluation

Note: rerun without `--limit-queries` after the smoke test; the smoke-test output metrics only summarize the selected queries. Indexing time includes tokenization. Latency includes scoring and ranking but not disk download. Metrics use test relevance judgments with relevance > 0; nDCG preserves graded judgments.

Do not compare these metrics directly against published Elasticsearch/Lucene BM25 scores without matching preprocessing, analyzer, and parameters. For the next five methods use the same `data/scidocs` files and qrels. OpenAlex/arXiv will be a **separate** online retrieval experiment, to avoid contaminating benchmark comparisons.

Dataset: https://github.com/beir-cellar/beir/wiki/Datasets-available
BM25 library: https://github.com/dorianbrown/rank_bm25

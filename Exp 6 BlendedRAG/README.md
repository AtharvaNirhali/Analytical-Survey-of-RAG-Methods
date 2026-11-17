# Experiment 6 — Blended RAG-Inspired Retrieval and RRF Ablations

## Overview
This folder evaluates **SciDocs field-aware retrieval** inspired by the ideas in Sawarkar et al.'s *Blended RAG*, plus a **separate three-way Reciprocal Rank Fusion (RRF)** experiment. It is **not an exact reproduction** of the original paper's OpenSearch field mappings or generation pipeline. The primary fusion combines existing BM25, SPLADE, and BGE-M3 rank lists.

## Folder layout
```text
Exp 6 BlendedRAG/
├── README.md
├── requirements.txt
├── run_blended.py
├── inputs/
│   ├── bm25_run.trec       # copied from Experiment 1
│   ├── splade_run.trec     # copied from Experiment 3
│   └── bge_m3_run.trec     # copied from Experiment 5
├── data/scidocs/          # official dataset, not provided as part of the code
└── results/blended/       # generated outputs
```
The three uploaded `.trec` files belong inside **`inputs/`**, with the exact filenames above. Do not place them directly beside `run_blended.py` unless you pass custom `--*-run` arguments.

## Setup (Ubuntu / conda)
```bash
conda create -n blended python=3.11 -y
conda activate blended
python -m pip install -r requirements.txt
```
This experiment can run on CPU; no new neural checkpoint downloads are required if input runs are available.

## Inputs
Use the **same SciDocs corpus and 1,000 test queries** as the other experiments. The script expects `data/scidocs/{corpus.jsonl,queries.jsonl,qrels/test.tsv}` and the three TREC files in `inputs/`.

## Run
```bash
python run_blended.py
```
Optional explicit paths and fusion settings:
```bash
python run_blended.py \
  --data data/scidocs \
  --bm25-run inputs/bm25_run.trec \
  --splade-run inputs/splade_run.trec \
  --bge-run inputs/bge_m3_run.trec \
  --candidate-k 100 --rrf-k 60 \
  --output results/blended
```
Use `--rrf-k 60` to match the experiment in the paper; do not tune this value on test judgments.

## Evaluated conditions
1. `bm25_best_fields` — separately score SciDocs title and abstract with BM25 and fuse the rankings; this is an **approximation**, not Elasticsearch `best_fields`.
2. `blended_three_rrf` — **equal-weight RRF of BM25 + SPLADE + BGE-M3** imported ranking lists.
3. `blended_field_aware_rrf` — RRF of title BM25, abstract BM25, and the three imported rank lists.

The implementation uses `RRF(d) = Σ 1/(60 + rank_i(d))`, and BEIR computes Precision@10, Recall@10, MRR@10, nDCG@10 and MAP@10.

## Outputs (`results/blended/`)
For each condition, the script writes `<condition>_run.trec` and `<condition>_metrics.json`. It also writes `experimental_protocol.json` containing configuration and source-run information.

## Important limitations
- Existing imported runs contain mostly **top-10** candidates per query; RRF cannot recover missing ranks 11–100. `--candidate-k 100` applies to the newly computed field-specific BM25 runs, **not** to the depth of imported TREC files.
- No test-ground-truth tuning. All three input ranking lists must cover the official query set.
- Field BM25 tokenization may differ from the original BM25 run.
- The script is an inspired adaptation/ablation, not the complete original Blended RAG system.

## References
- Original Blended RAG study: https://arxiv.org/abs/2404.07220
- Authors' code: https://github.com/ibm-self-serve-assets/Blended-RAG

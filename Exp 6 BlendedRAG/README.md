# EXP-05 — SciDocs field-aware hybrid retrieval (Blended RAG-inspired)

**Academic honesty:** Sawarkar et al. (2024), *Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers*, tests BM25, dense KNN, and sparse-encoder indexes with **match/multi-match, cross-fields, most-fields, best-fields and phrase-prefix** query configurations; it also tests a generator. The provided implementation is **not** an exact reproduction of the OpenSearch index/mapping/query configurations or the generative part. It uses SciDocs and adapts their field/query/index blending idea, alongside a **separate standard RRF fusion ablation**. The paper does not claim RRF is its core algorithm.

## Inputs — reuse existing experiment rankings

Copy the exact original `run.trec` files into the project's `inputs/` folder:
- EXP-00 BM25 → `inputs/bm25_run.trec`
- EXP-02 SPLADE → `inputs/splade_run.trec`
- EXP-04 BGE-M3 dense → `inputs/bge_m3_run.trec`

Copy the full existing **`data/scidocs`** directory (including `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) into this project. Do not substitute datasets. We use the same 1,000 official test queries and 25,657 docs.

Since the input TREC runs likely contain **only top 10 per query**, fusion cannot recover documents ranked 11–100 in those models. This is a *top-10 fusion ablation*, not a full-depth retrieval fusion. To strengthen the eventual paper, rerun individual retrieval models at candidate depth 100 or 1000 and replace the inputs.

## Setup

```bash
conda create -n blended python=3.11 -y
conda activate blended
python -m pip install -r requirements.txt
```

No GPU or new model downloads required: we reuse your previously computed retrieval rankings. Title/abstract BM25 runs on CPU; this keeps costs low.

## Run

```bash
python run_blended.py
```

## Outputs (`results/blended/`)

Three conditions, each with `*_run.trec` and `*_metrics.json`:

1. `bm25_best_fields`: title-only and abstract-only BM25 rankings fused by fixed RRF (paper-inspired field-selection **approximation**, not literal Elasticsearch `best_fields`).
2. `blended_three_rrf`: fixed equal-weight RRF of BM25, SPLADE, and BGE-M3 saved rankings (an additional fusion ablation, **not the paper's exact method**).
3. `blended_field_aware_rrf`: fixed equal-weight RRF of the five lists above (title, abstract, BM25, SPLADE, BGE-M3).

RRF score = sum(1 / (60 + rank)) per candidate. Equal weights and `k=60` are fixed beforehand, never tuned on the test qrels. The script directly evaluates with the official BEIR evaluator, saving P@10, Recall@10, MRR@10, nDCG@10, MAP@10.

**Methodology warning:** Query field variants BM25 use rank-bm25 default BM25Okapi with a simple unstemmed tokenizer; the original BM25 run may have a different tokenization. Comparisons should discuss these preprocessing differences. Timings exclude original embedding model training and encoding and are not full end-to-end latency measurements.

Original study: https://arxiv.org/abs/2404.07220
Authors' code: https://github.com/ibm-self-serve-assets/Blended-RAG

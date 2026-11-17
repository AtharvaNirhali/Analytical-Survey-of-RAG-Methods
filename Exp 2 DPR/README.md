# Experiment 2 — Dense Passage Retrieval (DPR) on BEIR SciDocs

> **Folder note:** The folder is named `Exp 2 DRP` in this repository; the retrieval model is **DPR** (Dense Passage Retrieval).

## Overview
Zero-shot DPR retrieval on the same SciDocs corpus and official test queries as Experiment 1. This uses Facebook's original pretrained Natural Questions **context encoder** and **question encoder**, without SciDocs fine-tuning.

- Context checkpoint: `facebook/dpr-ctx_encoder-single-nq-base`
- Question checkpoint: `facebook/dpr-question_encoder-single-nq-base`
- Ranking: exact dot product on unnormalized 768-dimensional representations.
- Metrics: Precision@10, Recall@10, MRR@10, nDCG@10, MAP@10.

## Layout
```text
Exp 2 DRP/
├── README.md
├── requirements.txt
├── run_dpr.py
├── verify_dpr.py
├── dpr_beir_verified.json   # optional small verification record
├── data/scidocs/           # downloaded automatically
└── results/dpr/           # generated
```

## Setup (Ubuntu / conda)
```bash
conda create -n dpr python=3.11 -y
conda activate dpr
# CUDA build if supported by your GPU driver:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
For CPU-only use, install PyTorch from the CPU wheel index and run with `--device cpu`. Internet is needed on the first run to download SciDocs and pretrained models.

## Run
```bash
# Smoke test: only 10 queries; corpus is still encoded or loaded in full
python run_dpr.py --limit-queries 10 --batch-size 4

# Full benchmark
python run_dpr.py --batch-size 4
```
On a 4 GB RTX 3050, try `--batch-size 2` or `--batch-size 1` if memory is insufficient; alternatively `--device cpu`. Context vectors are cached for future executions. The script processes the two encoders separately to reduce VRAM pressure.

## Outputs (`results/dpr/`)
- `metrics.json` — BEIR-style retrieval metrics and timings computed by the run script.
- `ranked_papers.csv` — retrieved papers for each query.
- `run.trec` — TREC rankings.
- `per_query_metrics.csv` — one metric row per query.
- `ctx_vectors_*.npy` — cached context vectors (**large; do not commit**).

## Independent BEIR verification — correction needed
`verify_dpr.py` is supplied, but its current `RUN_FILE` points to a **SPLADE directory**, not a DPR TREC file. Before using it, edit line 9 to:
```python
RUN_FILE = Path("results/dpr/run.trec")
```
Then install the BEIR evaluation dependency in this environment if needed (`python -m pip install 'beir>=2,<3'`) and execute:
```bash
python verify_dpr.py
```
It saves `dpr_beir_verified.json` in the current working directory. The supplied verification JSON is a prior result, not a replacement for running your current code.

## Protocol and limitations
- Pretrained DPR from question answering is transferred **zero-shot** to SciDocs paper-title queries.
- Context encoding uses `(title, text)`; question encoding uses the query string.
- `index_seconds_this_run` differs between first run and cached runs; query latency excludes corpus encoding.
- No external API, generation step, or test-set tuning.

## References
- DPR implementation: https://github.com/facebookresearch/DPR
- BEIR: https://github.com/beir-cellar/beir

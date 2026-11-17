# Experiment 5 — BGE-M3 Dense-Only Retrieval on BEIR SciDocs

## Overview
This experiment evaluates the **dense-only** retrieval mode of pretrained `BAAI/bge-m3` on the official SciDocs test split. It extracts normalized 1,024-dimensional CLS embeddings and ranks all documents by exact dot product. This is **not** an evaluation of BGE-M3's sparse or multi-vector modes.

## Layout
```text
Exp 5 BGE/
├── README.md
├── requirements.txt
├── run_bge_m3.py
├── data/scidocs/          # same official SciDocs data as other experiments
└── results/bge_m3/       # generated metrics, rankings, embedding cache
```

## Setup (Ubuntu / RTX 3050 4GB)
```bash
conda create -n bge_m3 python=3.11 -y
conda activate bge_m3
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```
Use an appropriate compatible PyTorch build if your system does not support the stated CUDA wheel. The checkpoint download is approximately 2.3 GB, so make sure sufficient disk space is available.

## Prepare data
Copy the same BEIR SciDocs dataset into `data/scidocs/` with `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`. The script does not download the dataset.

## Run
```bash
python run_bge_m3.py --batch-size 1 --doc-maxlen 256 --query-maxlen 64
```
If the GPU runs out of memory:
```bash
python run_bge_m3.py --device cpu --batch-size 1
```
To force a new corpus encoding rather than use saved vectors:
```bash
python run_bge_m3.py --reencode
```

## Outputs (`results/bge_m3/`)
- `metrics.json` — official BEIR metrics, latency, corpus encoding, run settings.
- `beir_verified.json` — the five core retrieval metrics.
- `run.trec` — top-10 TREC-format ranking, reused in Experiment 6.
- `ranked_papers.csv` — readable ranked papers.
- `doc_vectors_25657_maxlen256.npy` — cached normalized document vectors (**large**).
- `doc_ids.json` — row-to-document-ID mapping for the vectors.

## Evaluation and cross-experiment use
- No SciDocs training or tuning; uses official BEIR test qrels.
- The first run encodes all documents; later runs can reuse the `.npy` cache.
- Query timing covers query embedding + exact vector search, not corpus encoding.
- **Experiment 7 (HyDE) requires the `doc_vectors_25657_maxlen256.npy` and `doc_ids.json` files** from this experiment; share or regenerate them locally rather than committing large vectors to GitHub.

## References
- BGE-M3 model: https://huggingface.co/BAAI/bge-m3
- BEIR: https://github.com/beir-cellar/beir

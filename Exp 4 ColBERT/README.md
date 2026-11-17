# Experiment 4 — ColBERTv2 / PLAID on BEIR SciDocs

## Overview
This experiment performs **full-corpus, zero-shot ColBERTv2 retrieval** (not BM25 reranking) on SciDocs using the pretrained `colbert-ir/colbertv2.0` checkpoint and PLAID approximate indexing. It implements **ColBERTv2**, not the original ColBERTv1 from SIGIR 2020.

## Layout
```text
Exp 4 ColBERT/
├── README.md
├── run_colbert.py
├── verify_colbert.py
├── data/scidocs/               # copy official SciDocs dataset
├── results/colbert/           # generated files
├── colbert_indexes/           # generated large indexes; do not commit
└── __pycache__/               # generated .pyc files; do not commit
```
The uploaded `*.cpython-313.pyc` files are Python bytecode caches. **You do not need them to reproduce the experiment**; they can be excluded from Git with `__pycache__/` and `*.pyc`.

## Setup (Ubuntu / conda)
```bash
conda create -n colbert python=3.10 -y
conda activate colbert
python -m pip install 'numpy<2' 'transformers>=4.35,<5' 'huggingface-hub<1'
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'colbert-ai==0.2.22' 'transformers>=4.35,<5' 'huggingface-hub<1'
python -m pip check
```
Use a PyTorch CUDA build supported by your NVIDIA driver. Dependency resolution for ColBERT may require development tools. There is no `requirements.txt` among the files you supplied for this folder; the installation commands above come from your existing README.

## Dataset
Copy `data/scidocs/corpus.jsonl`, `data/scidocs/queries.jsonl`, and `data/scidocs/qrels/test.tsv` from the other experiments. The script checks that these files exist and **does not** download them.

## Run
```bash
python run_colbert.py --batch-size 2 --doc-maxlen 180
```
For an existing **complete** PLAID index, you may reuse it with:
```bash
python run_colbert.py --skip-index
```
On a 4 GB GPU, indexing may exhaust VRAM; reducing to `--batch-size 1 --doc-maxlen 128` may help, but **changes the evaluation setting** and must be reported. Default index compression is `--nbits 2`; default query length is 32.

## Verify results with BEIR
Official BEIR scoring is separate from the ColBERT environment. Activate an environment with `beir` installed (for example, your DPR environment after installing BEIR), then:
```bash
python verify_colbert.py
```
The verification script reads `results/colbert/run.trec` and writes `results/colbert/beir_verified.json`.

## Outputs (`results/colbert/`)
- `run.trec` — top-10 SciDocs ranks, used by verifier.
- `ranked_papers.csv` — query/document IDs, rank, score.
- `doc_ids.json` — internal PLAID passage IDs mapped to SciDocs IDs.
- `collection.tsv` — title+body corpus converted for indexing.
- `metrics.json` — run configuration and timing (BEIR scores are separate).
- `beir_verified.json` — official metrics after `verify_colbert.py`.

## Methodological notes
- Uses a pretrained MS MARCO-oriented checkpoint with no SciDocs fine-tuning.
- Search uses PLAID indexing (not exhaustive MaxSim).
- Query latency excludes corpus/index-building time.
- Do not commit large indexes or generated collection files; keep the script and README.

## References
- ColBERT repository: https://github.com/stanford-futuredata/ColBERT
- BEIR: https://github.com/beir-cellar/beir

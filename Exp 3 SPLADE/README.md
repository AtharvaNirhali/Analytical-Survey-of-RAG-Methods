# Experiment 3 — SPLADE++ Sparse Retrieval on BEIR SciDocs

## Overview
This experiment tests a pretrained **SPLADE++** sparse retrieval model on the SciDocs test set. It expands queries and documents into weighted vocabulary representations and ranks documents using exact sparse dot products. It does **not** fine-tune the model on SciDocs.

**Checkpoint:** `naver/splade-cocondenser-ensembledistil`  
**Core API:** `sentence_transformers.SparseEncoder`  
**Evaluation:** Official BEIR `EvaluateRetrieval` at top 10.

## Layout
```text
Exp 3 SPLADE/
├── README.md
├── requirements.txt
├── run_splade.py
├── verify_splade.py
├── data/scidocs/          # supply the BEIR SciDocs files
└── results/splade/       # generated outputs and cached vectors
```
The dataset must include `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. Unlike BM25/DPR, this runner does **not** automatically download SciDocs; reuse the same dataset from Experiment 1 or 2.

## Setup (Ubuntu / conda)
```bash
conda create -n splade python=3.11 -y
conda activate splade
# Install a compatible PyTorch build for your system first.
python -m pip install -r requirements.txt
```
`run_splade.py` imports PyTorch; `sentence-transformers` ordinarily brings it as a dependency. Verify CUDA availability with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run
```bash
python run_splade.py --batch-size 2 --max-length 256
```
If CUDA runs out of memory, use:
```bash
python run_splade.py --device cpu --batch-size 2
```
The first run downloads the checkpoint, encodes the corpus, and saves a sparse vector cache; subsequent runs can reuse the cache.

## Outputs (`results/splade/`)
- `metrics.json` — official BEIR metrics and runtime details.
- `run.trec` — top-10 ranked documents for every test query.
- `ranked_papers.csv` — query/title/rank/score table.
- `doc_vectors_*.npz` — cached sparse corpus vectors (**do not commit if large**).
- `doc_ids.json` — document ID order associated with the vector cache.

## Independent BEIR verification — correction needed
The supplied `verify_splade.py` correctly points at `results/splade/run.trec`, but it still contains copied DPR labels and saves its output as **`dpr_beir_verified.json`**. Before running, change its final output filename to `splade_beir_verified.json` (and optionally update the print messages), then run:
```bash
python verify_splade.py
```
The uploaded `dpr_beir_verified(1).json` contains SPLADE scores despite its DPR-looking filename; rename it descriptively if retaining it.

## Protocol and limitations
- Top-10 precision, recall, MRR, nDCG, MAP from official SciDocs judgments.
- `--max-length` defaults to 256; batch size 2; CUDA selected if available.
- Query latency includes query encoding plus exact sparse search; **excludes corpus encoding**.
- Cache is reusable only with the corresponding model, token limit, and document ordering.

## References
- SPLADE model checkpoint: https://huggingface.co/naver/splade-cocondenser-ensembledistil
- BEIR: https://github.com/beir-cellar/beir

# Experiment 7 — HyDE-Inspired Retrieval on BEIR SciDocs

## Overview
This experiment adapts **Hypothetical Document Embeddings (HyDE)** to SciDocs paper-title queries. A locally hosted **Ollama Llama 3.2 1B** model generates a plausible abstract for a *related paper*. The resulting text is embedded with `BAAI/bge-m3`, and the embedding is used to retrieve real SciDocs publications.

This is **not an exact reproduction** of Gao et al.'s original HyDE configuration (which used different LLM/retriever models). The controlled baseline is **Experiment 5 BGE-M3 dense-only**; the HyDE experiment reuses the same cached SciDocs document embeddings and alters the query representation.

## Layout
```text
Exp 7 HyDE/
├── README.md
├── requirements.txt
├── run_hyde.py
├── data/scidocs/                         # official SciDocs files
├── inputs/bge_m3/
│   ├── doc_ids.json                      # from Exp 5 BGE
│   └── doc_vectors_25657_maxlen256.npy   # from Exp 5 BGE; large, do not commit
└── results/hyde/                        # generated results/cache
```

## Setup (Ubuntu / conda)
```bash
conda create -n hyde python=3.11 -y
conda activate hyde
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```
Install Ollama separately, then:
```bash
ollama pull llama3.2:1b
ollama list
```
If the Ollama service is not running, start it in another terminal using `ollama serve` (systems with an existing service may not need this).

## Required inputs
- The same official SciDocs `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` as the other experiments.
- `inputs/bge_m3/doc_ids.json` and `inputs/bge_m3/doc_vectors_25657_maxlen256.npy` copied **unchanged** from Experiment 5's `results/bge_m3/`.

The script validates IDs, vector dimensions, and normalization. Do not substitute ColBERT/DPR embeddings.

## Run
```bash
# Smoke test, five queries only
python run_hyde.py --limit 5

# Complete 1,000-query experiment
python run_hyde.py
```
If the BGE encoder and Ollama contend for your 4 GB VRAM:
```bash
python run_hyde.py --device cpu
```
Optional ablations (use separate output folders):
```bash
python run_hyde.py --output-dir results/hyde_query_mix --mix-original
python run_hyde.py --output-dir results/hyde_3samples --hypotheticals 3
```
The standard run uses one hypothetical per query with temperature 0. `--regenerate` replaces the generation cache; this can change results.

## Outputs (`results/hyde/`)
- `hypothetical_documents.jsonl` — generated abstracts cached incrementally per query/sample, so an interrupted generation can resume.
- `run.trec` — top-10 SciDocs ranking.
- `ranked_papers.csv` — ranked documents with titles.
- `metrics.json` and `beir_verified.json` — official BEIR scores and experiment settings.
- `protocol.json` — prompt, LLM, encoder, and evaluation protocol.

## Methodology and limitations
- LLM generation does not use test relevance judgments.
- BGE-M3 document vectors use maximum document length 256; the hypothetical text uses `--hyde-maxlen 256` by default.
- Latency measurements differ **significantly** when the generation cache is reused. Do not present cached latency as fresh LLM generation latency.
- A smoke test is not the full benchmark. The experiment tests a SciDocs adaptation, not the original HyDE system.

## References
- HyDE paper: https://arxiv.org/abs/2212.10496
- BGE-M3 model: https://huggingface.co/BAAI/bge-m3
- BEIR: https://github.com/beir-cellar/beir

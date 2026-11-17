# EXP-06 — HyDE on BEIR SciDocs (BGE-M3 encoder)

## What it tests
HyDE (Gao et al., ACL 2023) generates hypothetical documents with an LLM and retrieves real papers by encoding the hypothetical text. This is a **SciDocs-adapted HyDE experiment**, **not an exact replication** of the original HyDE work (which used InstructGPT + Contriever). On SciDocs, queries are paper titles, so the prompt generates an abstract of a *potentially related paper*, rather than pretending the title is a question.

**Controlled baseline:** EXP-04 BGE-M3 dense-only = nDCG@10 **0.16253**. This experiment reuses its cached BGE-M3 document vectors and changes only the query representation. Results may improve or worsen; report honestly.

## Hardware
Dell G15, RTX 3050 4GB VRAM + local Ollama Llama 3.2 1B. The encoder and Ollama might compete for GPU memory. You can run encoder on CPU (`--device cpu`); Ollama may use GPU separately. Ollama requires installation outside pip requirements.

## Setup
```bash
conda create -n hyde python=3.11 -y
conda activate hyde
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
ollama list
```
Start Ollama service if not running: `ollama serve` (another terminal). Some systems already run Ollama as a service.

## Folders
```
HyDE_SciDocs_Experiment/
  run_hyde.py
  requirements.txt
  README.md
  data/scidocs/{corpus.jsonl,queries.jsonl,qrels/test.tsv}
  inputs/bge_m3/doc_ids.json
  inputs/bge_m3/doc_vectors_25657_maxlen256.npy
```
Copy both cached files **from EXP-04 BGE-M3 results**, without modifying them. The script checks exact ID order, corpus size, embedding size and normalization. Do not use the ColBERT doc IDs or vectors.

## Run
A smoke test (NOT final result):
```bash
python run_hyde.py --limit 5
```
Full 1,000-query experiment (generation cache resumes automatically):
```bash
python run_hyde.py
```
If GPU out-of-memory:
```bash
python run_hyde.py --device cpu
```
If Ollama model tag differs, e.g. `llama3.2:latest`, pass `--llm llama3.2:latest`. To regenerate cache, pass `--regenerate`; doing so can change results.

### Optional ablations — run in **different** output directories
```bash
python run_hyde.py --output-dir results/hyde_query_mix --mix-original
python run_hyde.py --output-dir results/hyde_3samples --hypotheticals 3
```
Notice: different output directories cause new generations; to make a controlled comparison, reuse the same hypothetical cache where appropriate and document generation settings.

## Outputs
- `results/hyde/hypothetical_documents.jsonl` — cached generated texts, one entry per query/sample, resumable
- `results/hyde/run.trec` — top-10 per query
- `results/hyde/ranked_papers.csv` — top 10 with titles
- `results/hyde/metrics.json` — BEIR metrics + setup/latency
- `results/hyde/beir_verified.json` — exact official BEIR metric subset
- `results/hyde/protocol.json` — saved prompt and experiment protocol

## Evaluation fairness
- Same SciDocs corpus, test qrels, top-10 cutoff and dense document vectors as EXP-04.
- LLM generation does not use qrels, answers, or test labels.
- Do not compare query latency from cached runs against EXP-04; cache hits skip expensive generation.
- The primary run is **one** hypothetical text per query, deterministic temperature 0. The original HyDE framework often averages multiple generations.
- The cached document embeddings were created using max document length 256 and normalized CLS of BGE-M3.
- HyDE's generation is adapted to SciDocs's title/citation nature, which may or may not benefit from hypothetical abstracts.

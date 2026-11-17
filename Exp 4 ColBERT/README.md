# Experiment 03 — ColBERTv2 / PLAID (zero-shot) on BEIR SciDocs

**Methodological note:** This implements **ColBERTv2** (2021/2022), not the original ColBERT (SIGIR 2020). Record this distinction in the survey. Uses authors' checkpoint and index engine, **full corpus retrieval**, not a BM25 reranker.

Copy the **same** `data/scidocs` folder used for BM25/DPR/SPLADE, with `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`.

## Installation

```bash
conda create -n colbert python=3.10 -y
conda activate colbert
python -m pip install 'numpy<2' 'transformers>=4.35,<5' 'huggingface-hub<1'
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'colbert-ai==0.2.22' 'transformers>=4.35,<5' 'huggingface-hub<1'
python -m pip check
```

ColBERT packaging changes over time; if dependency resolution conflicts, share the full error. The official ColBERT backend may require additional build tools. **4 GB GPU caution:** Full ColBERTv2 indexing can exceed available VRAM; start with small batch size and 180-token passage cap and do not claim a full benchmark from a truncated corpus. Indexing and disk/RAM use can be substantially greater than BM25/DPR/SPLADE.

## Run

```bash
python run_colbert.py --batch-size 2 --doc-maxlen 180
```

If CUDA OOM during indexing, try `--batch-size 1 --doc-maxlen 128`; keep this different document truncation in your experimental configuration. For a fully apples-to-apples comparison also consider the ColBERT passage-vs-paper input-length limitation. No guarantee that the official PLAID indexer fits 4GB GPU.

## Evaluate independently

Use your existing `dpr` environment with BEIR installed:

```bash
conda activate dpr
python verify_colbert.py
```

Save `results/colbert/beir_verified.json` and `results/colbert/metrics.json`. The BEIR metric output is the authoritative evaluation result; the run code deliberately does not import BEIR in the ColBERT environment.

## Outputs
- `results/colbert/run.trec`
- `results/colbert/ranked_papers.csv`
- `results/colbert/doc_ids.json`
- `results/colbert/collection.tsv`
- `results/colbert/metrics.json`
- `results/colbert/beir_verified.json` (after verification)

## Fairness and scope
- The **SciDocs test set is untouched** and used solely for evaluation.
- The checkpoint was pretrained on MS MARCO (out-of-domain zero-shot testing).
- Encoding/index time is not included in query latency.
- PLAID approximate index search differs from exhaustive ColBERT MaxSim; document the retrieval engine as PLAID and compression setting `nbits=2`.

# Experiment 4 — BGE-M3 (dense-only) on SciDocs

This is **zero-shot BGE-M3 dense retrieval**, not all three BGE-M3 modes. It is intentionally separated from the sparse and multi-vector modes so improvements can be attributed to one retrieval algorithm. Uses the BGE-M3 normalized CLS embedding and exact dot product. BEIR evaluation is **integrated**.

## Setup (Ubuntu / RTX 3050 4GB)

```bash
conda create -n bge_m3 python=3.11 -y
conda activate bge_m3
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Put the same SciDocs dataset in `data/scidocs/{corpus.jsonl,queries.jsonl,qrels/test.tsv}`.

```bash
python run_bge_m3.py --batch-size 1 --doc-maxlen 256 --query-maxlen 64
```

If your 4GB GPU runs out of memory, try `--device cpu` (much slower). Do not change the maxlen between methods without documenting the difference; note previous experiments used different limits. `--reencode` forces regeneration of cached document vectors.

The model is large (~2.3 GB on disk). Reserve at least several GB of free space for downloads, caches, and runtime.

Outputs:
- `results/bge_m3/metrics.json` — includes official BEIR metrics and timing
- `results/bge_m3/beir_verified.json` — five verified metrics
- `results/bge_m3/run.trec` — qid/Q0/docid/rank/score/run-name
- `results/bge_m3/ranked_papers.csv` — readable ranked papers
- `results/bge_m3/doc_vectors_25657_maxlen256.npy` — reusable dense corpus embeddings
- `results/bge_m3/doc_ids.json` — matrix row → document identifier map

**Evaluation caveat:** Exact dense retrieval measures BGE-M3's dense mode only. The BGE-M3 paper describes dense, learned sparse, and ColBERT-like token-level modes; a hybrid BGE-M3 experiment should be conducted separately. There is no test-label training or parameter tuning in this package.

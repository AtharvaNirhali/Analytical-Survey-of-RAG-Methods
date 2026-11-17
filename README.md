# Scientific Literature Retrieval — Seven-Experiment Benchmark

**A reproducible analytical comparison of sparse, dense, late-interaction, hybrid, and query-expansion retrieval approaches on BEIR SciDocs.**

This repository contains seven experiments comparing retrieval methods for scientific literature discovery. All reported results use the **BEIR SciDocs test split** (25,657 candidate documents, 1,000 evaluation queries), with rankings evaluated at **k = 10**. This is a *retrieval* benchmark: it does not evaluate downstream text generation, factuality, or full end-to-end RAG answers.

## Experiments

| Folder | Method | Retrieval configuration |
|---|---|---|
| [`Exp 1 BM25/`](./Exp%201%20BM25/) | BM25 | BM25Okapi, lowercased unstemmed word tokenization |
| [`Exp 2 DRP/`](./Exp%202%20DRP/) | DPR | Pretrained Facebook DPR (Natural Questions), exact dense dot product |
| [`Exp 3 SPLADE/`](./Exp%203%20SPLADE/) | SPLADE | `naver/splade-cocondenser-ensembledistil`, sparse dot product |
| [`Exp 4 ColBERT/`](./Exp%204%20ColBERT/) | ColBERTv2 | `colbert-ir/colbertv2.0`, PLAID retrieval (`nbits=2`) |
| [`Exp 5 BGE/`](./Exp%205%20BGE/) | BGE-M3 | `BAAI/bge-m3`, **dense-only** normalized CLS vectors |
| [`Exp 6 BlendedRAG/`](./Exp%206%20BlendedRAG/) | Hybrid / Blended RAG-inspired | Field-aware BM25 variations and fixed equal-weight reciprocal rank fusion (RRF) |
| [`Exp 7 HyDE/`](./Exp%207%20HyDE/) | HyDE-inspired | Local Llama 3.2 1B hypothetical related-paper abstracts; BGE-M3 dense retrieval |

> **Folder naming:** `Exp 2 DRP` preserves the supplied folder name; the implemented method is **DPR** (Dense Passage Retrieval). Internal script labels `EXP-00` through `EXP-06` are zero-based, whereas GitHub folders are numbered 1–7.

## Results — BEIR SciDocs test set

All effectiveness values below come from the uploaded experiment result files; **higher is better** for each measure.

| Method / run | P@10 | Recall@10 | MRR@10 | nDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.07630 | 0.15448 | 0.26797 | 0.14960 | 0.08696 |
| DPR | 0.03640 | 0.07403 | 0.13819 | 0.07040 | 0.03620 |
| SPLADE | 0.08150 | 0.16538 | 0.28076 | 0.15856 | 0.09245 |
| ColBERTv2 / PLAID | 0.07770 | 0.15728 | 0.27582 | 0.15256 | 0.08826 |
| BGE-M3 (dense only) | 0.08350 | 0.16922 | 0.28942 | 0.16253 | 0.09425 |
| **BM25 + SPLADE + BGE-M3 (RRF)** | **0.08950** | **0.18133** | **0.30655** | **0.17563** | **0.10431** |
| Field-aware five-list RRF | 0.08670 | 0.17582 | 0.29562 | 0.16941 | 0.10014 |
| BM25 title/abstract field fusion | 0.07190 | 0.14598 | 0.24978 | 0.13942 | 0.07932 |
| HyDE-inspired + BGE-M3 | 0.06260 | 0.12702 | 0.20943 | 0.11819 | 0.06585 |

**Main finding:** In these runs, the fixed three-way **BM25 + SPLADE + BGE-M3 RRF** system achieved the highest nDCG@10 (**0.17563**) and all four other reported top-10 effectiveness metrics. This observation is specific to the benchmark, models, input depth, and configurations used here; it is not a general claim that RRF always wins.

## Dataset and common protocol

- **Benchmark:** [BEIR](https://github.com/beir-cellar/beir) — SciDocs, **test** split.
- **Corpus:** 25,657 scientific records; **queries:** 1,000 test queries with relevance judgments.
- **Files expected by the scripts:** `data/scidocs/corpus.jsonl`, `data/scidocs/queries.jsonl`, `data/scidocs/qrels/test.tsv`.
- **Cutoff:** top 10; evaluation metrics are Precision@10, Recall@10, MRR@10, nDCG@10 and MAP@10.
- **No SciDocs test-qrel fitting:** pretrained checkpoints are used zero-shot and fusion parameters are fixed, not tuned against test labels.

To obtain SciDocs, follow the individual BM25 README (which can download the dataset) or use the BEIR dataset distribution. The dataset itself is **not included** in the repository.

## How to reproduce

Each experiment has a **separate `README.md` and `requirements.txt`**. Run commands from **within the corresponding experiment folder**, with the SciDocs dataset at the location expected by that script. The experiments use independent conda environments because their PyTorch, tokenizer, and retrieval dependencies differ.

Suggested sequence:

1. **BM25:** establish and reuse a local `data/scidocs` directory.
2. **DPR, SPLADE, ColBERTv2, BGE-M3:** execute individually and retain each experiment's `results/.../run.trec`.
3. **Blended RAG-inspired / RRF:** copy the existing **BM25**, **SPLADE**, and **BGE-M3** TREC files into its `inputs/` folder as `bm25_run.trec`, `splade_run.trec`, and `bge_m3_run.trec`; then execute `run_blended.py`.
4. **HyDE:** reuse BGE-M3's `doc_ids.json` and `doc_vectors_25657_maxlen256.npy` in `inputs/bge_m3/`; start local Ollama with `llama3.2:1b`, then execute `run_hyde.py`.

Example invocation (from the repository root; this command changes directories so the relative paths resolve):

```bash
cd 'Exp 1 BM25'
python run_bm25.py --limit-queries 10  # smoke test only
python run_bm25.py                     # full benchmark
```

See individual README files for environment creation, exact commands, CUDA/CPU options, dataset placement, caches, verification, and expected outputs. A smoke test is **not** a full benchmark.

## Repository layout

```text
.
├── README.md                 # this page
├── LICENSE                   # MIT license for original repository code
├── .gitignore
├── Exp 1 BM25/
├── Exp 2 DRP/                # DPR implementation
├── Exp 3 SPLADE/
├── Exp 4 ColBERT/
├── Exp 5 BGE/
├── Exp 6 BlendedRAG/
│   └── inputs/               # source run.trec files used for fusion
└── Exp 7 HyDE/
    └── inputs/bge_m3/        # BGE-M3 vector cache used for HyDE
```

Each experiment has its own scripts, setup instructions, and output structure. Small JSON metric files and experimental protocol files are suitable for version control; keep large embedding caches and downloaded model weights out of Git.

## Interpreting the comparison

- **Different retrieval implementations:** ColBERTv2 uses PLAID approximate indexing; other models use exact dense or sparse scoring. Token and document length limits vary (e.g., ColBERTv2 180 tokens versus BGE-M3 256 tokens). Performance comparisons are therefore **configuration-specific**, not perfectly controlled architectural ablations.
- **Blended RAG-inspired is not an exact paper reproduction:** The field variations are approximate, and the RRF runs are separate fusion ablations. The source TREC inputs contain top-10 rankings; the three-way RRF therefore fuses **shallow candidate lists** rather than full-depth retrieval outputs. The recorded `candidate_k=100` affects the additional field-specific BM25 lists, not the depth of the saved source runs.
- **BGE-M3** measures only the model's **dense** mode, not its native sparse or multi-vector capabilities.
- **HyDE** adapts hypothetical-document generation to SciDocs paper-title queries using local Llama 3.2 1B and BGE-M3; it is not a literal replication of the original HyDE model stack.
- **Timings are not directly comparable as end-to-end costs:** Reported query latency generally excludes corpus indexing/encoding; HyDE may include LLM generation and is affected by generation-cache hits. ColBERT's reported `index_seconds_this_run` may reflect index reuse.

## Reproducibility and verification

- Retain the complete `run.trec` outputs (standard TREC format) and record model checkpoints, maximum input lengths, hardware, and versions.
- Where available, use `beir_verified.json` as the independently/equivalently BEIR-scored metric record. BM25, DPR, SPLADE and several other methods also report metrics in `metrics.json`.
- Check local paths before running verification scripts: the originally supplied `verify_dpr.py` referred to a SPLADE location, and `verify_splade.py` used a DPR-style verification filename. These naming/path issues do **not** change the uploaded reported metrics, but could affect a fresh verification run.
- For closer scientific comparisons, consider additional datasets, uniform truncation rules, deeper fusion inputs (top 100–1,000), repeated timings, and statistical significance testing.

## Citation and attribution

If you use this repository, cite the original method papers and the **BEIR** benchmark in your work. The experiment-specific README files identify important checkpoints and research references. No GitHub URL, manuscript DOI, or author affiliation has been assumed here; add your repository link and paper citation once published.

## License

The **original code and documentation in this repository** are offered under the [MIT License](./LICENSE), subject to the rights of the applicable copyright holders. **BEIR/SciDocs data, pretrained weights, and third-party packages** retain their own licenses and terms. See `LICENSE` and each dependency's license before redistribution.

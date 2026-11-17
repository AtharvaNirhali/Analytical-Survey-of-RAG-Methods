# EXP-01 — Dense Passage Retrieval (DPR) on BEIR SciDocs

This uses the **original DPR architecture** through its official Facebook pretrained `single-nq-base` context and question encoders, distributed on Hugging Face. It evaluates SciDocs with no retraining and exactly the same official test queries/qrels as EXP-00.

## Environment (Ubuntu + conda)

```bash
conda create -n dpr python=3.11 -y
conda activate dpr
# Choose CUDA 12.1 wheels if supported by your NVIDIA driver, or use the default PyTorch installation.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:',torch.__version__,'cuda:',torch.cuda.is_available(),'transformers:',transformers.__version__)"
```

If CUDA cannot be used, install CPU PyTorch (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`) and run with `--device cpu`. You need internet for the first dataset and pretrained-model download.

## Run

```bash
# Smoke test, 10 queries; documents still encoded in full (or loaded from cache).
python run_dpr.py --limit-queries 10 --batch-size 4

# Official full test split
python run_dpr.py --batch-size 4
```

Start at batch size 4 on an RTX 3050 4GB, reduce to `--batch-size 2` or `--batch-size 1` if needed. The context encoder is freed before question encoding to avoid both encoder models occupying VRAM at once. Context embeddings are cached and reused on later runs. If CUDA memory errors persist use `--device cpu`.

## Outputs

`results/dpr/metrics.json`, `ranked_papers.csv`, `run.trec`, `per_query_metrics.csv`, and cached `ctx_vectors_*.npy`.

## Protocol

- Exact dot product on unnormalized 768-dimensional DPR representations.
- DPR context tokenizer uses `(title, text)` inputs; question encoder uses the query text.
- No SciDocs tuning, no external API, no generator.
- Evaluated metrics are averages over evaluated test queries; full results only when `--limit-queries` is not set.
- `index_seconds_this_run` includes corpus encoding + saving on first run, but only cache loading on subsequent runs. Compare first-run and cached timing separately.
- `mean_query_latency_ms` measures query encoding plus exhaustive search; it does not include corpus encoding.
- This is the **pretrained DPR retrieval model**, not a replication of the original paper's full Wikipedia QA experiment.

Official paper/code: https://github.com/facebookresearch/DPR
Pretrained checkpoints: https://huggingface.co/facebook/dpr-question_encoder-single-nq-base and https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
BEIR: https://github.com/beir-cellar/beir

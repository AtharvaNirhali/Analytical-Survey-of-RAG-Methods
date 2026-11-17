import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Change these paths if your directory structure differs.
DATA_DIR = Path("data/scidocs")
RUN_FILE = Path("results/splade/run.trec")

# Load official SciDocs relevance judgments.
_, queries, qrels = GenericDataLoader(
    data_folder=str(DATA_DIR)
).load(split="test")

# Read previously generated DPR rankings.
results = {}

with open(RUN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if not parts:
            continue

        qid, _, docid, rank, score, run_name = parts[:6]

        results.setdefault(qid, {})[docid] = float(score)

# Validate query coverage.
missing_queries = set(qrels) - set(results)

print("Expected queries:", len(qrels))
print("Queries in DPR results:", len(results))
print("Missing queries:", len(missing_queries))

# Evaluate using BEIR.
k_values = [10]

ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
    qrels, results, k_values
)

mrr = EvaluateRetrieval.evaluate_custom(
    qrels, results, k_values, metric="mrr"
)

verified_metrics = {
    "precision_at_10": precision["P@10"],
    "recall_at_10": recall["Recall@10"],
    "mrr_at_10": mrr["MRR@10"],
    "ndcg_at_10": ndcg["NDCG@10"],
    "map_at_10": _map["MAP@10"],
}

print("\nOFFICIAL BEIR RESULTS")
print(json.dumps(verified_metrics, indent=4))

with open("dpr_beir_verified.json", "w") as f:
    json.dump(verified_metrics, f, indent=4)

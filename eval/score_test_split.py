"""
Score all 5 methods (bm25/dense/hybrid/rerank/ltr) on ONLY the queries
held out for LTR testing -- comparing LTR against the other methods on
data it never trained on. Same NDCG@10/MRR functions as eval/score.py.

Usage:
    python -m eval.score_test_split
"""
import json
import math
from pathlib import Path

CANDIDATES_PATH = Path(__file__).parent / "candidates.json"
TEST_PREDICTIONS_PATH = Path(__file__).parent / "ltr_test_predictions.json"
BASE_METHODS = ["bm25", "dense", "hybrid", "rerank"]
K = 10


def dcg(relevances: list[int]) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
    )


def ndcg_at_k(ranked_labels: list[int], k: int = K) -> float:
    actual = dcg(ranked_labels[:k])
    ideal = dcg(sorted(ranked_labels, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def mrr(ranked_labels: list[int], threshold: int = 1) -> float:
    for rank, rel in enumerate(ranked_labels, start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


def main():
    candidates_data = json.loads(CANDIDATES_PATH.read_text())
    test_predictions = json.loads(TEST_PREDICTIONS_PATH.read_text())
    test_queries = list(test_predictions.keys())

    all_methods = BASE_METHODS + ["ltr"]
    scores = {m: {"ndcg": [], "mrr1": [], "mrr2": []} for m in all_methods}

    for query in test_queries:
        entry = candidates_data[query]
        label_by_id = {c["id"]: c["label"] for c in entry["candidates"]}

        method_rankings = dict(entry["results"])  # bm25/dense/hybrid/rerank
        method_rankings["ltr"] = test_predictions[query]

        for method in all_methods:
            ranked_ids = method_rankings.get(method, [])[:K]
            ranked_labels = [label_by_id.get(pid, 0) for pid in ranked_ids]
            scores[method]["ndcg"].append(ndcg_at_k(ranked_labels))
            scores[method]["mrr1"].append(mrr(ranked_labels, threshold=1))
            scores[method]["mrr2"].append(mrr(ranked_labels, threshold=2))

    print(f"Scored on {len(test_queries)} held-out test queries "
          f"(LTR never trained on these)\n")
    print(f"{'Method':<10} {'NDCG@10':>10} {'MRR(>=1)':>10} {'MRR(>=2)':>10}")
    print("-" * 44)
    for method in all_methods:
        avg_ndcg = sum(scores[method]["ndcg"]) / len(scores[method]["ndcg"])
        avg_mrr1 = sum(scores[method]["mrr1"]) / len(scores[method]["mrr1"])
        avg_mrr2 = sum(scores[method]["mrr2"]) / len(scores[method]["mrr2"])
        print(f"{method:<10} {avg_ndcg:>10.4f} {avg_mrr1:>10.4f} {avg_mrr2:>10.4f}")


if __name__ == "__main__":
    main()
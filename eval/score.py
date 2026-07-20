"""
Score BM25 / dense / hybrid / rerank against hand-labeled relevance
judgments in eval/candidates.json, using NDCG@10 and MRR.

Requires candidates.json to have been labeled (label field != None)
for a meaningful score -- unlabeled pairs are treated as irrelevant
(0) with a warning, since an unscored gap shouldn't silently vanish.

Usage:
    python -m eval.score
"""
import json
import math
from pathlib import Path

CANDIDATES_PATH = Path(__file__).parent / "candidates.json"
METHODS = ["bm25", "dense", "hybrid", "rerank"]
K = 10


def dcg(relevances: list[int]) -> float:
    """Discounted Cumulative Gain: sum of (2^rel - 1) / log2(rank + 1),
    rank starting at 1. The log-discount means relevance further down
    the list contributes less."""
    return sum(
        (2 ** rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
    )


def ndcg_at_k(ranked_labels: list[int], k: int = K) -> float:
    """NDCG = actual DCG / ideal DCG (DCG if results were perfectly
    sorted by relevance). Normalizes so scores are comparable across
    queries regardless of how many relevant docs exist for each."""
    actual = dcg(ranked_labels[:k])
    ideal = dcg(sorted(ranked_labels, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def mrr(ranked_labels: list[int]) -> float:
    """Reciprocal rank of the first relevant (label >= 1) result."""
    for rank, rel in enumerate(ranked_labels, start=1):
        if rel >= 1:
            return 1.0 / rank
    return 0.0


def main():
    data = json.loads(CANDIDATES_PATH.read_text())

    scores = {method: {"ndcg": [], "mrr": []} for method in METHODS}
    unlabeled_count = 0

    for query, entry in data.items():
        label_by_id = {c["id"]: c["label"] for c in entry["candidates"]}

        for method in METHODS:
            ranked_ids = entry["results"].get(method, [])
            ranked_labels = []
            for paper_id in ranked_ids:
                label = label_by_id.get(paper_id)
                if label is None:
                    unlabeled_count += 1
                    label = 0  # treat unlabeled as irrelevant, but track how often this happens
                ranked_labels.append(label)

            scores[method]["ndcg"].append(ndcg_at_k(ranked_labels))
            scores[method]["mrr"].append(mrr(ranked_labels))

    if unlabeled_count:
        print(f"WARNING: {unlabeled_count} query-paper pairs are unlabeled "
              f"(treated as irrelevant). Label them for accurate scores.\n")

    print(f"{'Method':<10} {'NDCG@10':>10} {'MRR':>10}")
    print("-" * 32)
    for method in METHODS:
        avg_ndcg = sum(scores[method]["ndcg"]) / len(scores[method]["ndcg"])
        avg_mrr = sum(scores[method]["mrr"]) / len(scores[method]["mrr"])
        print(f"{method:<10} {avg_ndcg:>10.4f} {avg_mrr:>10.4f}")


if __name__ == "__main__":
    main()
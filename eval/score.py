"""
Score BM25 / dense / hybrid / rerank against hand-labeled relevance
judgments in eval/candidates.json, using NDCG@10 and MRR.

MRR is computed at two thresholds:
  - MRR (label>=1): "somewhat relevant or better" -- prone to saturating
    near 1.0 if most labels are >=1, as in this corpus.
  - MRR (label>=2): "highly relevant" -- stricter, more likely to show
    real differences between methods without needing to relabel data.

Usage:
    python -m eval.score
"""
import json
import math
from pathlib import Path

CANDIDATES_PATH = Path(__file__).parent / "candidates.json"
METHODS = ["bm25", "dense", "hybrid", "rerank", "late_interaction", "semantic_id", "generative"]
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
    """Reciprocal rank of the first result with label >= threshold."""
    for rank, rel in enumerate(ranked_labels, start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


def main():
    data = json.loads(CANDIDATES_PATH.read_text())

    scores = {method: {"ndcg": [], "mrr1": [], "mrr2": []} for method in METHODS}
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
                    label = 0
                ranked_labels.append(label)

            scores[method]["ndcg"].append(ndcg_at_k(ranked_labels))
            scores[method]["mrr1"].append(mrr(ranked_labels, threshold=1))
            scores[method]["mrr2"].append(mrr(ranked_labels, threshold=2))

    if unlabeled_count:
        print(f"WARNING: {unlabeled_count} unlabeled pairs treated as irrelevant.\n")

    print(f"{'Method':<10} {'NDCG@10':>10} {'MRR(>=1)':>10} {'MRR(>=2)':>10}")
    print("-" * 44)
    for method in METHODS:
        avg_ndcg = sum(scores[method]["ndcg"]) / len(scores[method]["ndcg"])
        avg_mrr1 = sum(scores[method]["mrr1"]) / len(scores[method]["mrr1"])
        avg_mrr2 = sum(scores[method]["mrr2"]) / len(scores[method]["mrr2"])
        print(f"{method:<10} {avg_ndcg:>10.4f} {avg_mrr1:>10.4f} {avg_mrr2:>10.4f}")


if __name__ == "__main__":
    main()
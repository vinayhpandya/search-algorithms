"""
Run every eval query through all four search methods (BM25, dense,
hybrid, rerank) in a single process -- avoids reloading models per
query/method, and builds the candidate pool for labeling.

Output: eval/candidates.json
    {
      "<query>": {
        "results": {"bm25": [id, ...], "dense": [...], "hybrid": [...], "rerank": [...]},
        "candidates": [{"id":..., "title":..., "abstract":..., "label": null}, ...]
      },
      ...
    }

Usage:
    python -m eval.generate_candidates
"""
import json
from pathlib import Path

from search.bm25 import search_bm25
from search.dense import search_dense
from search.hybrid import search_hybrid
from search.rerank import search_rerank
from eval.queries import EVAL_QUERIES

OUT_PATH = Path(__file__).parent / "candidates.json"
K = 10  # results per method, per query


def normalize_bm25_or_dense(hits: list[dict]) -> list[dict]:
    """bm25.py / dense.py return raw OpenSearch hits (hit['_source'])."""
    return [
        {"id": h["_source"]["id"], "title": h["_source"]["title"], "abstract": h["_source"]["abstract"]}
        for h in hits
    ]


def normalize_hybrid_or_rerank(hits: list[dict]) -> list[dict]:
    """hybrid.py / rerank.py return custom dicts (hit['source'])."""
    return [
        {"id": h["source"]["id"], "title": h["source"]["title"], "abstract": h["source"]["abstract"]}
        for h in hits
    ]


def run_all_methods(query: str) -> dict[str, list[dict]]:
    return {
        "bm25": normalize_bm25_or_dense(search_bm25(query, k=K)),
        "dense": normalize_bm25_or_dense(search_dense(query, k=K)),
        "hybrid": normalize_hybrid_or_rerank(search_hybrid(query, k=K)),
        "rerank": normalize_hybrid_or_rerank(search_rerank(query, k=K)),
    }


def main():
    data = {}
    # load previous candidates.json if it exists, so re-running doesn't
    # wipe out labels you've already filled in
    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text())

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"[{i}/{len(EVAL_QUERIES)}] {query!r}")
        method_results = run_all_methods(query)

        # build the deduplicated candidate pool across all four methods
        pool: dict[str, dict] = {}
        for method_hits in method_results.values():
            for hit in method_hits:
                pool.setdefault(hit["id"], hit)

        existing_entry = data.get(query, {})
        existing_candidates = {c["id"]: c for c in existing_entry.get("candidates", [])}

        candidates = []
        for paper_id, hit in pool.items():
            if paper_id in existing_candidates:
                candidates.append(existing_candidates[paper_id])  # keep existing label
            else:
                candidates.append({
                    "id": paper_id,
                    "title": hit["title"],
                    "abstract": hit["abstract"],
                    "label": None,
                })

        data[query] = {
            "results": {method: [h["id"] for h in hits] for method, hits in method_results.items()},
            "candidates": candidates,
        }

    OUT_PATH.write_text(json.dumps(data, indent=2))
    total_pairs = sum(len(v["candidates"]) for v in data.values())
    print(f"\nWrote {len(data)} queries, {total_pairs} query-paper pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
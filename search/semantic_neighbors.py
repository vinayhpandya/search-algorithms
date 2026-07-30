"""
Find papers sharing the same semantic ID as a given paper -- a
"more like this" browsing feature, not a query-based search method.

Usage:
    python search/semantic_neighbors.py 2109.13086
"""
import argparse
import json
from pathlib import Path

SEMANTIC_IDS_PATH = Path(__file__).parent.parent / "eval" / "semantic_ids.json"


def get_cluster_neighbors(paper_id: str) -> list[dict]:
    semantic_ids = json.loads(SEMANTIC_IDS_PATH.read_text())
    if paper_id not in semantic_ids:
        raise ValueError(f"Unknown paper id: {paper_id}")

    target_id = tuple(semantic_ids[paper_id]["semantic_id"])
    return [
        {"id": pid, "title": data["title"]}
        for pid, data in semantic_ids.items()
        if tuple(data["semantic_id"]) == target_id and pid != paper_id
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_id")
    args = parser.parse_args()

    semantic_ids = json.loads(SEMANTIC_IDS_PATH.read_text())
    source_title = semantic_ids[args.paper_id]["title"]
    neighbors = get_cluster_neighbors(args.paper_id)

    print(f"\nPapers in the same cluster as: {source_title}\n")
    for n in neighbors:
        print(f"  - {n['title']}  (id={n['id']})")
    print(f"\n{len(neighbors)} neighbors found")


if __name__ == "__main__":
    main()
"""
Build hierarchical semantic IDs from existing SPECTER2 embeddings via
two-level k-means clustering. Each paper gets a semantic ID tuple
[coarse_cluster, fine_cluster] reflecting its position in a topic
hierarchy derived purely from embedding similarity -- distinct from
IVF-style clustering (which is invisible search-speed plumbing); here
the cluster assignments are the actual output, meant to be inspected
and stored as metadata.

Output:
    - eval/semantic_ids.json       (paper_id -> semantic_id, for inspection)
    - eval/cluster_centroids.json  (coarse cluster centroids, needed by
                                     search/cluster_filtered.py to find
                                     the nearest cluster for a new query)
    - backfills 'semantic_id' and 'coarse_cluster' fields into the
      OpenSearch 'papers' index

Usage:
    python search/build_semantic_ids.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch, helpers
from sklearn.cluster import KMeans

INDEX_NAME = "papers"
N_COARSE = 8
N_FINE = 4
OUT_PATH = Path(__file__).parent.parent / "eval" / "semantic_ids.json"
CENTROIDS_PATH = Path(__file__).parent.parent / "eval" / "cluster_centroids.json"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])


def fetch_all_papers() -> list[dict]:
    """Scroll through the full index to get every paper's id, title,
    and embedding (scroll needed since default search caps at ~10k,
    and is inefficient for bulk export anyway)."""
    papers = []
    resp = client.search(
        index=INDEX_NAME,
        body={"query": {"match_all": {}}, "_source": ["id", "title", "embedding"]},
        size=1000,
        scroll="2m",
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        for hit in hits:
            src = hit["_source"]
            if src.get("embedding"):
                papers.append({"id": src["id"], "title": src["title"], "embedding": src["embedding"]})
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    return papers


def main():
    print("Fetching papers + embeddings from OpenSearch...")
    papers = fetch_all_papers()
    print(f"Fetched {len(papers)} papers")

    embeddings = np.array([p["embedding"] for p in papers])

    print(f"Level 1: clustering into {N_COARSE} coarse groups...")
    coarse_model = KMeans(n_clusters=N_COARSE, random_state=42, n_init=10)
    coarse_labels = coarse_model.fit_predict(embeddings)

    # persist centroids -- needed at query time to find the nearest
    # coarse cluster for a brand-new query (see search/cluster_filtered.py)
    CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CENTROIDS_PATH.write_text(json.dumps(coarse_model.cluster_centers_.tolist(), indent=2))
    print(f"Saved {N_COARSE} coarse centroids -> {CENTROIDS_PATH}")

    semantic_ids = {}
    cluster_members = defaultdict(list)  # for the qualitative check below

    for coarse_id in range(N_COARSE):
        mask = coarse_labels == coarse_id
        indices_in_cluster = np.where(mask)[0]
        sub_embeddings = embeddings[indices_in_cluster]

        n_fine_actual = min(N_FINE, len(sub_embeddings))
        if n_fine_actual < 2:
            fine_labels = np.zeros(len(sub_embeddings), dtype=int)
        else:
            fine_model = KMeans(n_clusters=n_fine_actual, random_state=42, n_init=10)
            fine_labels = fine_model.fit_predict(sub_embeddings)

        for local_idx, global_idx in enumerate(indices_in_cluster):
            paper = papers[global_idx]
            sem_id = [int(coarse_id), int(fine_labels[local_idx])]
            semantic_ids[paper["id"]] = {"semantic_id": sem_id, "title": paper["title"]}
            cluster_members[tuple(sem_id)].append(paper["title"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(semantic_ids, indent=2))
    print(f"Wrote {len(semantic_ids)} semantic IDs -> {OUT_PATH}")

    print("\nBackfilling 'semantic_id' + 'coarse_cluster' fields into OpenSearch...")
    actions = (
        {
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": paper_id,
            "doc": {
                "semantic_id": data["semantic_id"],
                "coarse_cluster": data["semantic_id"][0],
            },
        }
        for paper_id, data in semantic_ids.items()
    )
    success, errors = helpers.bulk(client, actions, raise_on_error=False)
    print(f"Updated {success} papers, {len(errors)} errors")

    # qualitative sanity check: print titles from a few leaf clusters
    print("\n--- Sample clusters (topic coherence check) ---")
    sample_clusters = list(cluster_members.items())[:3]
    for sem_id, titles in sample_clusters:
        print(f"\nSemantic ID {list(sem_id)} ({len(titles)} papers):")
        for title in titles[:5]:
            print(f"  - {title}")


if __name__ == "__main__":
    main()
"""
Compute LTR feature vectors for every labeled query-paper pair in
eval/candidates.json. Needs the *actual* BM25/cosine scores, not just
rank position (which is all candidates.json's "results" field stores),
so this re-queries OpenSearch scoped to just each query's candidate
set rather than reusing generate_candidates.py's output directly.

Output: eval/ltr_features.json
    [{"query":..., "id":..., "label":..., "bm25_score":...,
      "dense_score":..., "citation_count":..., "year":...,
      "title_exact_match":...}, ...]

Usage:
    python -m eval.build_ltr_features
"""
import json
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch

from search.dense import embed_query

INDEX_NAME = "papers"
CANDIDATES_PATH = Path(__file__).parent / "candidates.json"
OUT_PATH = Path(__file__).parent / "ltr_features.json"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])


def bm25_scores_for_candidates(query: str, candidate_ids: list[str]) -> dict[str, float]:
    """BM25 score for exactly this query's candidate set -- uses the
    same field weighting (title^2 + abstract) as search/bm25.py, but
    filtered to only these documents so every candidate gets a real
    score, not just whichever ones a normal top-10 search would return."""
    body = {
        "size": len(candidate_ids),
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "abstract"],
                        "type": "best_fields",
                    }
                },
                "filter": {"ids": {"values": candidate_ids}},
            }
        },
    }
    resp = client.search(index=INDEX_NAME, body=body)
    scores = {hit["_source"]["id"]: hit["_score"] for hit in resp["hits"]["hits"]}
    # candidates that scored zero on every query term won't appear in
    # results at all (a "must" match_all-zero excludes them) -- fill in 0
    for cid in candidate_ids:
        scores.setdefault(cid, 0.0)
    return scores


def fetch_doc_fields(candidate_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch embedding/citation_count/year/title for a set of IDs."""
    resp = client.mget(body={"ids": candidate_ids}, index=INDEX_NAME)
    docs = {}
    for doc in resp["docs"]:
        if doc.get("found"):
            docs[doc["_id"]] = doc["_source"]
    return docs


def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def title_exact_match(query: str, title: str) -> float:
    """Fraction of query terms that appear literally in the title
    (case-insensitive) -- a cheap, interpretable lexical signal."""
    query_terms = set(query.lower().split())
    title_terms = set(title.lower().split())
    if not query_terms:
        return 0.0
    return len(query_terms & title_terms) / len(query_terms)


def main():
    candidates_data = json.loads(CANDIDATES_PATH.read_text())
    features = []

    for i, (query, entry) in enumerate(candidates_data.items(), 1):
        candidate_ids = [c["id"] for c in entry["candidates"]]
        labels_by_id = {c["id"]: c["label"] for c in entry["candidates"]}
        print(f"[{i}/{len(candidates_data)}] {query!r} ({len(candidate_ids)} candidates)")

        bm25_scores = bm25_scores_for_candidates(query, candidate_ids)
        docs = fetch_doc_fields(candidate_ids)
        query_vec = embed_query(query)

        for cid in candidate_ids:
            doc = docs.get(cid)
            if doc is None:
                continue  # shouldn't happen, but skip defensively

            features.append({
                "query": query,
                "id": cid,
                "label": labels_by_id[cid],
                "bm25_score": bm25_scores.get(cid, 0.0),
                "dense_score": cosine_sim(query_vec, doc["embedding"]),
                "citation_count": doc.get("citation_count", 0),
                "year": doc.get("year", 0),
                "title_exact_match": title_exact_match(query, doc["title"]),
            })

    OUT_PATH.write_text(json.dumps(features, indent=2))
    print(f"\nWrote {len(features)} feature rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
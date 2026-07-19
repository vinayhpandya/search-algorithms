"""
Hybrid search: fuse BM25 and dense k-NN results via Reciprocal Rank
Fusion (RRF). Score-based combination doesn't work well here since
BM25 and cosine-similarity scores live on incomparable scales; RRF
sidesteps that by combining based on rank position instead.

Usage:
    python search/hybrid.py "malware detection using graph neural networks"
"""
import argparse

from search.bm25 import search_bm25
from search.dense import search_dense

RRF_K = 60  # standard constant from the original RRF paper


def reciprocal_rank_fusion(*result_lists: list[dict], k: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            doc_id = hit["_source"]["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs[doc_id] = hit["_source"]

    ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [{"id": doc_id, "rrf_score": scores[doc_id], "source": docs[doc_id]} for doc_id in ranked_ids]


def search_hybrid(query: str, k: int = 10) -> list[dict]:
    bm25_hits = search_bm25(query, k=50)   # wider net before fusion
    dense_hits = search_dense(query, k=50)
    fused = reciprocal_rank_fusion(bm25_hits, dense_hits)
    return fused[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    hits = search_hybrid(args.query, args.k)
    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["source"]
        print(f"{i}. [rrf={hit['rrf_score']:.4f}] {src['title']} ({src['year']})")
        print(f"   id={src['id']}  citations={src.get('citation_count', 0)}")
        print()


if __name__ == "__main__":
    main()
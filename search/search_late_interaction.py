"""
Query the Qdrant late-interaction (ColBERT) index built by
build_qdrant_index.py.

Usage:
    python search/late_interaction_qdrant.py "malware detection using API calls"
"""
import argparse

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient

COLLECTION_NAME = "papers_colbert"
MODEL_NAME = "colbert-ir/colbertv2.0"

client = QdrantClient(url="http://localhost:6333")
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = LateInteractionTextEmbedding(MODEL_NAME)
    return _model


def search_late_interaction(query: str, k: int = 10) -> list[dict]:
    model = _load_model()
    # fastembed distinguishes query vs document encoding for ColBERT
    # (different special tokens/prefixing internally)
    query_embedding = list(model.query_embed(query))[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=k,
    ).points

    hits = []
    for r in results:
        payload = r.payload
        hits.append({
            "id": payload["arxiv_id"],
            "score": r.score,
            "source": {
                "id": payload["arxiv_id"],
                "title": payload["title"],
                "abstract": payload["abstract"],
                "year": payload["year"],
                "citation_count": payload.get("citation_count", 0),
            },
        })
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    hits = search_late_interaction(args.query, args.k)
    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["source"]
        print(f"{i}. [{hit['score']:.4f}] {src['title']} ({src['year']})")
        print(f"   id={src['id']}  citations={src.get('citation_count', 0)}")
        print()


if __name__ == "__main__":
    main()
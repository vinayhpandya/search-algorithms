"""
Rerank hybrid search's top candidates using a cross-encoder -- scores
each (query, document) pair jointly rather than independently, which
is more accurate than bi-encoder/BM25 scoring but too slow to run over
the full corpus, hence: rerank only the top-k shortlist.

Usage:
    python search/rerank.py "malware detection using graph neural networks"
"""
import argparse

from sentence_transformers import CrossEncoder

from search.hybrid import search_hybrid

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def search_rerank(query: str, k: int = 10, candidate_pool: int = 50) -> list[dict]:
    candidates = search_hybrid(query, k=candidate_pool)
    model = _load_model()

    pairs = [(query, f"{c['source']['title']} {c['source']['abstract']}") for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    hits = search_rerank(args.query, args.k)
    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["source"]
        print(f"{i}. [rerank={hit['rerank_score']:.4f}] {src['title']} ({src['year']})")
        print(f"   id={src['id']}  citations={src.get('citation_count', 0)}")
        print()


if __name__ == "__main__":
    main()
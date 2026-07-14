"""
BM25 lexical search against the 'papers' index.

Usage:
    python search/bm25.py "opcode sequence deep learning malware classification"
    python search/bm25.py "ransomware detection" --k 5
"""
import argparse

from opensearchpy import OpenSearch

INDEX_NAME = "papers"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])


def search_bm25(query: str, k: int = 10) -> list[dict]:
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                # title weighted 2x -- a term match in the title is a
                # stronger relevance signal than the same term buried
                # somewhere in the abstract.
                "fields": ["title^2", "abstract"],
                "type": "best_fields",
            }
        },
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return resp["hits"]["hits"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="search query")
    parser.add_argument("--k", type=int, default=10, help="number of results")
    args = parser.parse_args()

    hits = search_bm25(args.query, args.k)

    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        print(f"{i}. [{hit['_score']:.2f}] {src['title']} ({src['year']})")
        print(f"   id={src['id']}  citations={src.get('citation_count', 0)}")
        print()


if __name__ == "__main__":
    main()
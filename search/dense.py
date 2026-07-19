"""
Dense (semantic) search against the 'papers' index using SPECTER2
query embeddings + OpenSearch k-NN.

Usage:
    python search/dense.py "deep learning malware detection"
"""
import argparse

import torch
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModel

INDEX_NAME = "papers"
MODEL_NAME = "allenai/specter2_base"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
_tokenizer, _model = None, None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


@torch.no_grad()
def embed_query(query: str) -> list[float]:
    tokenizer, model = _load_model()
    inputs = tokenizer(query, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].tolist()


def search_dense(query: str, k: int = 10) -> list[dict]:
    vector = embed_query(query)
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return resp["hits"]["hits"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    hits = search_dense(args.query, args.k)
    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        print(f"{i}. [{hit['_score']:.4f}] {src['title']} ({src['year']})")
        print(f"   id={src['id']}  citations={src.get('citation_count', 0)}")
        print()


if __name__ == "__main__":
    main()
    
"""
Generative retrieval: query -> semantic ID (via trained Stage 2 model,
rqvae/train_query_to_id.py) -> papers sharing that ID, looked up directly.

Returns hits in the same {"source": {"id", "title", "abstract"}} shape as
hybrid.py / rerank.py / search_semantic_id.py, so it plugs into
eval/generate_candidates.py's normalize_hybrid_or_rerank without any
special-casing.

Usage:
    python -m search.search_generative "deep learning malware detection"
"""
import argparse
from pathlib import Path

import torch
from opensearchpy import OpenSearch

from rqvae.query_to_id_model import QueryToIDConfig, QueryToIDModel

INDEX_NAME = "papers"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "query_to_id" / "best.pt"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
_model = None
_id_to_papers = None  # tuple(semantic_id) -> [paper_ids], built once and cached


def _load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> QueryToIDModel:
    global _model
    if _model is None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = QueryToIDConfig(**ckpt["config"])
        _model = QueryToIDModel(cfg)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()
    return _model


def _load_id_index(index_name: str = INDEX_NAME) -> dict[tuple, list[str]]:
    """Build tuple(semantic_id) -> [paper_ids] locally in Python rather than
    querying OpenSearch for exact ID-tuple matches. semantic_id_rqvae is a
    plain list field; OpenSearch term queries on a multi-value field match
    ANY element, not the exact tuple, so an OpenSearch query can't cleanly
    express "papers where semantic_id == [6, 15] exactly." Fetching once
    and matching in Python sidesteps that, and is cheap at this corpus size."""
    global _id_to_papers
    if _id_to_papers is not None:
        return _id_to_papers

    id_to_papers: dict[tuple, list[str]] = {}
    query = {
        "query": {"exists": {"field": "semantic_id_rqvae"}},
        "_source": ["semantic_id_rqvae"],
    }
    resp = client.search(index=index_name, body=query, scroll="2m", size=500)
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]
    while hits:
        for hit in hits:
            id_tuple = tuple(hit["_source"]["semantic_id_rqvae"])
            id_to_papers.setdefault(id_tuple, []).append(hit["_id"])
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
    try:
        client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    _id_to_papers = id_to_papers
    return id_to_papers


def _fetch_docs(paper_ids: list[str], index_name: str = INDEX_NAME) -> dict[str, dict]:
    """Batch-fetch title/abstract for a list of paper_ids via mget, to
    match the {"source": {...}} hit shape the eval harness expects."""
    if not paper_ids:
        return {}
    resp = client.mget(
        index=index_name,
        body={"ids": paper_ids},
        _source=["id", "title", "abstract"],
    )
    return {doc["_id"]: doc["_source"] for doc in resp["docs"] if doc.get("found")}


def search_generative(
    query: str, k: int = 10, beam_size: int = 10, checkpoint_path: Path = DEFAULT_CHECKPOINT
) -> list[dict]:
    """Returns a ranked list of hits: [{"source": {"id", "title", "abstract"}}, ...]

    Ranking: papers from the top-beam predicted semantic ID come first,
    then the next-beam ID's papers, and so on, until k papers are collected
    or beams are exhausted. Within a single ID bucket there's no
    model-based ordering (all papers in a bucket are, by construction,
    equally close in ID-space) -- ties are broken by paper_id for
    determinism.
    """
    model = _load_model(checkpoint_path)
    id_to_papers = _load_id_index()

    beams = model.predict_topk(query, device="cpu", beam_size=beam_size)

    ranked_ids: list[str] = []
    seen = set()
    for id_seq, _score in beams:
        bucket = id_to_papers.get(tuple(id_seq), [])
        for paper_id in sorted(bucket):
            if paper_id not in seen:
                ranked_ids.append(paper_id)
                seen.add(paper_id)
        if len(ranked_ids) >= k:
            break

    ranked_ids = ranked_ids[:k]
    docs = _fetch_docs(ranked_ids)

    hits = []
    for paper_id in ranked_ids:
        doc = docs.get(paper_id)
        if doc is None:
            continue  # shouldn't happen, but don't crash the eval run over it
        hits.append({"source": {"id": doc.get("id", paper_id), "title": doc.get("title"), "abstract": doc.get("abstract")}})
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args()

    hits = search_generative(args.query, k=args.k, beam_size=args.beam_size)
    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["source"]
        print(f"{i}. {src['title']}")
        print(f"   id={src['id']}")


if __name__ == "__main__":
    main()
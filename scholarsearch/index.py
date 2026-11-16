"""Index a curated ScholarSearch Parquet corpus into an OpenSearch BM25 index.

Usage: python -m scholarsearch.index --root s3://BUCKET/scholarsearch --query 'malware detection'
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from urllib.parse import urlparse

from .storage import get_bytes, join

LOG = logging.getLogger(__name__)


def curated_uri(root: str, query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return join(root, "curated", f"openalex-{digest}.parquet")


def mapping() -> dict:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "doi": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "publication_year": {"type": "integer"},
            "citations": {"type": "long"},
            "topics": {"type": "keyword"},
            "source": {"type": "keyword"},
            "github_url": {"type": "keyword", "ignore_above": 2048},
        }}
    }


def paper_document(paper: dict) -> dict:
    return {
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "doi": paper.get("doi"),
        "arxiv_id": paper.get("arxiv_id"),
        "publication_year": paper.get("publication_year"),
        "citations": paper.get("citations") or 0,
        "topics": paper.get("topics") or [],
        "source": paper.get("source") or [],
        "github_url": paper.get("github_url"),
    }


def load_papers(uri: str) -> list[dict]:
    import polars as pl
    return pl.read_parquet(io.BytesIO(get_bytes(uri))).to_dicts()


def index_papers(*, root: str, query: str, url: str, index_name: str,
                 batch_size: int = 100, recreate: bool = False) -> dict:
    from opensearchpy import OpenSearch, helpers

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    endpoint = urlparse(url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("--opensearch-url must be http(s)://host:port")
    if endpoint.scheme == "http" and endpoint.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Plain HTTP only permitted for localhost (for example an SSM tunnel)")

    corpus = curated_uri(root, query)
    papers = load_papers(corpus)
    client = OpenSearch(hosts=[url], timeout=30, max_retries=3, retry_on_timeout=True)
    if not client.ping():
        raise RuntimeError(f"OpenSearch not reachable: {url}")
    if client.indices.exists(index=index_name):
        if recreate:
            LOG.warning("Deleting existing index %s as requested", index_name)
            client.indices.delete(index=index_name)
        else:
            LOG.info("Reusing index %s; documents with matching IDs will be updated", index_name)
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=mapping())

    def actions():
        for paper in papers:
            paper_id = paper.get("id")
            if not paper_id:
                continue
            yield {
                "_op_type": "index", "_index": index_name, "_id": str(paper_id),
                "_source": paper_document(paper),
            }

    indexed, errors = helpers.bulk(
        client, actions(), chunk_size=batch_size, request_timeout=90,
        max_retries=3, initial_backoff=1, max_backoff=8,
        raise_on_error=False, raise_on_exception=True,
    )
    if errors:
        raise RuntimeError(f"{len(errors)} bulk item(s) failed; first: {str(errors[0])[:500]}")
    client.indices.refresh(index=index_name)
    count = client.count(index=index_name)["count"]
    result = {"curated": corpus, "papers": len(papers), "indexed": indexed,
              "index": index_name, "index_document_count": count}
    LOG.info("Indexed %s documents; OpenSearch reports %s", indexed, count)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--opensearch-url", default="http://localhost:9200")
    p.add_argument("--index", default="scholarsearch-malware-v1")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--recreate", action="store_true", help="DELETE an existing index before indexing")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(index_papers(root=args.root, query=args.query, url=args.opensearch_url,
                                  index_name=args.index, batch_size=args.batch_size,
                                  recreate=args.recreate), indent=2))


if __name__ == "__main__":
    main()

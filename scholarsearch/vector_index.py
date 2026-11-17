"""Build an OpenSearch 2.15 Lucene HNSW index from PR4's S3 vectors.

Run on your Mac with the main project environment and an SSM-forwarded local
OpenSearch endpoint. The original PR3 BM25 index is never modified.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from urllib.parse import urlparse

import numpy as np

from .config import setting
from .embeddings import MODEL, PAPER_ADAPTER, artifact_prefix, source_digest
from .index import curated_uri, load_papers, paper_document
from .storage import get_bytes, join

LOG = logging.getLogger(__name__)


def hnsw_mapping(dimension: int = 768) -> dict:
    if dimension < 1:
        raise ValueError('dimension must be positive')
    result = {
        'settings': {'index': {'knn': True, 'number_of_shards': 1, 'number_of_replicas': 0}},
        'mappings': {'properties': {
            'paper_id': {'type': 'keyword'},
            'embedding': {'type': 'knn_vector', 'dimension': dimension,
                          'method': {'name': 'hnsw', 'engine': 'lucene',
                                     'space_type': 'innerproduct',
                                     'parameters': {'m': 16, 'ef_construction': 128}}},
            'title': {'type': 'text'}, 'abstract': {'type': 'text'},
            'doi': {'type': 'keyword'}, 'arxiv_id': {'type': 'keyword'},
            'publication_year': {'type': 'integer'}, 'citations': {'type': 'long'},
            'topics': {'type': 'keyword'}, 'source': {'type': 'keyword'},
            'github_url': {'type': 'keyword', 'ignore_above': 2048},
        }},
    }
    return result


def validate_endpoint(url: str) -> None:
    endpoint = urlparse(url)
    if endpoint.scheme not in ('http', 'https') or not endpoint.hostname:
        raise ValueError('OpenSearch URL must be http(s)://host:port')
    if endpoint.username or endpoint.password:
        raise ValueError('Credentials in OpenSearch URL are not supported')
    if endpoint.scheme == 'http' and endpoint.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Plain HTTP is only allowed over local SSM tunnel')


def load_artifacts(root: str, query: str) -> tuple[list[dict], np.ndarray, dict]:
    papers = load_papers(curated_uri(root, query))
    ids = [str(p.get('id') or '') for p in papers]
    if not ids or any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Curated corpus must have unique nonempty paper IDs')
    digest = source_digest(papers)
    prefix = artifact_prefix(root, query, digest)
    manifest = json.loads(get_bytes(join(prefix, 'manifest.json')))
    if (manifest.get('source_digest') != digest or manifest.get('source_uri') != curated_uri(root, query)
            or manifest.get('model') != MODEL or manifest.get('paper_adapter') != PAPER_ADAPTER
            or manifest.get('count') != len(ids) or manifest.get('normalized') is not True):
        raise ValueError('PR4 manifest does not match the current curated corpus')
    saved_ids = json.loads(get_bytes(manifest['paper_ids_uri']))
    if saved_ids != ids:
        raise ValueError('PR4 paper ID order does not match curated corpus')
    vectors = np.load(io.BytesIO(get_bytes(manifest['vectors_uri'])), allow_pickle=False)
    if vectors.dtype != np.float32 or vectors.shape != (len(ids), manifest.get('dimension')):
        raise ValueError('PR4 vectors have unexpected shape or dtype')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError('PR4 vectors must be finite and L2-normalized')
    return papers, vectors, manifest


def build_index(*, root: str, query: str, url: str, index_name: str,
                batch_size: int = 50, recreate: bool = False, client=None, bulk=None) -> dict:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    validate_endpoint(url)
    papers, vectors, manifest = load_artifacts(root, query)
    if client is None:
        from opensearchpy import OpenSearch
        client = OpenSearch(hosts=[url], timeout=30, max_retries=3, retry_on_timeout=True)
    if bulk is None:
        from opensearchpy.helpers import bulk as bulk_helper
        bulk = bulk_helper
    if not client.ping():
        raise RuntimeError(f'OpenSearch not reachable: {url}')
    # A failed prior index build is not silently reused. Require explicit rebuild.
    if client.indices.exists(index=index_name):
        if not recreate:
            raise ValueError(f'Index {index_name!r} already exists. Use --recreate to delete and rebuild it.')
        LOG.warning('Deleting index %s as explicitly requested', index_name)
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=hnsw_mapping(int(manifest['dimension'])))

    def actions():
        for paper, vector in zip(papers, vectors):
            yield {'_op_type': 'index', '_index': index_name, '_id': str(paper['id']),
                   '_source': {**paper_document(paper), 'paper_id': str(paper['id']),
                               'embedding': vector.tolist()}}

    indexed, errors = bulk(client, actions(), chunk_size=batch_size, request_timeout=120,
                           raise_on_error=False, raise_on_exception=True)
    if errors:
        raise RuntimeError(f'{len(errors)} bulk items failed: {str(errors[0])[:400]}')
    client.indices.refresh(index=index_name)
    count = client.count(index=index_name)['count']
    if count != len(papers) or indexed != len(papers):
        raise RuntimeError(f'Index verification failed: indexed={indexed}, count={count}, expected={len(papers)}')
    return {'index': index_name, 'indexed': indexed, 'count': count,
            'dimension': int(vectors.shape[1]), 'engine': 'lucene', 'method': 'hnsw',
            'source_digest': manifest['source_digest']}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=None)
    p.add_argument('--query', default=None, help='Corpus ingestion query, NOT the search query')
    p.add_argument('--opensearch-url', default=None)
    p.add_argument('--index', default=None)
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--recreate', action='store_true', help='Explicitly DELETE and rebuild this index')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build_index(root=args.root or setting('SCHOLARSEARCH_S3_ROOT'),
                                query=args.query or setting('SCHOLARSEARCH_QUERY', 'malware detection'),
                                url=args.opensearch_url or setting('OPENSEARCH_URL', 'http://localhost:9200'),
                                index_name=args.index or setting('OPENSEARCH_HYBRID_INDEX', 'scholarsearch-malware-hnsw-v1'),
                                batch_size=args.batch_size, recreate=args.recreate), indent=2))


if __name__ == '__main__':
    main()

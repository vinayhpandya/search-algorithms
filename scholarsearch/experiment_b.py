"""PR6 Experiment B: bounded OpenAlex scale-up with an immutable corpus freeze.

This module never touches the original PR1-PR6 root; it uses a separate S3 prefix.
Run `plan`, then several small `ingest` calls, then `freeze` before PR4 embedding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone

from .config import load_env, setting
from .curate import curate
from .embeddings import source_digest
from .index import curated_uri, load_papers
from .ingest import run as ingest_run
from .storage import exists, get_bytes, join, put_bytes

LOG = logging.getLogger(__name__)


def paths(root: str, query: str, experiment: str = 'experiment-b') -> dict[str, str]:
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,39}', experiment):
        raise ValueError('Experiment must be a short lowercase name (letters, digits, hyphens)')
    if not query.strip():
        raise ValueError('Search query cannot be empty')
    exp_root = join(root, 'experiments', experiment)
    key = hashlib.sha256(query.encode()).hexdigest()[:12]
    return {'root': exp_root, 'checkpoint': join(exp_root, 'raw', 'openalex', key, 'checkpoint.json'),
            'freeze': join(exp_root, 'corpus-freeze.json'), 'curated': curated_uri(exp_root, query)}


def status(root: str, query: str, experiment: str = 'experiment-b') -> dict:
    p = paths(root, query, experiment)
    checkpoint = json.loads(get_bytes(p['checkpoint'])) if exists(p['checkpoint']) else None
    frozen = json.loads(get_bytes(p['freeze'])) if exists(p['freeze']) else None
    return {'experiment_root': p['root'], 'query': query, 'checkpoint': checkpoint,
            'frozen': frozen, 'curated_uri': p['curated']}


def ingest(root: str, query: str, *, experiment: str = 'experiment-b',
           pages: int = 10, page_size: int = 100, min_interval: float = 1.0,
           api_key: str | None = None, email: str | None = None, session=None) -> dict:
    p = paths(root, query, experiment)
    if exists(p['freeze']):
        raise ValueError('This experiment is frozen. Start a new experiment name to ingest more.')
    if not 1 <= pages <= 25:
        raise ValueError('Use 1-25 pages per invocation, then inspect status and resume')
    if not 1 <= page_size <= 100:
        raise ValueError('OpenAlex supports at most 100 works per page')
    checkpoint = ingest_run(root=p['root'], query=query, max_pages=pages,
                            page_size=page_size, min_interval=min_interval,
                            api_key=api_key, email=email, session=session)
    return {'experiment_root': p['root'], 'checkpoint': checkpoint,
            'note': 'Resumable; call ingest again until enough distinct papers are present.'}


def freeze(root: str, query: str, *, experiment: str = 'experiment-b',
           min_papers: int = 10000) -> dict:
    p = paths(root, query, experiment)
    if exists(p['freeze']):
        saved = json.loads(get_bytes(p['freeze']))
        if saved.get('query') != query or saved.get('experiment_root') != p['root']:
            raise ValueError('Frozen metadata does not match this experiment')
        return saved
    if not exists(p['checkpoint']):
        raise ValueError('No ingestion checkpoint. Ingest pages first.')
    checkpoint = json.loads(get_bytes(p['checkpoint']))
    if checkpoint.get('query') != query:
        raise ValueError('Ingestion checkpoint query mismatch')
    if min_papers < 1:
        raise ValueError('min_papers must be positive')
    # Curate deterministic, deduplicated data before we create a frozen record.
    result = curate(p['root'], query, fmt='parquet')
    papers = load_papers(p['curated'])
    if len(papers) != result['papers'] or len(papers) < min_papers:
        raise ValueError(f'Only {len(papers)} unique papers; need at least {min_papers}. '
                         'Ingest more pages and freeze again (no embedding has started).')
    frozen = {'experiment_root': p['root'], 'query': query,
              'curated_uri': p['curated'], 'count': len(papers),
              'source_digest': source_digest(papers), 'ingested_pages': int(checkpoint['page']),
              'created_at': datetime.now(timezone.utc).isoformat()}
    # Written last: this guards against future ingest calls changing the corpus.
    put_bytes(p['freeze'], json.dumps(frozen, indent=2).encode())
    return frozen


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'status', 'ingest', 'freeze'])
    parser.add_argument('--root', default=None, help='Base root; experiment gets its own sub-prefix')
    parser.add_argument('--query', default=None)
    parser.add_argument('--experiment', default='experiment-b')
    parser.add_argument('--pages', type=int, default=10, help='Pages per invocation, 1-25')
    parser.add_argument('--page-size', type=int, default=100)
    parser.add_argument('--min-interval', type=float, default=1.0)
    parser.add_argument('--min-papers', type=int, default=10000)
    args = parser.parse_args()
    root = args.root or setting('SCHOLARSEARCH_S3_ROOT')
    query = args.query or setting('SCHOLARSEARCH_QUERY', 'malware detection')
    logging.basicConfig(level=logging.INFO)
    if args.action == 'ingest':
        import os
        result = ingest(root, query, experiment=args.experiment, pages=args.pages,
                        page_size=args.page_size, min_interval=args.min_interval,
                        api_key=os.getenv('OPENALEX_API_KEY'), email=os.getenv('OPENALEX_EMAIL'))
    elif args.action == 'freeze':
        result = freeze(root, query, experiment=args.experiment, min_papers=args.min_papers)
    else:
        result = status(root, query, experiment=args.experiment)
        if args.action == 'plan':
            result['recommended_steps'] = [
                'Repeat ingest --pages 10 until you have about 150 pages (15k results before dedup)',
                'Run freeze --min-papers 10000 (or a higher threshold if reached)',
                'Embed using scholarsearch.embeddings --root <experiment_root> and matching --query',
                'Benchmark using scholarsearch.ann_benchmark --root <experiment_root> and matching --query',
            ]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Resumable SPECTER2 paper embedding job; saves verified batches and a final S3 artifact.

SPECTER2 paper encoder: allenai/specter2_base + allenai/specter2 (proximity).
Use the adhoc_query adapter for short user queries, not this paper adapter.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import logging
from datetime import datetime, timezone

import numpy as np

from .config import load_env, setting
from .index import curated_uri, load_papers
from .storage import exists, get_bytes, join, put_bytes

LOG = logging.getLogger(__name__)
MODEL = 'allenai/specter2_base'
PAPER_ADAPTER = 'allenai/specter2'


def source_digest(papers: list[dict]) -> str:
    h = hashlib.sha256()
    for paper in papers:
        h.update(json.dumps([paper.get('id'), paper.get('title'), paper.get('abstract')], ensure_ascii=False).encode())
        h.update(b'\n')
    return h.hexdigest()[:16]


def paper_text(paper: dict, sep_token: str = '[SEP]') -> tuple[str, str]:
    title = (paper.get('title') or '').strip()
    abstract = (paper.get('abstract') or '').strip()
    return f'{title}{sep_token}{abstract}' if abstract else title, ('title_abstract' if abstract else 'title_only')


def normalize(array: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(array, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Embeddings must be a finite 2D matrix')
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('Zero-norm embedding encountered')
    return a / norms


def npy_bytes(a: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.save(out, a, allow_pickle=False)
    return out.getvalue()


def read_npy(uri: str) -> np.ndarray:
    return np.load(io.BytesIO(get_bytes(uri)), allow_pickle=False)


class SpecterEncoder:
    def __init__(self, *, adapter: str = PAPER_ADAPTER, device: str = 'cpu'):
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoAdapterModel.from_pretrained(MODEL)
        self.model.load_adapter(adapter, source='hf', load_as='scholarsearch', set_active=True)
        self.model.to(device).eval()
        self.device = device
        self.sep_token = self.tokenizer.sep_token

    def encode(self, texts: list[str]) -> np.ndarray:
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=512,
                             return_tensors='pt', return_token_type_ids=False)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with self.torch.inference_mode():
            hidden = self.model(**tok).last_hidden_state[:, 0, :]
        return hidden.detach().float().cpu().numpy()


def artifact_prefix(root: str, query: str, digest: str) -> str:
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
    return join(root, 'embeddings', f'openalex-{query_hash}', f'specter2-{digest}')


def embed(*, root: str, query: str, encoder=None, batch_size: int = 4, device: str = 'cpu') -> dict:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    papers = load_papers(curated_uri(root, query))
    if not papers or any(not p.get('id') for p in papers):
        raise ValueError('Corpus empty or missing paper ID')
    ids = [str(p['id']) for p in papers]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate paper IDs in corpus')
    digest = source_digest(papers)
    prefix = artifact_prefix(root, query, digest)
    done_uri = join(prefix, 'manifest.json')
    if exists(done_uri):
        manifest = json.loads(get_bytes(done_uri))
        if (manifest.get('source_digest') != digest or manifest.get('count') != len(papers)
                or manifest.get('model') != MODEL or manifest.get('paper_adapter') != PAPER_ADAPTER):
            raise ValueError('Existing manifest incompatible with corpus')
        if not exists(manifest['vectors_uri']) or not exists(manifest['paper_ids_uri']):
            raise RuntimeError('Final manifest exists but vector or ID artifact is missing')
        stored_ids = json.loads(get_bytes(manifest['paper_ids_uri']))
        if stored_ids != ids:
            raise ValueError('Completed embedding ID mapping does not match corpus')
        return manifest
    if encoder is None:
        encoder = SpecterEncoder(device=device)
    all_vectors = []
    text_types = []
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start+batch_size]
        batch_ids = ids[start:start+batch_size]
        texts_and_types = [paper_text(p, encoder.sep_token) for p in batch]
        text_types.extend(kind for _, kind in texts_and_types)
        data_uri = join(prefix, 'batches', f'{start:08d}.npy')
        ids_uri = join(prefix, 'batches', f'{start:08d}.json')
        if exists(data_uri) and exists(ids_uri):
            saved_ids = json.loads(get_bytes(ids_uri))
            if saved_ids != batch_ids:
                raise ValueError(f'Existing batch {start} IDs do not match corpus')
            vectors = read_npy(data_uri)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError(f'Existing batch {start} has wrong row count')
            LOG.info('Reused embedding batch starting %d', start)
        else:
            vectors = normalize(encoder.encode([t for t, _ in texts_and_types]))
            # Write data first; IDs are the batch's completion marker.
            put_bytes(data_uri, npy_bytes(vectors))
            put_bytes(ids_uri, json.dumps(batch_ids).encode())
            LOG.info('Saved embedding batch %d / %d', min(start+len(batch), len(papers)), len(papers))
        all_vectors.append(normalize(vectors))
    dimensions = {arr.shape[1] for arr in all_vectors}
    if len(dimensions) != 1:
        raise ValueError('Embedding dimension changed between batches')
    stacked = np.concatenate(all_vectors, axis=0)
    if stacked.shape[0] != len(ids):
        raise ValueError('Vector count mismatch')
    vectors_uri = join(prefix, 'vectors.npy')
    ids_uri = join(prefix, 'paper_ids.json')
    put_bytes(vectors_uri, npy_bytes(stacked))
    put_bytes(ids_uri, json.dumps(ids, ensure_ascii=False).encode())
    manifest = {'source_digest': digest, 'source_uri': curated_uri(root, query),
                'model': MODEL, 'paper_adapter': PAPER_ADAPTER, 'batch_size': batch_size,
                'dimension': int(stacked.shape[1]), 'count': len(ids),
                'title_only': text_types.count('title_only'),
                'title_abstract': text_types.count('title_abstract'),
                'vectors_uri': vectors_uri, 'paper_ids_uri': ids_uri,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'normalized': True}
    # Commit manifest last, only after complete output objects are uploaded.
    put_bytes(done_uri, json.dumps(manifest, indent=2).encode())
    return manifest


def main():
    load_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=None)
    p.add_argument('--query', default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(embed(root=args.root or setting('SCHOLARSEARCH_S3_ROOT'),
                           query=args.query or setting('SCHOLARSEARCH_QUERY', 'malware detection'),
                           batch_size=args.batch_size or int(setting('EMBEDDING_BATCH_SIZE', '4')),
                           device=args.device), indent=2))


if __name__ == '__main__':
    main()

"""Offline BM25 / exact-SPECTER2 / RRF retrieval for the frozen ScholarSearch corpus.

Two commands:
  encode --queries queries_draft.tsv --out query_vectors.jsonl  # .venv-embed
  run --root s3://.../experiments/experiment-b --query 'malware detection'
      --queries queries_draft.tsv --vectors query_vectors.jsonl --out retrieval_results.json

No OpenSearch, model training, or new cloud infrastructure. All results are
unjudged; this is *not* a relevance benchmark until an LLM/human judges pairs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .embeddings import MODEL, source_digest
from .index import curated_uri, load_papers
from .search import rrf

QUERY_ADAPTER = 'allenai/specter2_adhoc_query'
TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_queries(path: str) -> list[dict]:
    # Query drafts may have a UTF-8 BOM, CRLF endings, or aligned columns
    # pasted from a terminal. Preserve query spaces while parsing columns.
    with open(path, encoding='utf-8-sig', newline='') as f:
        raw_lines = [line.rstrip('\r\n') for line in f if line.strip()]
    if not raw_lines:
        raise ValueError(f'Empty query file: {path}')
    header = raw_lines[0]
    delimiter = '\t' if '\t' in header else None
    if delimiter is None:
        # Multiple spaces are treated as column boundaries; single spaces
        # within natural-language query text are retained.
        rows = [re.split(r' {2,}', line.strip()) for line in raw_lines]
    else:
        rows = list(csv.reader(raw_lines, delimiter=delimiter))
    fieldnames = [field.strip().lstrip('\ufeff').lower() for field in rows[0]]
    if not {'query_id', 'query'}.issubset(fieldnames):
        raise ValueError(
            f'Expected TSV columns query_id, query; parsed header={fieldnames!r}. '
            'Use a tab-separated TSV with query_id, query, intent, reviewed.'
        )
    queries = []
    for line_num, values in enumerate(rows[1:], start=2):
        if len(values) > len(fieldnames):
            raise ValueError(f'Too many columns on query line {line_num}')
        row = dict(zip(fieldnames, values))
        qid, q = (row.get('query_id') or '').strip(), (row.get('query') or '').strip()
        if not qid or not q:
            raise ValueError(f'Query line {line_num} requires an ID and text')
        queries.append({'query_id': qid, 'query': q, 'intent': (row.get('intent') or '').strip()})
    if not queries or len(set(r['query_id'] for r in queries)) != len(queries):
        raise ValueError('Empty query set or duplicate query IDs')
    if len(set(r['query'].casefold() for r in queries)) != len(queries):
        raise ValueError('Duplicate query text')
    return queries


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or '').lower())


class BM25:
    """A simple deterministic BM25 implementation with title repeated 3x.

    This is a reference lexical baseline, NOT equivalent to OpenSearch's
    separately analyzed title^3 + abstract multi_match scoring.
    """
    def __init__(self, papers: list[dict], k1: float = 1.2, b: float = 0.75):
        self.n = len(papers)
        self.k1, self.b = k1, b
        self.doc_len = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, paper in enumerate(papers):
            tokens = tokenize(paper.get('title') or '') * 3 + tokenize(paper.get('abstract') or '')
            self.doc_len.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term].append((i, count))
        self.avg_len = sum(self.doc_len) / max(1, self.n)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log1p((self.n - df + 0.5) / (df + 0.5))
            for idx, tf in postings:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / max(1, self.avg_len))
                scores[idx] += idf * tf * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]


def load_query_vectors(path: str, queries: list[dict]) -> np.ndarray:
    data = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get('query_id')
            if qid in data:
                raise ValueError(f'Duplicate query embedding {qid}')
            data[qid] = row
    vectors = []
    if set(data) != {r['query_id'] for r in queries}:
        raise ValueError('Query-vector IDs do not match query TSV')
    for row in queries:
        record = data[row['query_id']]
        if (record.get('query') != row['query'] or record.get('model') != MODEL
                or record.get('adapter') != QUERY_ADAPTER):
            raise ValueError(f'Query model/text mismatch for {row["query_id"]}')
        v = np.asarray(record.get('vector'), dtype=np.float32)
        if v.shape != (768,) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1, atol=1e-3):
            raise ValueError(f'Invalid query embedding for {row["query_id"]}')
        vectors.append(v)
    return np.ascontiguousarray(np.stack(vectors), dtype=np.float32)


def encode(queries: list[dict], *, out: str, batch_size: int = 8, device: str = 'cpu') -> None:
    # Keep torch/adapters isolated in .venv-embed. The 'run' operation only needs numpy.
    from .embeddings import SpecterEncoder, normalize
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    encoder = SpecterEncoder(adapter=QUERY_ADAPTER, device=device)
    # Explicitly activate the adapter; otherwise adapter-transformers can
    # silently run the unadapted base model for a query forward pass.
    encoder.model.set_active_adapters('scholarsearch')
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            vectors = normalize(encoder.encode([r['query'] for r in batch]))
            if vectors.shape != (len(batch), 768):
                raise ValueError('SPECTER2 returned an unexpected embedding shape')
            for row, vector in zip(batch, vectors):
                f.write(json.dumps({'query_id': row['query_id'], 'query': row['query'],
                                    'model': MODEL, 'adapter': QUERY_ADAPTER,
                                    'vector': vector.tolist()}) + '\n')
            print(f'Encoded {min(start + len(batch), len(queries))}/{len(queries)} queries', flush=True)
    print(f'Saved {path}')


def _docs(pairs: list[tuple[int, float]], papers: list[dict]) -> list[dict]:
    return [{'id': str(papers[i]['id']), 'score': float(score),
             'title': papers[i].get('title') or '', 'abstract': papers[i].get('abstract') or ''}
            for i, score in pairs]


def retrieve(papers: list[dict], doc_vectors: np.ndarray, queries: list[dict], query_vectors: np.ndarray,
             *, k: int = 10, candidate_k: int = 50) -> dict:
    if k < 1 or candidate_k < k:
        raise ValueError('Require candidate_k >= k >= 1')
    ids = [str(p.get('id') or '') for p in papers]
    if not all(ids) or len(ids) != len(set(ids)) or len(ids) != len(doc_vectors):
        raise ValueError('Missing/duplicate IDs or inconsistent corpus/vector count')
    if doc_vectors.shape[1] != 768 or query_vectors.shape != (len(queries), 768):
        raise ValueError('Expected aligned 768-dimensional embeddings')
    if not np.isfinite(doc_vectors).all() or not np.allclose(np.linalg.norm(doc_vectors, axis=1), 1, atol=1e-3):
        raise ValueError('Paper vectors are not normalized')
    bm25 = BM25(papers)
    output = []
    for row, qvec in zip(queries, query_vectors):
        start = time.perf_counter_ns()
        bm = _docs(bm25.search(row['query'], candidate_k), papers)
        bm_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        scores = doc_vectors @ qvec  # exact inner-product against all 15K papers
        indices = sorted(range(len(papers)), key=lambda i: (-float(scores[i]), ids[i]))[:candidate_k]
        dense = _docs([(i, float(scores[i])) for i in indices], papers)
        dense_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        hybrid = rrf([bm, dense], k=60, limit=k)
        hybrid_ms = (time.perf_counter_ns() - start) / 1e6
        output.append({'query_id': row['query_id'], 'query': row['query'],
                       'methods': {'bm25': bm[:k], 'dense': dense[:k], 'hybrid': hybrid},
                       'latency_ms': {'bm25': bm_ms, 'dense': dense_ms,
                                      'hybrid_rrf_only': hybrid_ms,
                                      'hybrid_total': bm_ms + dense_ms + hybrid_ms}})
        print(f"{row['query_id']} done", flush=True)
    return {'status': 'UNJUDGED_RETRIEVAL_ONLY', 'method_details': {
                'bm25': 'Python BM25, title tokens repeated 3x plus abstract',
                'dense': 'Exact normalized SPECTER2 inner product, not ANN',
                'hybrid': 'RRF k=60 on top candidate_k BM25 and dense results',
                'latency': 'Retrieval-only timing; excludes query embedding, S3 loading and index construction'},
            'corpus': {'paper_count': len(papers), 'source_digest': source_digest(papers)},
            'query_count': len(queries), 'k': k, 'candidate_k': candidate_k,
            'created_at': datetime.now(timezone.utc).isoformat(), 'results': output}


def run(*, root: str, query: str, queries_path: str, vector_path: str, out: str,
        k: int = 10, candidate_k: int = 50):
    from .ann_benchmark import read_corpus
    papers = load_papers(curated_uri(root, query))
    vectors, manifest = read_corpus(root, query)
    expected = [str(p['id']) for p in papers]
    from .storage import get_bytes
    paper_ids = json.loads(get_bytes(manifest['paper_ids_uri']))
    if paper_ids != expected:
        raise ValueError('Corpus and embedding paper IDs differ')
    qs = read_queries(queries_path)
    qvecs = load_query_vectors(vector_path, qs)
    report = retrieve(papers, vectors, qs, qvecs, k=k, candidate_k=candidate_k)
    report['corpus']['embedding_model'] = manifest['model']
    report['corpus']['query_adapter'] = QUERY_ADAPTER
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {len(qs)} queries -> {path} (UNJUDGED)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    e = sub.add_parser('encode')
    e.add_argument('--queries', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--batch-size', type=int, default=8)
    e.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    r = sub.add_parser('run')
    r.add_argument('--root', required=True)
    r.add_argument('--query', default='malware detection')
    r.add_argument('--queries', required=True)
    r.add_argument('--vectors', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--k', type=int, default=10)
    r.add_argument('--candidate-k', type=int, default=50)
    args = p.parse_args()
    if args.action == 'encode':
        encode(read_queries(args.queries), out=args.out, batch_size=args.batch_size, device=args.device)
    else:
        run(root=args.root, query=args.query, queries_path=args.queries,
            vector_path=args.vectors, out=args.out, k=args.k, candidate_k=args.candidate_k)


if __name__ == '__main__':
    main()

"""Resumable LLM judging and pooled NDCG@10 / MRR@10 for eval_retrieve output.

This is a pragmatic LLM-judged prototype benchmark, NOT human ground truth.

Usage:
  uv run --env-file .env python -m scholarsearch.eval_judge judge --retrieval ... --labels ... --query-limit 10
  uv run python -m scholarsearch.eval_judge report --retrieval ... --labels ... --out ... --query-limit 10
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
import statistics
from pathlib import Path
import time
from urllib import error, request

RUBRIC = ("Judge relevance for a scholarly paper search engine. Use ONLY the query, paper title, and abstract. "
          "Return a JSON object with integer label and short reason. "
          "label=2: paper directly addresses the query; "
          "label=1: related but does not directly answer the query; "
          "label=0: irrelevant or insufficient information to establish relevance. "
          "Do not assume relevance from the broad field alone.")


def load_retrieval(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('status') != 'UNJUDGED_RETRIEVAL_ONLY':
        raise ValueError('Expected an unjudged eval_retrieve report')
    if not data.get('results') or not data.get('corpus', {}).get('source_digest'):
        raise ValueError('Incomplete retrieval report')
    ids = [row['query_id'] for row in data['results']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate query IDs')
    for row in data['results']:
        if not {'bm25', 'dense', 'hybrid'}.issubset(row['methods']):
            raise ValueError('Retrieval report missing method')
        for method, hits in row['methods'].items():
            if len([hit['id'] for hit in hits]) != len({hit['id'] for hit in hits}):
                raise ValueError(f'Duplicate paper IDs in {row["query_id"]}/{method}')
    return data


def candidates(data: dict, query_limit: int | None = None) -> list[dict]:
    rows = data['results'] if query_limit is None else data['results'][:query_limit]
    pairs = []
    for row in rows:
        by_id = {}
        for hits in row['methods'].values():
            for hit in hits:
                by_id.setdefault(str(hit['id']), hit)
        for paper_id, hit in by_id.items():
            pairs.append({'query_id': row['query_id'], 'query': row['query'],
                          'paper_id': paper_id, 'title': hit.get('title') or '',
                          'abstract': hit.get('abstract') or ''})
    return pairs


def read_labels(path: str, digest: str) -> dict[tuple[str, str], dict]:
    saved = {}
    if not Path(path).exists():
        return saved
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('source_digest') != digest:
            raise ValueError(f'Label file contains a different corpus at line {line_number}')
        label = record.get('label')
        if type(label) is not int or label not in (0, 1, 2):
            raise ValueError(f'Invalid relevance label at line {line_number}')
        key = (record['query_id'], record['paper_id'])
        if key in saved:
            raise ValueError(f'Duplicate judgment at line {line_number}: {key}')
        saved[key] = record
    return saved


def call_openai(pair: dict, *, model: str, api_key: str, api_url: str = 'https://api.openai.com/v1/chat/completions') -> dict:
    payload = {
        'model': model,
        'temperature': 0,
        'max_tokens': 125,
        'response_format': {'type': 'json_object'},
        'messages': [
            {'role': 'system', 'content': RUBRIC},
            {'role': 'user', 'content': json.dumps({
                'query': pair['query'], 'title': pair['title'],
                'abstract': pair['abstract'][:2500]}, ensure_ascii=False)},
        ],
    }
    req = request.Request(api_url, data=json.dumps(payload).encode('utf-8'), method='POST',
                          headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    for attempt in range(4):
        try:
            with request.urlopen(req, timeout=60) as response:
                outer = json.loads(response.read())
            obj = json.loads(outer['choices'][0]['message']['content'])
            if type(obj.get('label')) is not int or obj['label'] not in (0, 1, 2):
                raise ValueError(f'Invalid judge label: {obj}')
            return {'label': obj['label'], 'reason': str(obj.get('reason', ''))[:500],
                    'usage': outer.get('usage', {})}
        except error.HTTPError as e:
            # Never retry invalid credentials, quota exhaustion, or invalid model.
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                detail = e.read().decode('utf-8', errors='replace')[:350]
                raise RuntimeError(f'OpenAI HTTP {e.code}: {detail}') from e
        except (error.URLError, TimeoutError) as e:
            if attempt == 3:
                raise RuntimeError(f'LLM judge request failed: {e}') from e
        time.sleep(2 ** attempt)
    raise RuntimeError('LLM judge failed')


def judge(data: dict, *, labels_path: str, model: str, query_limit: int | None = None,
          max_new: int | None = None, api_key: str | None = None, judge_fn=None) -> dict:
    digest = data['corpus']['source_digest']
    old = read_labels(labels_path, digest)
    pairs = candidates(data, query_limit)
    pending = [p for p in pairs if (p['query_id'], p['paper_id']) not in old]
    pending = pending if max_new is None else pending[:max_new]
    if pending and judge_fn is None and not (api_key or os.environ.get('OPENAI_API_KEY')):
        raise ValueError('OPENAI_API_KEY missing; set in .env (never commit it)')
    path = Path(labels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_new = 0
    with path.open('a', encoding='utf-8') as f:
        for pair in pending:
            result = (judge_fn(pair) if judge_fn else call_openai(
                pair, model=model, api_key=api_key or os.environ['OPENAI_API_KEY']))
            record = {**pair, 'source_digest': digest, 'label': result['label'],
                      'reason': result.get('reason', ''), 'judge_model': model,
                      'judge_type': 'llm', 'created_at': datetime.now(timezone.utc).isoformat(),
                      'usage': result.get('usage', {})}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()  # append-only checkpoint: interrupted jobs resume, not repay for completed pairs
            total_new += 1
            if total_new % 10 == 0 or total_new == len(pending):
                print(f'New labels: {total_new}/{len(pending)}; saved {path}', flush=True)
    return {'total_pairs': len(pairs), 'previously_judged': len(pairs) - len([p for p in pairs if (p['query_id'],p['paper_id']) not in old]),
            'newly_judged': total_new, 'still_missing': len(pairs) - len(old.keys() & {(p['query_id'],p['paper_id']) for p in pairs}) - total_new}


def dcg(labels: list[int]) -> float:
    return sum((2 ** label - 1) / math.log2(rank + 1) for rank, label in enumerate(labels, 1))


def evaluate(data: dict, labels: dict[tuple[str, str], dict], *, query_limit: int | None = None,
             methods: tuple[str, ...] | list[str] | None = None) -> dict:
    rows = data['results'] if query_limit is None else data['results'][:query_limit]
    methods = tuple(methods or ('bm25', 'dense', 'hybrid'))
    scores = defaultdict(list)
    missing = []
    for row in rows:
        qid = row['query_id']
        pooled_ids = {hit['id'] for hits in row['methods'].values() for hit in hits}
        for paper_id in pooled_ids:
            if (qid, paper_id) not in labels:
                missing.append({'query_id': qid, 'paper_id': paper_id})
    if missing:
        raise ValueError(f'{len(missing)} unlabeled query-paper pairs; run judge again. Example: {missing[:3]}')
    k = int(data.get('k', 10))
    for row in rows:
        qid = row['query_id']
        ideal_labels = sorted((labels[(qid, hit_id)]['label'] for hit_id in
                               {h['id'] for hits in row['methods'].values() for h in hits}), reverse=True)[:k]
        ideal = dcg(ideal_labels)
        for method in methods:
            retrieved = [labels[(qid, hit['id'])]['label'] for hit in row['methods'][method][:k]]
            ndcg = dcg(retrieved) / ideal if ideal else 0.
            mrr = next((1 / rank for rank, val in enumerate(retrieved, 1) if val >= 2), 0.)
            scores[method].append({'query_id': qid, 'ndcg_at_10': ndcg, 'mrr_at_10_rel2': mrr})
    metrics = {method: {'ndcg_at_10': sum(r['ndcg_at_10'] for r in values) / len(values),
                        'mrr_at_10_rel2': sum(r['mrr_at_10_rel2'] for r in values) / len(values),
                        'per_query': values}
               for method, values in scores.items()}
    # The runner records only post-embedding retrieval latency; it excludes
    # S3 loading and index construction, and hybrid includes both retrievals.
    latency_keys = {m: ('hybrid_total' if m == 'hybrid' else m) for m in methods}
    for method, key in latency_keys.items():
        times = sorted(float(row.get('latency_ms', {}).get(key, 0)) for row in rows)
        if any(key not in row.get('latency_ms', {}) for row in rows):
            continue
        p95 = times[min(len(times) - 1, math.ceil(len(times) * 0.95) - 1)]
        metrics[method]['retrieval_p50_ms'] = statistics.median(times)
        metrics[method]['retrieval_p95_ms'] = p95
    return {'status': 'COMPLETE_LLM_JUDGED_POOLED', 'warning': 'LLM-generated judgments, not human ground truth. Results are pooled over the methods in the retrieval input; unjudged corpus documents are not known irrelevant.',
            'source_digest': data['corpus']['source_digest'], 'queries': len(rows),
            'query_ids': [row['query_id'] for row in rows], 'k': k,
            'unique_judged_pairs': len(candidates(data, query_limit)), 'metrics': metrics,
            'method_details': data.get('method_details', {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    j = sub.add_parser('judge', help='Resumable LLM relevance judgment (incurs API charges)')
    r = sub.add_parser('report', help='Compute metrics only after all requested pairs are labeled')
    for p in (j, r):
        p.add_argument('--retrieval', required=True)
        p.add_argument('--labels', required=True)
        p.add_argument('--query-limit', type=int, default=None)
    j.add_argument('--model', default='gpt-4o-mini')
    j.add_argument('--max-new', type=int, default=None, help='Hard cap on new billable judgments')
    r.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.query_limit is not None and args.query_limit < 1:
        parser.error('--query-limit must be positive')
    data = load_retrieval(args.retrieval)
    if args.action == 'judge':
        if args.max_new is not None and args.max_new < 1:
            parser.error('--max-new must be positive')
        print(json.dumps(judge(data, labels_path=args.labels, model=args.model,
                               query_limit=args.query_limit, max_new=args.max_new), indent=2))
    else:
        labels = read_labels(args.labels, data['corpus']['source_digest'])
        report = evaluate(data, labels, query_limit=args.query_limit)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'out': str(path), 'queries': report['queries'],
                          'metrics': {m: {k: v for k, v in val.items() if k != 'per_query'}
                                      for m, val in report['metrics'].items()}}, indent=2))


if __name__ == '__main__':
    main()

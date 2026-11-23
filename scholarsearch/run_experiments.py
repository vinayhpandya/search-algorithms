"""One command to run ScholarSearch experiments and save every result under a run ID.

Reuses the existing modules instead of reimplementing them:
  ann        -> scholarsearch.ann_benchmark.run_benchmark   (Experiment A: index speed/recall)
  retrieval  -> scholarsearch.eval_retrieve.retrieve        (Experiment B: BM25 / dense / hybrid)
                or scholarsearch.methods.run_methods        (adds HNSW, cross-encoder, ColBERT)
  judge      -> scholarsearch.eval_judge.judge              (LLM-judge only NEW pairs, hard $ cap)
  report     -> scholarsearch.eval_judge.evaluate           (NDCG@10 / MRR)

Every stage writes its output as soon as it finishes, to <out_root>/<run_id>/, which can be a
local folder or s3://bucket/prefix. A run_manifest.json records the config, git commit,
stage status and timings, so any number in the final report can be traced to its run.

The judge stage is the only stage that spends money. It reuses every existing label, judges only
missing (query, paper) pairs with the same model/rubric, and stops before a cumulative spending
cap (tracked in judge_spend.json next to the labels, across all runs) would be exceeded. The
report stage never calls an LLM: if pairs are still unjudged it reports how many, not scores.

Usage:
  uv run python -m scholarsearch.run_experiments --config experiments/baseline-15k.json
  uv run python -m scholarsearch.run_experiments --config ... --stages ann
  uv run python -m scholarsearch.run_experiments --config ... --stages report --from-run <run folder>
  uv run python -m scholarsearch.run_experiments --config ... --dry-run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

from .storage import exists, get_bytes, join, put_bytes

STAGES = ('ann', 'retrieval', 'judge', 'report')

ANN_DEFAULTS = {'queries': 50, 'k': 10, 'seed': 42, 'nlist': 128, 'nprobes': [1, 4, 8, 16],
                'hnsw_m': 16, 'ef_search': [16, 64, 128], 'pq_m': 64, 'pq_bits': 8,
                'repeats': 3, 'warmups': 1, 'threads': 1}
RETRIEVAL_DEFAULTS = {'k': 10, 'candidate_k': 50, 'pool': 100}
RETRIEVAL_KEYS = {'enabled', 'queries', 'query_vectors', 'k', 'candidate_k', 'pool', 'methods',
                  'hnsw', 'models'}
JUDGE_KEYS = {'enabled', 'model', 'max_usd', 'price_per_million_input', 'price_per_million_output',
              'max_new_per_run', 'estimated_usd_per_pair', 'ledger'}


class BudgetExceeded(Exception):
    pass


# --------------------------------------------------------------------------- config

def load_config(path: str) -> dict:
    config = json.loads(get_bytes(path))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    for key in ('name', 'root', 'query', 'out_root'):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f'Config requires a non-empty string "{key}"')
    if not all(c.isalnum() or c in '-_' for c in config['name']):
        raise ValueError('Config "name" may contain only letters, digits, "-" and "_"')
    unknown = set(config.get('ann', {})) - set(ANN_DEFAULTS) - {'enabled'}
    if unknown:
        raise ValueError(f'Unknown ann settings: {sorted(unknown)}')
    retrieval = config.get('retrieval', {})
    unknown = set(retrieval) - RETRIEVAL_KEYS
    if unknown:
        raise ValueError(f'Unknown retrieval settings: {sorted(unknown)}')
    if retrieval.get('enabled', True):
        for key in ('queries', 'query_vectors'):
            if not retrieval.get(key):
                raise ValueError(f'retrieval.{key} is required when retrieval is enabled')
    if 'methods' in retrieval:
        from .methods import parse_methods
        parse_methods(retrieval['methods'])
    report = config.get('report', {})
    judge = config.get('judge')
    if (report.get('enabled', True) or judge) and not report.get('labels'):
        raise ValueError('report.labels is required for the report and judge stages')
    if judge:  # validated whenever present; it only RUNS with "enabled": true or --stages judge
        unknown = set(judge) - JUDGE_KEYS
        if unknown:
            raise ValueError(f'Unknown judge settings: {sorted(unknown)}')
        for key in ('model', 'max_usd', 'price_per_million_input', 'price_per_million_output'):
            if key not in judge:
                raise ValueError(f'judge.{key} is required (prices are not guessed; check your provider)')
        if not 0 < float(judge['max_usd']) <= 100:
            raise ValueError('judge.max_usd must be in (0, 100]')


def enabled_stages(config: dict, requested: list[str] | None) -> list[str]:
    defaults = {'judge': False}  # spending money is opt-in
    stages = requested or [s for s in STAGES
                           if config.get(s, {}).get('enabled', defaults.get(s, True))]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise ValueError(f'Unknown stages {bad}; choose from {list(STAGES)}')
    return [s for s in STAGES if s in stages]  # always execute in pipeline order


def make_run_id(config: dict, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    return f"{now:%Y%m%dT%H%M%SZ}-{config['name']}-{digest}"


def git_commit() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             timeout=5, check=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True,
                               timeout=5, check=True).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return None


def local_copy(uri: str, workdir: Path) -> str:
    """Existing readers take local paths; download s3:// inputs once into the run's temp dir."""
    if not uri.startswith('s3://'):
        return uri
    path = workdir / hashlib.sha256(uri.encode()).hexdigest()[:12] / Path(uri).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(get_bytes(uri))
    return str(path)


# --------------------------------------------------------------------------- corpus

def load_corpus(root: str, query: str, *, need_papers: bool):
    """Return (papers or None, vectors, manifest). Validation lives in ann_benchmark.read_corpus."""
    from .ann_benchmark import read_corpus
    vectors, manifest = read_corpus(root, query)
    papers = None
    if need_papers:
        from .index import curated_uri, load_papers
        papers = load_papers(curated_uri(root, query))
        if [str(p['id']) for p in papers] != json.loads(get_bytes(manifest['paper_ids_uri'])):
            raise ValueError('Corpus and embedding paper IDs differ')
    return papers, vectors, manifest


# --------------------------------------------------------------------------- stages

def stage_ann(config: dict, vectors, manifest: dict) -> dict:
    from .ann_benchmark import run_benchmark
    s = {**ANN_DEFAULTS, **{k: v for k, v in config.get('ann', {}).items() if k != 'enabled'}}
    result = run_benchmark(vectors, query_count=s['queries'], k=s['k'], seed=s['seed'],
                           nlist=s['nlist'], nprobes=tuple(s['nprobes']), hnsw_m=s['hnsw_m'],
                           ef_search=tuple(s['ef_search']), pq_m=s['pq_m'], pq_bits=s['pq_bits'],
                           repeats=s['repeats'], warmups=s['warmups'], threads=s['threads'])
    result['source'] = {k: manifest.get(k) for k in
                        ('vectors_uri', 'source_digest', 'model', 'paper_adapter')}
    return result


def stage_retrieval(config: dict, papers, vectors, manifest: dict, workdir: Path,
                    reranker_factories: dict | None = None) -> dict:
    from .eval_retrieve import QUERY_ADAPTER, load_query_vectors, read_queries, retrieve
    s = {**RETRIEVAL_DEFAULTS, **config.get('retrieval', {})}
    queries = read_queries(local_copy(s['queries'], workdir))
    qvecs = load_query_vectors(local_copy(s['query_vectors'], workdir), queries)
    if 'methods' in s:
        from .methods import run_methods
        result = run_methods(papers, vectors, queries, qvecs, methods=s['methods'], k=s['k'],
                             candidate_k=s['candidate_k'], pool=s['pool'], hnsw=s.get('hnsw'),
                             models=s.get('models'), reranker_factories=reranker_factories)
    else:  # original three methods, unchanged code path
        result = retrieve(papers, vectors, queries, qvecs, k=s['k'], candidate_k=s['candidate_k'])
    result['corpus']['embedding_model'] = manifest.get('model')
    result['corpus']['query_adapter'] = QUERY_ADAPTER
    return result


def label_coverage(retrieval: dict, labels: dict) -> dict:
    """How many top-k (query, paper) pairs lack a judgment, overall and per method."""
    k = int(retrieval.get('k', 10))
    missing, per_method = set(), {}
    for row in retrieval['results']:
        for method, hits in row['methods'].items():
            gaps = {(row['query_id'], h['id']) for h in hits[:k]} - labels.keys()
            per_method[method] = per_method.get(method, 0) + len(gaps)
            missing |= gaps
    pooled = {(row['query_id'], h['id']) for row in retrieval['results']
              for hits in row['methods'].values() for h in hits[:k]}
    return {'pooled_pairs': len(pooled), 'judged': len(pooled) - len(missing),
            'missing': len(missing), 'missing_by_method': per_method,
            'missing_examples': [{'query_id': q, 'paper_id': p} for q, p in sorted(missing)[:5]]}


def _labels_and_ledger(config: dict) -> tuple[str, str]:
    labels_uri = config['report']['labels']
    ledger = config.get('judge', {}).get('ledger') or labels_uri.rsplit('/', 1)[0] + '/judge_spend.json'
    return labels_uri, ledger


def stage_judge(config: dict, retrieval: dict, workdir: Path, *, run_id: str, call=None) -> dict:
    """Judge missing pairs only; stop before the cumulative cap; always sync labels + ledger."""
    import os
    from .eval_judge import call_openai, candidates, judge, read_labels
    j = config['judge']
    labels_uri, ledger_uri = _labels_and_ledger(config)
    ledger = json.loads(get_bytes(ledger_uri)) if exists(ledger_uri) else {'total_usd': 0., 'runs': []}
    remaining = float(j['max_usd']) - float(ledger['total_usd'])
    local = labels_uri if not labels_uri.startswith('s3://') else str(workdir / 'judge-labels.jsonl')
    if labels_uri.startswith('s3://') and exists(labels_uri):
        Path(local).write_bytes(get_bytes(labels_uri))
    digest = retrieval['corpus']['source_digest']
    before = read_labels(local, digest)
    pending = [p for p in candidates(retrieval) if (p['query_id'], p['paper_id']) not in before]
    api_key = os.environ.get('OPENAI_API_KEY')
    if pending and call is None and not api_key:
        raise ValueError('OPENAI_API_KEY is not set (on AWS it is read from SSM Parameter Store)')
    call = call or (lambda pair: call_openai(pair, model=j['model'], api_key=api_key))
    in_price = float(j['price_per_million_input']) / 1e6
    out_price = float(j['price_per_million_output']) / 1e6
    floor = float(j.get('estimated_usd_per_pair', 0.0005))
    spent, judged = 0., 0

    def judge_fn(pair):
        nonlocal spent, judged
        projected = max(floor, 1.5 * spent / judged) if judged else floor
        if spent + projected > remaining:
            raise BudgetExceeded
        result = call(pair)
        usage = result.get('usage') or {}
        spent += usage.get('prompt_tokens', 0) * in_price + usage.get('completion_tokens', 0) * out_price
        judged += 1
        return result

    stopped = None
    try:
        judge(retrieval, labels_path=local, model=j['model'], max_new=j.get('max_new_per_run'),
              judge_fn=judge_fn)
    except BudgetExceeded:
        stopped = 'budget'
    finally:  # persist whatever was judged, even if the API failed midway
        if labels_uri.startswith('s3://') and Path(local).exists():
            put_bytes(labels_uri, Path(local).read_bytes())
        ledger['total_usd'] = round(float(ledger['total_usd']) + spent, 6)
        ledger['runs'].append({'run_id': run_id, 'model': j['model'], 'pairs_judged': judged,
                               'usd': round(spent, 6), 'at': datetime.now(timezone.utc).isoformat()})
        put_bytes(ledger_uri, (json.dumps(ledger, indent=2) + '\n').encode())
    still_missing = len(pending) - judged
    return {'status': 'STOPPED_AT_BUDGET' if stopped else 'complete', 'pairs_needed': len(pending),
            'pairs_judged': judged, 'still_missing': still_missing, 'usd_this_run': round(spent, 6),
            'usd_total': ledger['total_usd'], 'usd_cap': float(j['max_usd']), 'model': j['model'],
            'labels': labels_uri, 'ledger': ledger_uri}


def stage_report(config: dict, retrieval: dict, workdir: Path) -> dict:
    from .eval_judge import evaluate, read_labels
    labels = read_labels(local_copy(config['report']['labels'], workdir),
                         retrieval['corpus']['source_digest'])
    coverage = label_coverage(retrieval, labels)
    if coverage['missing']:
        return {'status': 'INCOMPLETE_UNJUDGED_PAIRS', 'coverage': coverage,
                'note': 'No scores computed: unjudged pairs would bias NDCG. Run capped judging first.'}
    report = evaluate(retrieval, labels, methods=retrieval.get('methods'))
    if retrieval.get('setup'):
        report['setup'] = retrieval['setup']
    report['coverage'] = coverage
    return report


# --------------------------------------------------------------------------- orchestration

def run(config: dict, *, stages: list[str] | None = None, from_run: str | None = None,
        loader=load_corpus, now: datetime | None = None, reranker_factories: dict | None = None,
        judge_call=None) -> dict:
    validate_config(config)
    stages = enabled_stages(config, stages)
    if {'judge', 'report'} & set(stages) and 'retrieval' not in stages and not from_run:
        raise ValueError('judge/report need retrieval results: add the retrieval stage or --from-run')
    if 'judge' in stages and 'judge' not in config:
        raise ValueError('The judge stage needs a "judge" section in the config')
    run_id = make_run_id(config, now)
    run_dir = join(config['out_root'], run_id)
    manifest = {'run_id': run_id, 'run_dir': run_dir, 'config': config, 'stages': stages,
                'from_run': from_run, 'git_commit': git_commit(),
                'started_at': datetime.now(timezone.utc).isoformat(), 'results': {}}

    def save(name: str, data: dict) -> str:
        uri = join(run_dir, f'{name}.json')
        put_bytes(uri, (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode())
        return uri

    save('run_manifest', manifest)
    papers = vectors = corpus_manifest = None
    if {'ann', 'retrieval'} & set(stages):
        papers, vectors, corpus_manifest = loader(config['root'], config['query'],
                                                  need_papers='retrieval' in stages)
    retrieval = None
    if from_run:
        uri = join(from_run, 'retrieval.json')
        if not exists(uri):
            raise ValueError(f'No retrieval.json in {from_run}')
        retrieval = json.loads(get_bytes(uri))

    with tempfile.TemporaryDirectory(prefix='scholarsearch-run-') as tmp:
        workdir = Path(tmp)
        for stage in stages:
            t0 = time.perf_counter()
            entry = {'status': 'running'}
            manifest['results'][stage] = entry
            try:
                if stage == 'ann':
                    data = stage_ann(config, vectors, corpus_manifest)
                elif stage == 'retrieval':
                    data = retrieval = stage_retrieval(config, papers, vectors, corpus_manifest,
                                                       workdir, reranker_factories)
                elif stage == 'judge':
                    data = stage_judge(config, retrieval, workdir, run_id=run_id, call=judge_call)
                else:
                    data = stage_report(config, retrieval, workdir)
                entry.update(status=data.get('status', 'complete') if stage in ('judge', 'report')
                             else 'complete', output=save(stage, data))
            except Exception as exc:  # record the failure, keep earlier outputs, then stop
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                             traceback=traceback.format_exc()[-2000:])
                raise
            finally:
                entry['seconds'] = round(time.perf_counter() - t0, 3)
                manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
                save('run_manifest', manifest)
            print(f'[{run_id}] {stage}: {entry["status"]} in {entry["seconds"]}s', flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    from .config import load_env
    load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='JSON config file (local or s3://)')
    p.add_argument('--stages', nargs='+', choices=STAGES, help='Subset of stages; default: all enabled')
    p.add_argument('--out-root', help='Override config out_root (local folder or s3://)')
    p.add_argument('--from-run', help='Reuse retrieval.json from an earlier run folder')
    p.add_argument('--dry-run', action='store_true', help='Validate config and print the plan only')
    args = p.parse_args(argv)
    config = load_config(args.config)
    if args.out_root:
        config['out_root'] = args.out_root
    if args.dry_run:
        stages = enabled_stages(config, args.stages)
        print(json.dumps({'run_id': make_run_id(config), 'stages': stages,
                          'out_dir': join(config['out_root'], make_run_id(config))}, indent=2))
        return
    manifest = run(config, stages=args.stages, from_run=args.from_run)
    print(f"Done. Results in {manifest['run_dir']}")


if __name__ == '__main__':
    main()

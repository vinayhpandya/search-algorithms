"""ScholarSearch frozen-corpus relevance evaluation V1.

Standard-library core. Reading the 15K snapshot requires the existing ScholarSearch
storage/index code and Polars already present in the project. No model calls or indexing.

Commands: snapshot, freeze, validate-run, pool, judgments, score, demo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CORPUS_ROOT = 's3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b'
EXPECTED_DIGEST = '379edc7bf2f15f55'
EXPECTED_COUNT = 15000
SPLIT_SEED = 42


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + '\n')


def load_json(path):
    return json.loads(Path(path).read_text())


def write_tsv(path, fields, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def read_tsv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in data))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def openalex_id(paper_id):
    s = str(paper_id).strip()
    s = s.removeprefix('https://openalex.org/').removeprefix('http://openalex.org/')
    if not re.fullmatch(r'W\d+', s):
        raise ValueError(f'Invalid OpenAlex paper ID: {paper_id!r}')
    return s


def snapshot_from_papers(papers, *, root, query, expected_digest=EXPECTED_DIGEST,
                         expected_count=EXPECTED_COUNT):
    # Reuse the exact production source digest, not a new approximation.
    from scholarsearch.embeddings import source_digest
    from scholarsearch.index import curated_uri
    ids = [openalex_id(p['id']) for p in papers]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate OpenAlex paper IDs')
    digest = source_digest(papers)
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f'Expected {expected_count} papers, got {len(ids)}')
    if expected_digest and digest != expected_digest:
        raise ValueError(f'Corpus digest mismatch: expected {expected_digest}, got {digest}')
    if any(not str(p.get('title') or '').strip() for p in papers):
        raise ValueError('Corpus has missing titles')
    term_counter = Counter()
    stop = {'the','and','for','with','from','using','based','via','of','on','to','in','a','an',
            'by','approach','study','new','towards','method','methods','analysis','research'}
    for p in papers:
        terms = set(re.findall(r'[a-z]{4,}', (p.get('title') or '').lower())) - stop
        term_counter.update(terms)
    manifest = {
        'schema': 'scholarsearch.corpus.v1',
        'source_uri': curated_uri(root, query),
        'source_digest': digest,
        'documents': len(ids),
        'id_namespace': 'OpenAlex W-id',
        'paper_ids_sha256': hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        'paper_ids': ids,
        'top_title_terms': [{'term': t, 'papers': n} for t, n in term_counter.most_common(80)],
        'evaluation_status': 'unjudged',
    }
    return manifest


def snapshot(args):
    from scholarsearch.index import curated_uri, load_papers
    papers = load_papers(curated_uri(args.root, args.query))
    manifest = snapshot_from_papers(papers, root=args.root, query=args.query,
                                    expected_digest=args.expected_digest,
                                    expected_count=args.expected_count)
    out = Path(args.out)
    if out.exists() and not args.force:
        raise ValueError(f'{out} already exists. No overwrite without --force')
    write_json(out, manifest)
    print(f"Verified {manifest['documents']} papers digest={manifest['source_digest']} -> {out}")
    print('Top title terms:', ', '.join(x['term'] for x in manifest['top_title_terms'][:12]))


def prepare_template(args):
    manifest = load_json(args.manifest)
    if manifest['schema'] != 'scholarsearch.corpus.v1':
        raise ValueError('Invalid corpus manifest')
    # Deliberately blank: no false assertion these queries are validated/grounded.
    write_tsv(args.out, ['query_id','query','intent','reviewed'],
              [{'query_id': f'q{i:03d}', 'query':'', 'intent':'', 'reviewed':''}
               for i in range(1, args.count + 1)])
    print(f'Created {args.count} empty query slots -> {args.out}')


def freeze_queries(rows, *, manifest, dev=40, seed=SPLIT_SEED):
    if manifest.get('schema') != 'scholarsearch.corpus.v1':
        raise ValueError('Invalid corpus manifest')
    if not rows or not 0 < dev < len(rows):
        raise ValueError('Need queries with both dev and test splits')
    ids, normalized = set(), set()
    for r in rows:
        qid = (r.get('query_id') or '').strip()
        query = ' '.join((r.get('query') or '').split())
        if not re.fullmatch(r'q\d{3,5}', qid) or qid in ids:
            raise ValueError(f'Duplicate/invalid query_id {qid!r}')
        if len(query) < 12 or query.casefold() in normalized:
            raise ValueError(f'Missing/short/duplicate query for {qid}')
        if (r.get('reviewed') or '').strip().lower() != 'yes':
            raise ValueError(f'{qid} must be explicitly reviewed=yes')
        ids.add(qid)
        normalized.add(query.casefold())
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    dev_ids = set(shuffled[:dev])
    frozen = [{'query_id': r['query_id'].strip(),
               'text': ' '.join(r['query'].split()),
               'intent': r.get('intent','').strip(),
               'split': 'dev' if r['query_id'].strip() in dev_ids else 'test'}
              for r in rows]
    return frozen


def freeze(args):
    manifest = load_json(args.manifest)
    queries = freeze_queries(read_tsv(args.input), manifest=manifest, dev=args.dev, seed=args.seed)
    out = Path(args.out)
    if out.exists() and not args.force:
        raise ValueError(f'{out} already exists. No overwrite without --force')
    write_jsonl(out, queries)
    write_json(out.with_name('benchmark_manifest.json'), {
        'schema':'scholarsearch.evaluation.v1',
        'corpus_digest':manifest['source_digest'],
        'corpus_documents':manifest['documents'],
        'query_file':out.name,
        'query_sha256':sha256_file(out),
        'splits':dict(Counter(r['split'] for r in queries)),
        'relevance_scale':'0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant',
        'judgment_status':'not_started',
        'historical_evaluation_reused':False,
    })
    print(f'Frozen {len(queries)} reviewed queries -> {out}')


def parse_run(path, valid_qids=None, valid_ids=None):
    data = defaultdict(list)
    run_name = None
    seen = set()
    for line_num, raw in enumerate(Path(path).read_text().splitlines(), 1):
        if not raw.strip():
            continue
        cols = raw.split()
        if len(cols) != 6 or cols[1] != 'Q0':
            raise ValueError(f'{path}:{line_num}: expected qid Q0 paper_id rank score run_name')
        qid, _, pid, rank, score, name = cols
        pid = openalex_id(pid)
        if valid_qids is not None and qid not in valid_qids:
            raise ValueError(f'{path}: unknown query {qid}')
        if valid_ids is not None and pid not in valid_ids:
            raise ValueError(f'{path}: paper not in frozen corpus: {pid}')
        try:
            rank = int(rank)
            score = float(score)
        except ValueError as e:
            raise ValueError(f'{path}:{line_num}: rank/score invalid') from e
        if rank < 1 or not math.isfinite(score) or (qid, pid) in seen:
            raise ValueError(f'{path}:{line_num}: invalid rank/score or duplicate paper')
        seen.add((qid, pid))
        if run_name is not None and name != run_name:
            raise ValueError(f'{path}: more than one run name')
        run_name = name
        data[qid].append((rank, pid, score))
    for qid, entries in data.items():
        ranks = sorted(x[0] for x in entries)
        if ranks != list(range(1,len(entries)+1)):
            raise ValueError(f'{path}: rank gap or duplicate for {qid}')
        data[qid] = [pid for rank, pid, score in sorted(entries)]
    if not data:
        raise ValueError(f'Empty run: {path}')
    return run_name, dict(data)


def read_benchmark(args):
    manifest = load_json(args.manifest)
    queries = read_jsonl(args.queries)
    bpath = Path(args.queries).with_name('benchmark_manifest.json')
    bench = load_json(bpath)
    if (manifest['source_digest'] != bench['corpus_digest'] or
        sha256_file(args.queries) != bench['query_sha256']):
        raise ValueError('Corpus or frozen query checksum mismatch')
    return manifest, queries


def load_runs(args, manifest, queries):
    qids = {q['query_id'] for q in queries}
    pids = set(manifest['paper_ids'])
    runs = {}
    for file in args.runs:
        name, rows = parse_run(file, qids, pids)
        if name in runs:
            raise ValueError(f'Duplicate run name {name}')
        runs[name] = rows
    return runs


def validate_run(args):
    manifest, queries = read_benchmark(args)
    runs = load_runs(args, manifest, queries)
    for name, rows in runs.items():
        print(f'VALID {name}: queries={len(rows)}, results={sum(map(len,rows.values()))}')


def pool(args):
    manifest, queries = read_benchmark(args)
    runs = load_runs(args, manifest, queries)
    seen = set()
    rows = []
    provenance = defaultdict(list)
    for q in queries:
        qid = q['query_id']
        for name, run in runs.items():
            for rank, pid in enumerate(run.get(qid, [])[:args.depth], 1):
                provenance[(qid,pid)].append({'run':name,'rank':rank})
                if (qid,pid) not in seen:
                    seen.add((qid,pid))
                    rows.append({'query_id':qid,'query':q['text'], 'paper_id':pid,
                                 'relevance':'','judge':'','notes':''})
    # Requires a separate read to materialize title/abstract only; in this PR
    # we explicitly do not make judges label opaque IDs without paper context.
    if not args.allow_id_only:
        raise ValueError('Judging needs paper titles/abstracts. Use --corpus-jsonl PATH '
                         'with frozen paper rows; --allow-id-only is validation only.')
    write_tsv(args.out, ['query_id','query','paper_id','relevance','judge','notes'], rows)
    write_json(str(args.out) + '.provenance.json',
               {f'{qid}\t{pid}': values for (qid,pid),values in provenance.items()})
    print(f'Pooled {len(rows)} candidates -> {args.out}')


def annotate_pool(args):
    manifest, queries = read_benchmark(args)
    expected = set(manifest['paper_ids'])
    papers = {}
    for row in read_jsonl(args.corpus_jsonl):
        pid = openalex_id(row['id'])
        if pid in papers:
            raise ValueError(f'Duplicate paper in corpus export: {pid}')
        papers[pid] = row
    if set(papers) != expected:
        raise ValueError('Exported paper IDs do not match frozen manifest')
    original = read_tsv(args.pool)
    enriched = []
    for row in original:
        p = papers[openalex_id(row['paper_id'])]
        enriched.append({**row, 'title':p.get('title') or '', 'abstract':p.get('abstract') or ''})
    # Avoid leaking method identity into reviewer sheet.
    write_tsv(args.out, ['query_id','query','paper_id','title','abstract','relevance','judge','notes'], enriched)
    print(f'Blind review sheet: {len(enriched)} query-paper pairs -> {args.out}')


def export_corpus(args):
    from scholarsearch.index import curated_uri, load_papers
    manifest = load_json(args.manifest)
    papers = load_papers(manifest['source_uri'])
    from scholarsearch.embeddings import source_digest
    if source_digest(papers) != manifest['source_digest']:
        raise ValueError('Dataset changed since snapshot')
    ids = [openalex_id(p['id']) for p in papers]
    if ids != manifest['paper_ids']:
        raise ValueError('Corpus ID order changed')
    write_jsonl(args.out, [{'id':p['id'],'title':p.get('title') or '',
                             'abstract':p.get('abstract') or ''} for p in papers])
    print(f'Exported {len(papers)} titles and abstracts -> {args.out}')


def judgments(args):
    manifest, queries = read_benchmark(args)
    qids = {q['query_id'] for q in queries}
    pids = set(manifest['paper_ids'])
    judged = {}
    for row in read_tsv(args.reviewed):
        qid, pid = row['query_id'], openalex_id(row['paper_id'])
        if qid not in qids or pid not in pids or (qid,pid) in judged:
            raise ValueError(f'Unknown or duplicate judged pair {qid}/{pid}')
        label = row.get('relevance','').strip()
        if not label:
            continue
        if label not in ('0','1','2','3'):
            raise ValueError(f'Invalid relevance {label!r}')
        if not (row.get('judge') or '').strip():
            raise ValueError('Each judgment needs a judge identifier for provenance')
        judged[(qid,pid)] = int(label)
    rows = [f'{qid} 0 {pid} {rel}' for (qid,pid), rel in sorted(judged.items())]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(rows) + ('\n' if rows else ''))
    write_json(str(args.out)+'.metadata.json', {
        'corpus_digest':manifest['source_digest'],
        'query_sha256':sha256_file(args.queries),
        'qrels_sha256':sha256_file(args.out),
        'judged_pairs':len(judged),
        'scale':'0..3',
        'provenance':'reviewed TSV; label authors are in the source review sheet',
    })
    print(f'Exported {len(judged)} explicit judgments -> {args.out}')


def read_qrels(path):
    out = {}
    for ln, line in enumerate(Path(path).read_text().splitlines(),1):
        fields = line.split()
        if len(fields) != 4 or fields[1] != '0' or fields[3] not in ('0','1','2','3'):
            raise ValueError(f'Invalid qrels line {ln}')
        key = fields[0], openalex_id(fields[2])
        if key in out:
            raise ValueError(f'Duplicate qrel {key}')
        out[key] = int(fields[3])
    return out


def ndcg_at_k(labels, ideal_labels, k):
    def dcg(values):
        return sum((2**r-1)/math.log2(i+2) for i,r in enumerate(values[:k]))
    denom = dcg(sorted(ideal_labels, reverse=True))
    return dcg(labels)/denom if denom else 0.


def score_runs(runs, queries, qrels, *, split, k):
    evaluated = [q for q in queries if split == 'all' or q['split'] == split]
    if not evaluated:
        raise ValueError('No queries in selected split')
    output = {}
    for name, run in runs.items():
        results, gaps = [], []
        for q in evaluated:
            qid = q['query_id']
            found = run.get(qid, [])
            top = found[:k]
            known = {pid: rel for (query_id,pid),rel in qrels.items() if query_id == qid}
            unknown = [pid for pid in top if (qid,pid) not in qrels]
            if len(top) < k or unknown or not any(x>=2 for x in known.values()):
                gaps.append({'query_id':qid,'returned':len(top),'unjudged_top_k':len(unknown),
                             'known_relevant':sum(x>=2 for x in known.values())})
                continue
            labels = [known[pid] for pid in top]
            ideal = list(known.values())
            first = next((1/(i+1) for i,v in enumerate(labels) if v>=2),0.)
            results.append({'query_id':qid, 'ndcg_at_k':ndcg_at_k(labels,ideal,k),
                            'mrr_at_k':first})
        output[name] = {
            'status':'COMPLETE_POOLED_JUDGMENTS' if not gaps else 'INCOMPLETE_JUDGMENTS_NO_AGGREGATE_SCORE',
            'queries_requested':len(evaluated), 'queries_covered':len(results),
            'missing':gaps,
            'ndcg_at_k':round(sum(r['ndcg_at_k'] for r in results)/len(results),6) if not gaps else None,
            'mrr_at_k':round(sum(r['mrr_at_k'] for r in results)/len(results),6) if not gaps else None,
            'per_query':results if not gaps else [],
        }
    return output


def score(args):
    manifest, queries = read_benchmark(args)
    runs = load_runs(args,manifest,queries)
    metadata = load_json(str(args.qrels)+'.metadata.json')
    if (metadata['corpus_digest'] != manifest['source_digest'] or
        metadata['query_sha256'] != sha256_file(args.queries) or
        metadata['qrels_sha256'] != sha256_file(args.qrels)):
        raise ValueError('Qrels do not match benchmark/corpus or have changed')
    qrels = read_qrels(args.qrels)
    report = {'schema':'scholarsearch.eval.report.v1',
              'corpus_digest':manifest['source_digest'],
              'query_sha256':sha256_file(args.queries),
              'qrels_sha256':sha256_file(args.qrels),
              'split':args.split,'k':args.k,
              'metrics':score_runs(runs,queries,qrels,split=args.split,k=args.k),
              'caveat':'Incomplete relevance judgments: pooled qrels are not exhaustive. '
                       'Only score if all observed top-k are explicitly judged and there are judged positives. '
                       'Candidate recall is NOT estimated from unjudged papers.'}
    write_json(args.out,report)
    print(json.dumps({name:{key:val for key,val in data.items() if key!='per_query' and key!='missing'}
                      for name,data in report['metrics'].items()},indent=2))


def demo(args):
    # Toy examples ONLY; never presented as benchmark measurements.
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    papers = [{'id':f'W{i}','title':f'Toy paper {i}','abstract':'toy only'} for i in range(1,7)]
    # Corpus digest for synthetic toy, deliberately not the real digest.
    manifest = {'schema':'scholarsearch.corpus.v1','source_digest':'TOY_NOT_REAL',
                'documents':6,'paper_ids':[p['id'] for p in papers]}
    write_json(out/'corpus_manifest.json',manifest)
    rows = [{'query_id':f'q{i:03d}','query':f'Toy research intent number {i}',
             'intent':'toy','reviewed':'yes'} for i in range(1,5)]
    write_tsv(out/'query_draft.tsv', ['query_id','query','intent','reviewed'],rows)
    frozen = freeze_queries(rows,manifest=manifest,dev=2)
    write_jsonl(out/'queries.jsonl',frozen)
    write_json(out/'benchmark_manifest.json',{
        'schema':'scholarsearch.evaluation.v1','corpus_digest':'TOY_NOT_REAL',
        'query_sha256':sha256_file(out/'queries.jsonl'),
        'historical_evaluation_reused':False})
    (out/'runs').mkdir(exist_ok=True)
    ranks = [1,2,3,4,5,6]
    for name, ordering in [('bm25_toy',ranks),('dense_toy',[2,1,3,4,5,6])]:
        lines = [f"{q['query_id']} Q0 W{pid} {rank} {7-rank}.0 {name}"
                 for q in frozen for rank,pid in enumerate(ordering,1)]
        (out/'runs'/f'{name}.trec').write_text('\n'.join(lines)+'\n')
    judged = {(q['query_id'],f'W{i}'): (3 if i==2 else 2 if i==1 else 0)
              for q in frozen for i in range(1,7)}
    qfile = out/'qrels.tsv'
    qfile.write_text(''.join(f'{qid} 0 {pid} {r}\n' for (qid,pid),r in sorted(judged.items())))
    write_json(str(qfile)+'.metadata.json', {
        'corpus_digest':'TOY_NOT_REAL',
        'query_sha256':sha256_file(out/'queries.jsonl'),
        'qrels_sha256':sha256_file(qfile),
        'toy_only':True})
    runs = {name:parse_run(out/'runs'/f'{name}.trec')[1] for name in ('bm25_toy','dense_toy')}
    report = {'TOY_DATA_DO_NOT_PRESENT_AS_REAL':True,
              'metrics':score_runs(runs,frozen,judged,split='test',k=3)}
    write_json(out/'toy_report.json',report)
    print('Toy pipeline verified ->',out/'toy_report.json')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='cmd',required=True)
    s = subs.add_parser('snapshot',help='Read and verify pinned S3 corpus without building an index')
    s.add_argument('--root',default=CORPUS_ROOT)
    s.add_argument('--query',default='malware detection')
    s.add_argument('--expected-digest',default=EXPECTED_DIGEST)
    s.add_argument('--expected-count',type=int,default=EXPECTED_COUNT)
    s.add_argument('--out',default='artifacts/evaluation/v1/corpus_manifest.json')
    s.add_argument('--force',action='store_true')
    s.set_defaults(func=snapshot)
    s = subs.add_parser('template',help='Prepare blank query authoring template')
    s.add_argument('--manifest',required=True)
    s.add_argument('--count',type=int,default=100)
    s.add_argument('--out',required=True)
    s.set_defaults(func=prepare_template)
    s = subs.add_parser('freeze',help='Freeze explicitly reviewed new queries')
    s.add_argument('--manifest',required=True)
    s.add_argument('--input',required=True)
    s.add_argument('--dev',type=int,default=40)
    s.add_argument('--seed',type=int,default=SPLIT_SEED)
    s.add_argument('--out',required=True)
    s.add_argument('--force',action='store_true')
    s.set_defaults(func=freeze)
    def benchmark_args(s):
        s.add_argument('--manifest',required=True)
        s.add_argument('--queries',required=True)
    s = subs.add_parser('validate-run',help='Validate TREC runs vs frozen corpus IDs and query IDs')
    benchmark_args(s)
    s.add_argument('--runs',nargs='+',required=True)
    s.set_defaults(func=validate_run)
    s = subs.add_parser('pool',help='Pool candidates, with method-blind judgment sheet')
    benchmark_args(s)
    s.add_argument('--runs',nargs='+',required=True)
    s.add_argument('--depth',type=int,default=10)
    s.add_argument('--out',required=True)
    s.add_argument('--allow-id-only',action='store_true',help='Output IDs only; NOT suitable for actual judging')
    s.set_defaults(func=pool)
    s = subs.add_parser('export-corpus',help='Export paper titles and abstracts from verified frozen S3 corpus')
    s.add_argument('--manifest',required=True)
    s.add_argument('--out',required=True)
    s.set_defaults(func=export_corpus)
    s = subs.add_parser('annotate-pool',help='Join candidate IDs to verified paper text for judging')
    benchmark_args(s)
    s.add_argument('--pool',required=True)
    s.add_argument('--corpus-jsonl',required=True)
    s.add_argument('--out',required=True)
    s.set_defaults(func=annotate_pool)
    s = subs.add_parser('judgments',help='Export explicit human-reviewed labels')
    benchmark_args(s)
    s.add_argument('--reviewed',required=True)
    s.add_argument('--out',required=True)
    s.set_defaults(func=judgments)
    s = subs.add_parser('score',help='Refuse aggregate score if top-k judgments are incomplete')
    benchmark_args(s)
    s.add_argument('--runs',nargs='+',required=True)
    s.add_argument('--qrels',required=True)
    s.add_argument('--split',choices=['dev','test','all'],default='test')
    s.add_argument('--k',type=int,default=10)
    s.add_argument('--out',required=True)
    s.set_defaults(func=score)
    s = subs.add_parser('demo',help='Generate toy data and prove scoring works without AWS')
    s.add_argument('--out',default='artifacts/evaluation/toy-demo')
    s.set_defaults(func=demo)
    args = p.parse_args(argv)
    if getattr(args,'depth',1) <= 0 or getattr(args,'k',1) <= 0 or getattr(args,'count',1) <= 0:
        p.error('depth/k/count must be positive')
    args.func(args)


if __name__ == '__main__':
    main()

import json
from pathlib import Path
import tempfile
import unittest

from scholarsearch.eval_judge import candidates, judge, read_labels, evaluate, load_retrieval


def example():
    def hit(id):
        return {'id': id, 'title': f'Paper {id}', 'abstract': 'An abstract', 'score': 1.0}
    return {'status': 'UNJUDGED_RETRIEVAL_ONLY', 'corpus': {'source_digest': 'abcdef'},
            'k': 2, 'results': [{'query_id':'q001','query':'malware', 'methods': {
                'bm25': [hit('a'), hit('b')], 'dense': [hit('c'), hit('a')],
                'hybrid': [hit('a'), hit('c')]}}]}


class JudgeTests(unittest.TestCase):
    def test_dedup_and_resume(self):
        data=example()
        self.assertEqual(len(candidates(data)),3)
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/'labels.jsonl')
            calls=[]
            def fake(pair):
                calls.append(pair['paper_id'])
                return {'label': {'a':2,'b':0,'c':1}[pair['paper_id']], 'reason':'test'}
            first=judge(data, labels_path=p, model='fake', max_new=2, judge_fn=fake)
            self.assertEqual(first['still_missing'],1)
            second=judge(data, labels_path=p, model='fake', judge_fn=fake)
            self.assertEqual(second['newly_judged'],1)
            self.assertEqual(len(calls),3)
            labels=read_labels(p,'abcdef')
            result=evaluate(data,labels)
            self.assertEqual(result['queries'],1)
            self.assertAlmostEqual(result['metrics']['bm25']['mrr_at_10_rel2'],1.0)
            self.assertAlmostEqual(result['metrics']['dense']['mrr_at_10_rel2'],0.5)
            self.assertGreater(result['metrics']['bm25']['ndcg_at_10'],result['metrics']['dense']['ndcg_at_10'])

    def test_refuse_unjudged(self):
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            evaluate(example(), {})

    def test_refuse_mismatched_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.jsonl'
            p.write_text(json.dumps({'query_id':'q001','paper_id':'a','label':1,'source_digest':'wrong'})+'\n')
            with self.assertRaisesRegex(ValueError,'different corpus'):
                read_labels(str(p),'abcdef')

    def test_verify_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'retrieval.json'
            p.write_text(json.dumps(example()))
            self.assertEqual(len(load_retrieval(str(p))['results']),1)


if __name__=='__main__':
    unittest.main()

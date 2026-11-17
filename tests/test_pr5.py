import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scholarsearch.embeddings import MODEL, PAPER_ADAPTER, artifact_prefix, source_digest
from scholarsearch.index import curated_uri
from scholarsearch.query_embed import QUERY_ADAPTER, encode_query
from scholarsearch.search import (bm25_body, dense_body, load_query_vector, rrf, search)
from scholarsearch.vector_index import (build_index, hnsw_mapping, load_artifacts,
                                        validate_endpoint)
from scholarsearch.storage import join


class FakeEncoder:
    def encode(self, texts):
        a = np.zeros((len(texts), 768), np.float32)
        a[:, 0] = 3
        a[:, 1] = 4
        return a


class FakeClient:
    def __init__(self):
        self.calls = []
        self.indices = self
        self.existing = False
        self.doc_count = 0

    def ping(self):
        return True

    def exists(self, index):
        return self.existing

    def create(self, index, body):
        self.calls.append(('create', index, body))
        self.existing = True

    def delete(self, index):
        self.calls.append(('delete', index))
        self.existing = False

    def refresh(self, index):
        pass

    def count(self, index):
        return {'count': self.doc_count}

    def search(self, index, body):
        self.calls.append(('search', body))
        if 'knn' in body['query']:
            hits = [('b', 9.), ('c', 8.)]
        else:
            hits = [('a', 10.), ('b', 8.)]
        return {'hits': {'hits': [{'_id': x, '_score': score, '_source': {'title': x}} for x, score in hits]}}


class PR5Tests(unittest.TestCase):
    def test_hnsw_open_search_215(self):
        m = hnsw_mapping()
        e = m['mappings']['properties']['embedding']
        self.assertEqual((e['dimension'], e['method']['engine'], e['method']['space_type']),
                         (768, 'lucene', 'innerproduct'))
        self.assertEqual(m['settings']['index']['knn'], True)
        with self.assertRaises(ValueError):
            validate_endpoint('http://example.com:9200')

    def test_query_adapter_and_vector_metadata(self):
        result = encode_query('malware detection', FakeEncoder())
        self.assertEqual(result['adapter'], QUERY_ADAPTER)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'query.json'
            path.write_text(json.dumps(result))
            self.assertEqual(len(load_query_vector(str(path), 'malware detection')), 768)
            with self.assertRaises(ValueError):
                load_query_vector(str(path), 'different query')

    def test_artifacts_alignment_and_indexing(self):
        papers = [{'id': 'a', 'title': 'first'}, {'id': 'b', 'title': 'second'}]
        root, query = '/tmp/data', 'malware detection'
        digest = source_digest(papers)
        prefix = artifact_prefix(root, query, digest)
        vectors = np.zeros((2, 768), np.float32)
        vectors[:, 0] = 1
        out = io.BytesIO()
        np.save(out, vectors)
        manifest = {'source_digest': digest, 'source_uri': curated_uri(root, query),
                    'model': MODEL, 'paper_adapter': PAPER_ADAPTER, 'count': 2,
                    'dimension': 768, 'normalized': True,
                    'vectors_uri': join(prefix, 'vectors.npy'), 'paper_ids_uri': join(prefix, 'paper_ids.json')}
        objects = {join(prefix, 'manifest.json'): json.dumps(manifest).encode(),
                   manifest['vectors_uri']: out.getvalue(),
                   manifest['paper_ids_uri']: b'["a", "b"]'}
        with patch('scholarsearch.vector_index.load_papers', return_value=papers), \
             patch('scholarsearch.vector_index.get_bytes', side_effect=lambda uri: objects[uri]):
            self.assertEqual(load_artifacts(root, query)[1].shape, (2, 768))
            c = FakeClient()
            def bulk(client, actions, **kwargs):
                sent = list(actions)
                self.assertEqual([x['_id'] for x in sent], ['a', 'b'])
                self.assertEqual(len(sent[0]['_source']['embedding']), 768)
                client.doc_count = len(sent)
                return len(sent), []
            result = build_index(root=root, query=query, url='http://localhost:9200',
                                 index_name='new-hnsw', client=c, bulk=bulk)
            self.assertEqual(result['indexed'], 2)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                build_index(root=root, query=query, url='http://localhost:9200',
                            index_name='new-hnsw', client=c, bulk=bulk)
            objects[manifest['paper_ids_uri']] = b'["b", "a"]'
            with self.assertRaisesRegex(ValueError, 'ID order'):
                load_artifacts(root, query)

    def test_search_bm25_dense_hybrid_rrf(self):
        c = FakeClient()
        vector = [1.] + [0.] * 767
        results = search(text='malware detection', mode='hybrid', url='http://localhost:9200',
                         index='new-hnsw', limit=3, candidate_k=4, query_vector=vector, client=c)
        self.assertEqual([x['id'] for x in results], ['b', 'a', 'c'])
        self.assertEqual((results[0]['bm25_rank'], results[0]['dense_rank']), (2, 1))
        self.assertNotIn('embedding', results[0])
        self.assertEqual(len(c.calls), 2)
        self.assertIn('multi_match', c.calls[0][1]['query'])
        self.assertIn('knn', c.calls[1][1]['query'])
        self.assertEqual(search(text='q', mode='bm25', url='http://localhost:9200',
                                index='idx', limit=2, client=c)[0]['id'], 'a')
        self.assertEqual(search(text='q', mode='dense', url='http://localhost:9200',
                                index='idx', limit=2, query_vector=vector, client=c)[0]['id'], 'b')


if __name__ == '__main__':
    unittest.main()

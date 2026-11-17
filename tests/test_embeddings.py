import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from unittest.mock import patch

from scholarsearch.embeddings import (
    artifact_prefix, embed, paper_text, read_npy, source_digest,
)
from scholarsearch.index import curated_uri


class FakeEncoder:
    sep_token = '[SEP]'

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[len(t), len(t) + 3, 2] for t in texts], dtype='float32')


class EmbeddingTests(unittest.TestCase):
    def test_text_fallback(self):
        self.assertEqual(paper_text({'title': 'One', 'abstract': 'A'}, '[SEP]'),
                         ('One[SEP]A', 'title_abstract'))
        self.assertEqual(paper_text({'title': 'Two', 'abstract': None}),
                         ('Two', 'title_only'))

    def test_local_storage_and_repeat_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers = [
                {'id': 'W1', 'title': 'Graph malware', 'abstract': 'A method'},
                {'id': 'W2', 'title': 'Malware', 'abstract': None},
                {'id': 'W3', 'title': 'Networks', 'abstract': 'Network malware'},
            ]
            encoder = FakeEncoder()
            with patch("scholarsearch.embeddings.load_papers", return_value=papers):
                out = embed(root=tmp, query='malware detection', encoder=encoder, batch_size=2)
            self.assertEqual(out['count'], 3)
            self.assertEqual(out['title_only'], 1)
            self.assertEqual(out['dimension'], 3)
            self.assertEqual(encoder.calls, 2)
            arr = read_npy(out['vectors_uri'])
            np.testing.assert_allclose(np.linalg.norm(arr, axis=1), 1, atol=1e-6)
            self.assertEqual(json.loads(Path(out['paper_ids_uri']).read_text()), ['W1', 'W2', 'W3'])
            with patch("scholarsearch.embeddings.load_papers", return_value=papers):
                out2 = embed(root=tmp, query='malware detection', encoder=encoder, batch_size=2)
            self.assertEqual(out2, out)
            self.assertEqual(encoder.calls, 2)

    def test_resume_after_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers = [{'id': 'W1', 'title': 'A', 'abstract': 'One'},
                      {'id': 'W2', 'title': 'B', 'abstract': 'Two'}]
            encoder = FakeEncoder()
            prefix = Path(artifact_prefix(tmp, 'malware detection', source_digest(papers)))
            (prefix / 'batches').mkdir(parents=True)
            data = encoder.encode(['A[SEP]One'])
            with (prefix / 'batches' / '00000000.npy').open('wb') as file:
                np.save(file, data)
            (prefix / 'batches' / '00000000.json').write_text('["W1"]')
            with patch("scholarsearch.embeddings.load_papers", return_value=papers):
                out = embed(root=tmp, query='malware detection', encoder=encoder, batch_size=1)
            self.assertEqual(encoder.calls, 2)  # one manual, one newly generated
            self.assertEqual(out['count'], 2)


if __name__ == '__main__':
    unittest.main()

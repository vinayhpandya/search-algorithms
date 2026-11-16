import hashlib
import unittest

from scholarsearch.index import curated_uri, mapping, paper_document


class IndexTests(unittest.TestCase):
    def test_curated_uri_matches_curation(self):
        digest = hashlib.sha256(b"malware detection").hexdigest()[:12]
        self.assertEqual(curated_uri("s3://example/scholarsearch", "malware detection"),
                         f"s3://example/scholarsearch/curated/openalex-{digest}.parquet")

    def test_nullable_paper_fields(self):
        doc = paper_document({"id": "work-1", "title": "Malware"})
        self.assertEqual(doc["abstract"], "")
        self.assertEqual(doc["topics"], [])
        self.assertEqual(doc["citations"], 0)
        self.assertEqual(mapping()["settings"]["number_of_replicas"], 0)


if __name__ == "__main__":
    unittest.main()

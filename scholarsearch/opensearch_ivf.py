"""Create a separately named trained OpenSearch Faiss IVF index (do not overwrite baseline)."""
from __future__ import annotations
import argparse
import json
import os
import numpy as np


def build_mapping(dim, nlist):
    return {"settings": {"index": {"knn": True}}, "mappings": {"properties": {
        "paper_id": {"type": "keyword"},
        "title": {"type": "text"}, "abstract": {"type": "text"},
        "embedding": {"type": "knn_vector", "dimension": dim,
           "method": {"name": "ivf", "engine": "faiss", "space_type": "innerproduct",
                      "parameters": {"nlist": nlist, "encoder": {"name": "flat"}}}}
    }}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--nlist", type=int, default=32)
    p.add_argument("--out", default="artifacts/opensearch-ivf-mapping.json")
    a = p.parse_args()
    from pathlib import Path
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_mapping(a.dimension, a.nlist), indent=2))
    print(f"Wrote experimental mapping candidate: {target}")
    print("NOTE: This is a template, not a provisioned index. Validate training/model requirements against your OpenSearch version.")

if __name__ == "__main__":
    main()

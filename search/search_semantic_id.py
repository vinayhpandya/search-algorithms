"""
Generative retrieval via the trained query -> semantic ID model:
predict a semantic ID for the query, retrieve that cluster's papers,
rank them via scoped BM25. Matches the same interface as every other
search/*.py module so it plugs into the existing eval pipeline
(generate_candidates.py, benchmark_latency.py) without special-casing.

Usage:
    python search/semantic_id_search.py "malware detection using API calls"
"""
import argparse
import json
import tarfile
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from eval.build_ltr_features import bm25_scores_for_candidates

MODEL_TARBALL = Path(__file__).parent.parent / "eval" / "semantic_id_model.tar.gz"
MODEL_DIR = Path(__file__).parent.parent / "eval" / "semantic_id_model_extracted"
SEMANTIC_IDS_PATH = Path(__file__).parent.parent / "eval" / "semantic_ids.json"

_tokenizer = None
_model = None
_semantic_ids = None


def _load():
    global _tokenizer, _model, _semantic_ids
    if _model is None:
        if not MODEL_DIR.exists():
            with tarfile.open(MODEL_TARBALL) as tar:
                tar.extractall(MODEL_DIR)
        model_path = MODEL_DIR / "model"
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        _semantic_ids = json.loads(SEMANTIC_IDS_PATH.read_text())
    return _tokenizer, _model, _semantic_ids


def _predict_semantic_id(query: str, tokenizer, model) -> tuple[int, int] | None:
    inputs = tokenizer(f"predict semantic id: {query}", return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=8)
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    try:
        parts = text.split()
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def search_semantic_id(query: str, k: int = 10) -> list[dict]:
    tokenizer, model, semantic_ids = _load()

    predicted_id = _predict_semantic_id(query, tokenizer, model)
    if predicted_id is None:
        return []

    cluster_paper_ids = [
        pid for pid, data in semantic_ids.items()
        if tuple(data["semantic_id"]) == predicted_id
    ]
    if not cluster_paper_ids:
        return []

    bm25_scores = bm25_scores_for_candidates(query, cluster_paper_ids)
    ranked = sorted(cluster_paper_ids, key=lambda pid: bm25_scores.get(pid, 0.0), reverse=True)[:k]

    return [
        {
            "id": pid,
            "source": {
                "id": pid,
                "title": semantic_ids[pid]["title"],
                "abstract": "",  # not stored in semantic_ids.json; fetched below if needed
            },
        }
        for pid in ranked
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    hits = search_semantic_id(args.query, args.k)
    print(f"\n{len(hits)} results for: {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['source']['title']}  (id={hit['id']})")


if __name__ == "__main__":
    main()
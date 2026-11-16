"""Convert checkpointed OpenAlex pages to canonical, deduplicated JSONL/Parquet."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .storage import get_bytes, put_bytes, join


def inverted_abstract(index):
    if not index:
        return None
    length = max((p for positions in index.values() for p in positions), default=-1) + 1
    words = [""] * length
    for word, positions in index.items():
        for p in positions:
            words[p] = word
    return " ".join(words)


def canonical(record):
    ids = record.get("ids") or {}
    return {
        "id": record["id"], "title": record.get("display_name") or record.get("title") or "",
        "abstract": inverted_abstract(record.get("abstract_inverted_index")),
        "doi": record.get("doi") or ids.get("doi"), "arxiv_id": None,
        "publication_year": record.get("publication_year"),
        "citations": record.get("cited_by_count", 0),
        "topics": [x.get("display_name") for x in (record.get("topics") or []) if x.get("display_name")],
        "source": ["openalex"], "github_url": None,
    }


def curate(root, query, *, output=None, fmt="jsonl"):
    import hashlib
    key = hashlib.sha256(query.encode()).hexdigest()[:12]
    prefix = join(root, "raw", "openalex", key)
    checkpoint = json.loads(get_bytes(join(prefix, "checkpoint.json")))
    dedup = {}
    for page in range(int(checkpoint["page"])):
        data = json.loads(get_bytes(join(prefix, f"page-{page:06d}.json")))
        for record in data["results"]:
            if record.get("id"):
                dedup[record["id"]] = canonical(record)
    records = sorted(dedup.values(), key=lambda x: x["id"])
    output = output or join(root, "curated", f"openalex-{key}.{fmt}")
    if fmt == "jsonl":
        data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records).encode()
    elif fmt == "parquet":
        import io
        import polars as pl
        buf = io.BytesIO()
        pl.DataFrame(records).write_parquet(buf)
        data = buf.getvalue()
    else:
        raise ValueError("Choose jsonl or parquet")
    put_bytes(output, data)
    return {"papers": len(records), "output": output, "bytes": len(data)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="artifacts")
    p.add_argument("--query", default="malware detection")
    p.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    p.add_argument("--output")
    args = p.parse_args()
    print(json.dumps(curate(args.root, args.query, output=args.output, fmt=args.format), indent=2))

if __name__ == "__main__":
    main()

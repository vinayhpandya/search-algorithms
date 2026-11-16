"""Bounded, resumable OpenAlex ingestion. No request fanout or aggressive retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import random
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from .storage import exists, get_bytes, join, put_bytes

LOG = logging.getLogger(__name__)
BASE = "https://api.openalex.org/works"


def fetch_page(session, params, *, min_interval=1.0, attempts=6, sleep=time.sleep):
    """One request at a time; obey Retry-After and exponential backoff."""
    for attempt in range(attempts):
        try:
            response = session.get(BASE, params=params, timeout=35)
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(300, max(min_interval, float(retry_after))) if retry_after else min(120, 2 ** attempt + random.random())
                except ValueError:
                    delay = min(120, 2 ** attempt + random.random())
                LOG.warning("OpenAlex returned %s; retry in %.1fs", response.status_code, delay)
                if attempt == attempts - 1:
                    response.raise_for_status()
                sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("results"), list):
                raise ValueError("Invalid OpenAlex response: missing results")
            return payload
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == attempts - 1:
                raise
            delay = min(120, 2 ** attempt + random.random())
            LOG.warning("OpenAlex transport failure: %s; retry in %.1fs", exc, delay)
            sleep(delay)
    raise RuntimeError("Retry attempts exhausted")


def run(*, root, query, max_pages, page_size=100, api_key=None, email=None, min_interval=1.0, session=None, sleep=time.sleep):
    """Write each page *before* advancing checkpoint; duplicate fetches are safe."""
    if not (1 <= page_size <= 200):
        raise ValueError("page_size must be between 1 and 200")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    identifier = hashlib.sha256(query.encode()).hexdigest()[:12]
    prefix = join(root, "raw", "openalex", identifier)
    checkpoint_uri = join(prefix, "checkpoint.json")
    checkpoint = json.loads(get_bytes(checkpoint_uri)) if exists(checkpoint_uri) else {"cursor": "*", "page": 0, "done": False}
    if checkpoint.get("query", query) != query:
        raise ValueError("Checkpoint query mismatch")
    if checkpoint.get("done"):
        return checkpoint
    client = session or requests.Session()
    for _ in range(max_pages):
        page = int(checkpoint["page"])
        params = {"search": query, "per-page": page_size, "cursor": checkpoint["cursor"]}
        if api_key:
            params["api_key"] = api_key
        if email:
            params["mailto"] = email
        payload = fetch_page(client, params, min_interval=min_interval, sleep=sleep)
        records = payload["results"]
        uri = join(prefix, f"page-{page:06d}.json")
        # Persist full API page for provenance and future schema changes.
        put_bytes(uri, json.dumps(payload, ensure_ascii=False).encode())
        cursor = payload.get("meta", {}).get("next_cursor")
        checkpoint = {"query": query, "cursor": cursor, "page": page + 1,
                      "done": not bool(cursor) or not bool(records),
                      "updated_at": datetime.now(timezone.utc).isoformat(), "last_page": uri}
        put_bytes(checkpoint_uri, json.dumps(checkpoint, indent=2).encode())
        LOG.info("Saved %s (%d papers); checkpoint page=%s", uri, len(records), checkpoint["page"])
        if checkpoint["done"]:
            break
        sleep(min_interval)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts", help="Local directory or s3://bucket/prefix")
    parser.add_argument("--query", default="malware detection")
    parser.add_argument("--max-pages", type=int, default=2, help="Maximum pages this invocation; safe to resume")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--api-key", default=None, help="Prefer OPENALEX_API_KEY env variable")
    parser.add_argument("--email", default=None)
    args = parser.parse_args()
    import os
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(root=args.root, query=args.query, max_pages=args.max_pages,
                         page_size=args.page_size, min_interval=args.min_interval,
                         api_key=args.api_key or os.environ.get("OPENALEX_API_KEY"),
                         email=args.email or os.environ.get("OPENALEX_EMAIL")), indent=2))

if __name__ == "__main__":
    main()

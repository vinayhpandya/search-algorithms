#!/usr/bin/env bash
set -Eeuo pipefail
# Run from repo root; do not package .env, AWS profiles, HF tokens, or full repo.
if [[ ! -f scholarsearch/embeddings.py || ! -f scholarsearch/index.py ]]; then
  echo 'Run this script from the search-algorithms repository root.' >&2
  exit 1
fi
TMP_TAR="$(mktemp -t scholarsearch-source.XXXXXX.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT
# Only Python source files; no caches, local checkpoints, or secrets.
tar -czf "$TMP_TAR" --exclude='__pycache__' --exclude='*.pyc' scholarsearch
aws s3 cp "$TMP_TAR" \
  s3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b/bootstrap/source.tar.gz \
  --profile scholarsearch-terraform --region us-west-2
printf 'Uploaded ScholarSearch Python source (no .env or credentials).\n'

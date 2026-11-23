#!/usr/bin/env bash
set -Eeuo pipefail
# Run from repo root; do not package .env, AWS profiles, HF tokens, or full repo.
# Uploads, under the Experiment B prefix (the only prefix the workers may access):
#   bootstrap/source.tar.gz        Python source (used by embedding-worker and experiment-worker)
#   configs/*.json                 experiment configs (always refreshed)
#   inputs/evaluation/v1/*         frozen queries, query vectors, judgments (uploaded only if missing,
#                                  so judgments added later on AWS are never overwritten)
if [[ ! -f scholarsearch/embeddings.py || ! -f scholarsearch/index.py ]]; then
  echo 'Run this script from the search-algorithms repository root.' >&2
  exit 1
fi
PROFILE="${AWS_PROFILE:-scholarsearch-terraform}"
REGION="${AWS_REGION:-us-west-2}"
PREFIX='s3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b'
s3() { aws --profile "$PROFILE" --region "$REGION" s3 "$@"; }

TMP_TAR="$(mktemp -t scholarsearch-source.XXXXXX.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT
# Only Python source files; no caches, local checkpoints, or secrets.
tar -czf "$TMP_TAR" --exclude='__pycache__' --exclude='*.pyc' scholarsearch
s3 cp "$TMP_TAR" "$PREFIX/bootstrap/source.tar.gz"
printf 'Uploaded ScholarSearch Python source (no .env or credentials).\n'

for config in experiments/*.json; do
  s3 cp "$config" "$PREFIX/configs/$(basename "$config")"
done
printf 'Uploaded experiment configs.\n'

for input in queries_draft.tsv query_vectors.jsonl llm_judgments.jsonl; do
  dest="$PREFIX/inputs/evaluation/v1/$input"
  if s3 ls "$dest" >/dev/null 2>&1; then
    printf 'Keeping existing %s\n' "$dest"
  else
    s3 cp "artifacts/evaluation/v1/$input" "$dest"
  fi
done
printf 'Evaluation inputs ready.\n'

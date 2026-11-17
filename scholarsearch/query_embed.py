"""Encode a short search query using SPECTER2's adhoc-query adapter.

Use the isolated .venv-embed (adapters + Transformers 4.x). Writes a small JSON
file consumed by scholarsearch.search in the main project's environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import MODEL, SpecterEncoder, normalize

QUERY_ADAPTER = 'allenai/specter2_adhoc_query'


def encode_query(text: str, encoder=None) -> dict:
    if not text.strip():
        raise ValueError('Search query must not be empty')
    if encoder is None:
        encoder = SpecterEncoder(adapter=QUERY_ADAPTER, device='cpu')
    vector = normalize(encoder.encode([text.strip()]))
    if vector.shape != (1, 768):
        raise ValueError(f'Expected one 768-dimensional SPECTER2 query vector, got {vector.shape}')
    return {'query': text.strip(), 'model': MODEL, 'adapter': QUERY_ADAPTER,
            'dimension': 768, 'normalized': True, 'vector': vector[0].tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', required=True)
    p.add_argument('--output', required=True, help='Local JSON path, not S3')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = p.parse_args()
    encoder = SpecterEncoder(adapter=QUERY_ADAPTER, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(encode_query(args.text, encoder=encoder)), encoding='utf-8')
    print(f'Wrote {output} (768-dimensional normalized query vector)')


if __name__ == '__main__':
    main()

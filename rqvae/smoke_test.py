"""
Cheap local smoke test: runs a handful of epochs on a small slice of your
real OpenSearch index, on CPU. Not meant to produce a usable model -- just
to catch shape errors, OpenSearch connectivity issues, and checkpoint I/O
problems before spending A10 time on Modal.

Usage:
    uv run python -m rqvae.smoke_test
    uv run python -m rqvae.smoke_test --max-docs 500 --epochs 3
"""
import argparse

from rqvae.model import RQVAEConfig
from rqvae.train import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--opensearch-host", type=str, default="localhost")
    parser.add_argument("--opensearch-port", type=int, default=9200)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parser.parse_args()

    cfg = RQVAEConfig(
        codebook_size=args.codebook_size,
        num_levels=args.num_levels,
        latent_dim=args.latent_dim,
    )
    max_combinations = args.codebook_size ** args.num_levels
    print(
        f"Config: codebook_size={args.codebook_size}, num_levels={args.num_levels} "
        f"-> max {max_combinations} possible ID combinations"
    )

    print(
        f"Smoke test: {args.max_docs} docs, {args.epochs} epochs, "
        f"batch_size={args.batch_size}, forced CPU"
    )

    run_training(
        cfg=cfg,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device="cpu",                 # force CPU regardless of what's available locally
        opensearch_host=args.opensearch_host,
        opensearch_port=args.opensearch_port,
        max_docs=args.max_docs,
        patience=args.epochs,         # don't early-stop during a smoke test
    )


if __name__ == "__main__":
    main()
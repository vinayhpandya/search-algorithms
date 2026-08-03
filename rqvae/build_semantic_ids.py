"""
Run a trained RQ-VAE checkpoint over the full corpus, generate a semantic ID
(codeword tuple) for every paper, and backfill it into OpenSearch as a new
field -- deliberately separate from the existing k-means-based `semantic_id`
field so both can be compared side by side rather than one overwriting the
other.

Usage:
    uv run python -m rqvae.build_semantic_ids
    uv run python -m rqvae.build_semantic_ids --checkpoint checkpoints/rqvae/best.pt
"""
import argparse
from collections import defaultdict

import torch
from opensearchpy import OpenSearch, helpers

from rqvae.data import fetch_embeddings_from_opensearch, normalize_embeddings
from rqvae.model import RQVAE, RQVAEConfig

TARGET_FIELD = "semantic_id_rqvae"
INDEX_NAME = "papers"


def load_model(checkpoint_path: str) -> RQVAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = RQVAEConfig(**ckpt["config"])
    model = RQVAE(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']} "
        f"(val_recon_loss={ckpt['val_recon_loss']:.6f})"
    )
    print(f"Config: {ckpt['config']}")
    return model


def generate_ids(
    model: RQVAE, embeddings, batch_size: int = 256
) -> list[list[int]]:
    """Run encode_to_ids over the full corpus in batches, return one ID
    tuple (list of ints) per paper, in the same order as `embeddings`."""
    all_codes = []
    tensor = torch.from_numpy(embeddings).float()
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i : i + batch_size]
            codes = model.encode_to_ids(batch)  # (B, num_levels)
            all_codes.append(codes)
    codes = torch.cat(all_codes, dim=0)
    return codes.tolist()


def backfill_opensearch(
    client: OpenSearch, paper_ids: list[str], ids: list[list[int]], index_name: str
):
    actions = [
        {
            "_op_type": "update",
            "_index": index_name,
            "_id": paper_id,
            "doc": {TARGET_FIELD: id_tuple},
        }
        for paper_id, id_tuple in zip(paper_ids, ids)
    ]
    success, errors = helpers.bulk(client, actions, raise_on_error=False)
    print(f"Backfilled {success} documents with '{TARGET_FIELD}'")
    if errors:
        print(f"{len(errors)} documents failed to update (showing first 3):")
        for err in errors[:3]:
            print(err)


def report_diagnostics(
    client: OpenSearch,
    index_name: str,
    paper_ids: list[str],
    ids: list[list[int]],
    sample_clusters: int = 5,
):
    """Collision rate + a qualitative topic-coherence spot check, same
    spirit as the k-means build_semantic_ids.py sanity check."""
    id_to_papers: dict[tuple, list[str]] = defaultdict(list)
    for paper_id, id_tuple in zip(paper_ids, ids):
        id_to_papers[tuple(id_tuple)].append(paper_id)

    n_papers = len(paper_ids)
    n_unique_ids = len(id_to_papers)
    collisions = n_papers - n_unique_ids
    print(f"\n--- Diagnostics ---")
    print(f"Papers: {n_papers}, unique IDs: {n_unique_ids}")
    print(f"Papers sharing an ID with at least one other paper: {collisions}")

    cluster_sizes = sorted((len(v) for v in id_to_papers.values()), reverse=True)
    print(f"Largest clusters (paper count): {cluster_sizes[:10]}")
    print(f"Singleton IDs (1 paper only): {sum(1 for s in cluster_sizes if s == 1)}")

    # qualitative check: sample a few multi-paper clusters and print titles
    multi_paper_clusters = [
        (id_tuple, members)
        for id_tuple, members in id_to_papers.items()
        if len(members) > 1
    ]
    print(f"\n--- Sample clusters (topic coherence check) ---")
    for id_tuple, members in multi_paper_clusters[:sample_clusters]:
        titles = []
        for paper_id in members[:5]:
            doc = client.get(index=index_name, id=paper_id, _source=["title"])
            titles.append(doc["_source"].get("title", "<no title>"))
        print(f"\nID {id_tuple} ({len(members)} papers):")
        for t in titles:
            print(f"  - {t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rqvae/best.pt")
    parser.add_argument("--opensearch-host", type=str, default="localhost")
    parser.add_argument("--opensearch-port", type=int, default=9200)
    parser.add_argument("--index", type=str, default=INDEX_NAME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate IDs and print diagnostics without writing to OpenSearch",
    )
    args = parser.parse_args()

    client = OpenSearch(hosts=[{"host": args.opensearch_host, "port": args.opensearch_port}])

    model = load_model(args.checkpoint)

    loaded = fetch_embeddings_from_opensearch(client, index_name=args.index)
    print(f"Fetched {len(loaded.paper_ids)} embeddings")

    # must match training-time normalization exactly, or the encoder sees
    # out-of-distribution inputs and IDs will be meaningless
    normalized = normalize_embeddings(loaded.embeddings)

    ids = generate_ids(model, normalized)

    report_diagnostics(client, args.index, loaded.paper_ids, ids)

    if args.dry_run:
        print("\n--dry-run set: skipping OpenSearch backfill")
    else:
        backfill_opensearch(client, loaded.paper_ids, ids, args.index)


if __name__ == "__main__":
    main()
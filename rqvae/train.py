"""
Training loop for Stage 1 RQ-VAE.

Trains the encoder/decoder/quantizer end-to-end on SPECTER2 embeddings,
tracks reconstruction loss, commitment loss, and per-level codebook
utilization (the primary signal for codebook collapse), and checkpoints
the best model by validation reconstruction loss.

Usage (local):
    python -m rqvae.train

Usage (Modal): see modal_app.py -- this module exposes `run_training()`
so it can be called from either entrypoint without duplicating logic.
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch
from opensearchpy import OpenSearch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rqvae.data import load_and_prepare
from rqvae.model import RQVAE, RQVAEConfig

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "rqvae"
LOG_PATH = CHECKPOINT_DIR / "train_log.jsonl"


def evaluate(model: RQVAE, val_loader, device: str) -> dict:
    model.eval()
    total_recon, total_commit, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch, update_codebooks=False)
            total_recon += out["recon_loss"].item()
            total_commit += out["commitment_loss"].item()
            n_batches += 1
    model.train()
    return {
        "val_recon_loss": total_recon / max(n_batches, 1),
        "val_commitment_loss": total_commit / max(n_batches, 1),
    }


def log_line(record: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_training(
    cfg: RQVAEConfig = RQVAEConfig(),
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    opensearch_host: str = "localhost",
    opensearch_port: int = 9200,
    patience: int = 15,
    max_docs: int | None = None,
) -> Path:
    print(f"Training on device: {device}")
    client = OpenSearch(hosts=[{"host": opensearch_host, "port": opensearch_port}])
    loaded, train_loader, val_loader = load_and_prepare(
        client, batch_size=batch_size, max_docs=max_docs
    )

    model = RQVAE(cfg).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path = CHECKPOINT_DIR / "best.pt"

    for epoch in range(num_epochs):
        model.train()
        epoch_recon, epoch_commit, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch, update_codebooks=True)
            out["loss"].backward()
            optimizer.step()

            epoch_recon += out["recon_loss"].item()
            epoch_commit += out["commitment_loss"].item()
            n_batches += 1

        scheduler.step()

        train_recon = epoch_recon / max(n_batches, 1)
        train_commit = epoch_commit / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        utilization = model.quantizer.codebook_utilization().tolist()

        record = {
            "epoch": epoch,
            "train_recon_loss": train_recon,
            "train_commitment_loss": train_commit,
            "lr": scheduler.get_last_lr()[0],
            "codebook_utilization_per_level": utilization,
            **val_metrics,
        }
        log_line(record)

        # utilization is the first thing worth watching -- flag collapse loudly
        min_util = min(utilization)
        collapse_flag = " <-- LOW UTILIZATION, check for collapse" if min_util < 0.1 else ""
        print(
            f"epoch {epoch:3d} | train_recon {train_recon:.4f} | "
            f"val_recon {val_metrics['val_recon_loss']:.4f} | "
            f"utilization {['%.2f' % u for u in utilization]}{collapse_flag}"
        )

        if val_metrics["val_recon_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_recon_loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_recon_loss": best_val_loss,
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Best val reconstruction loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    run_training()
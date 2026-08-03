"""
Training loop for Stage 2: query -> semantic ID.

Structurally mirrors rqvae/train.py (epochs, val split, checkpointing) but
tracks classification-style metrics (per-digit accuracy, exact-match
accuracy) instead of reconstruction loss, since this is now a discrete
prediction task, not an autoencoder.

Usage:
    uv run python -m rqvae.train_query_to_id
    uv run python -m rqvae.train_query_to_id --epochs 30
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rqvae.query_data import build_dataloaders
from rqvae.query_to_id_model import QueryToIDConfig, QueryToIDModel

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "query_to_id"
LOG_PATH = CHECKPOINT_DIR / "train_log.jsonl"


def evaluate(model: QueryToIDModel, val_loader, device: str) -> dict:
    model.eval()
    total_loss, total_digit_acc, total_exact_acc, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for queries, ids in val_loader:
            ids = ids.to(device)
            out = model(queries, ids)
            total_loss += out["loss"].item()
            total_digit_acc += out["per_digit_acc"]
            total_exact_acc += out["exact_match_acc"]
            n_batches += 1
    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_per_digit_acc": total_digit_acc / max(n_batches, 1),
        "val_exact_match_acc": total_exact_acc / max(n_batches, 1),
    }


def log_line(record: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_training(
    cfg: QueryToIDConfig = QueryToIDConfig(),
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_path: Path | None = None,
    patience: int = 8,
) -> Path:
    print(f"Training on device: {device}")
    train_loader, val_loader = build_dataloaders(
        path=data_path or Path(__file__).parent.parent / "data" / "synthetic_queries.jsonl",
        batch_size=batch_size,
    )

    model = QueryToIDModel(cfg).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable_params):,} "
        f"(encoder frozen={cfg.freeze_encoder})"
    )
    optimizer = Adam(trainable_params, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path = CHECKPOINT_DIR / "best.pt"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_digit_acc, epoch_exact_acc, n_batches = 0.0, 0.0, 0.0, 0

        for queries, ids in train_loader:
            ids = ids.to(device)
            optimizer.zero_grad()
            out = model(queries, ids)
            out["loss"].backward()
            optimizer.step()

            epoch_loss += out["loss"].item()
            epoch_digit_acc += out["per_digit_acc"]
            epoch_exact_acc += out["exact_match_acc"]
            n_batches += 1

        scheduler.step()

        train_loss = epoch_loss / max(n_batches, 1)
        train_digit_acc = epoch_digit_acc / max(n_batches, 1)
        train_exact_acc = epoch_exact_acc / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_per_digit_acc": train_digit_acc,
            "train_exact_match_acc": train_exact_acc,
            "lr": scheduler.get_last_lr()[0],
            **val_metrics,
        }
        log_line(record)

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_metrics['val_loss']:.4f} | "
            f"train_exact {train_exact_acc:.2f} | val_exact {val_metrics['val_exact_match_acc']:.2f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_exact_match_acc": val_metrics["val_exact_match_acc"],
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze-encoder", dest="freeze_encoder", action="store_false")
    args = parser.parse_args()

    cfg = QueryToIDConfig(freeze_encoder=args.freeze_encoder)
    run_training(cfg=cfg, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
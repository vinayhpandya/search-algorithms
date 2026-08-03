"""
RQ-VAE (Residual-Quantized VAE) for learning semantic IDs from paper embeddings.

Architecture: Encoder -> Residual Vector Quantizer (N levels) -> Decoder
Trained end-to-end with reconstruction + commitment loss; codebooks updated
via EMA (exponential moving average) rather than gradient descent, which is
the standard, more stable choice for VQ/RQ-VAE style models.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RQVAEConfig:
    input_dim: int = 768          # SPECTER2 embedding dimension
    latent_dim: int = 64          # compressed latent dimension, pre-quantization
    hidden_dims: tuple = (512, 256)   # encoder hidden layers (decoder mirrors this)
    num_levels: int = 4           # number of residual quantization levels
    codebook_size: int = 256      # codewords per level -> ID digit range [0, 255]
    commitment_weight: float = 0.25   # beta: weight on commitment loss
    ema_decay: float = 0.99       # EMA update rate for codebooks
    dead_code_threshold: int = 0  # a code with 0 assignments in an epoch is "dead"


class MLP(nn.Module):
    """Simple MLP used for both encoder and decoder, LayerNorm + GELU between layers."""

    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # no norm/activation after the final layer -- it's a raw projection
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-level residual quantizer. Each level has its own codebook.

    At level i: find nearest codeword to the current residual, that codeword's
    index becomes ID digit i, then subtract it out to get the residual for
    level i+1. Codebooks are updated via EMA of assigned encoder outputs
    (not backprop) -- this is the standard VQ-VAE-family approach and is
    considerably more stable than gradient-based codebook learning.

    A straight-through estimator (STE) is used so gradients flow from the
    decoder back to the encoder despite the non-differentiable argmin lookup.
    """

    def __init__(self, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.num_levels = cfg.num_levels
        self.codebook_size = cfg.codebook_size
        self.latent_dim = cfg.latent_dim
        self.decay = cfg.ema_decay

        # one codebook per level, each (codebook_size, latent_dim)
        codebooks = torch.randn(cfg.num_levels, cfg.codebook_size, cfg.latent_dim) * 0.02
        self.register_buffer("codebooks", codebooks)

        # EMA bookkeeping buffers: cluster sizes and running sums per codeword
        self.register_buffer(
            "ema_cluster_size", torch.zeros(cfg.num_levels, cfg.codebook_size)
        )
        self.register_buffer("ema_embed_sum", codebooks.clone())

        # tracks how many consecutive steps each codeword has gone unused,
        # for dead-code detection/reset
        self.register_buffer(
            "steps_unused", torch.zeros(cfg.num_levels, cfg.codebook_size)
        )

        # separate from ema_cluster_size on purpose: this only ever increases
        # from genuine batch assignments and is explicitly zeroed (not set to
        # 1.0) on dead-code reset. ema_cluster_size gets forced to 1.0 on
        # reset so the codebook position math stays stable, but that means
        # it can't be trusted as a "was this codeword actually used by real
        # data" signal -- this buffer is.
        self.register_buffer(
            "genuine_usage", torch.zeros(cfg.num_levels, cfg.codebook_size)
        )

    def _quantize_level(
        self, residual: torch.Tensor, level: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize `residual` (B, latent_dim) against codebook[level]. Returns
        (quantized_vectors, code_indices)."""
        codebook = self.codebooks[level]  # (codebook_size, latent_dim)

        # squared L2 distance from every residual to every codeword:
        # ||r - c||^2 = ||r||^2 - 2 r.c + ||c||^2
        dist = (
            residual.pow(2).sum(1, keepdim=True)
            - 2 * residual @ codebook.t()
            + codebook.pow(2).sum(1)
        )
        code_idx = dist.argmin(dim=1)  # (B,)
        quantized = codebook[code_idx]  # (B, latent_dim)
        return quantized, code_idx

    def _ema_update(self, level: int, residual: torch.Tensor, code_idx: torch.Tensor):
        """Update codebook[level] via EMA of the encoder outputs assigned to
        each codeword this batch. Also resets any codeword that's been
        unused for too many consecutive steps ('dead code' reset)."""
        one_hot = F.one_hot(code_idx, self.codebook_size).float()  # (B, codebook_size)

        batch_cluster_size = one_hot.sum(0)  # (codebook_size,) usage counts this batch
        batch_embed_sum = one_hot.t() @ residual  # (codebook_size, latent_dim)

        # genuine_usage only ever grows from real assignments -- never
        # touched by the reset logic below, so it stays a trustworthy
        # "was this codeword actually chosen by real data" signal
        self.genuine_usage[level] += batch_cluster_size

        self.ema_cluster_size[level].mul_(self.decay).add_(
            batch_cluster_size, alpha=1 - self.decay
        )
        self.ema_embed_sum[level].mul_(self.decay).add_(
            batch_embed_sum, alpha=1 - self.decay
        )

        # normalize to get the new codeword positions (avoid div-by-zero for unused codes)
        cluster_size = self.ema_cluster_size[level].clamp(min=1e-5)
        new_codebook = self.ema_embed_sum[level] / cluster_size.unsqueeze(1)

        # dead code tracking: codewords unused this batch get their counter bumped
        unused_mask = batch_cluster_size == 0
        self.steps_unused[level][unused_mask] += 1
        self.steps_unused[level][~unused_mask] = 0

        # reset codewords dead for too long: reseed from a random residual in this batch
        dead_mask = self.steps_unused[level] > 5  # dead for 5+ consecutive batches
        if dead_mask.any() and residual.size(0) > 0:
            n_dead = int(dead_mask.sum().item())
            random_residuals = residual[
                torch.randint(0, residual.size(0), (n_dead,), device=residual.device)
            ]
            new_codebook[dead_mask] = random_residuals
            self.steps_unused[level][dead_mask] = 0
            # reset EMA stats so the new position isn't immediately pulled
            # back toward its old (unused) state
            self.ema_cluster_size[level][dead_mask] = 1.0
            self.ema_embed_sum[level][dead_mask] = random_residuals
            # NOT touching genuine_usage here on purpose -- a reset codeword
            # hasn't earned real usage yet, it's just been repositioned to a
            # better starting point. It only counts toward utilization once
            # real data actually picks it afterward.
            self.genuine_usage[level][dead_mask] = 0.0

        self.codebooks[level] = new_codebook

    def forward(
        self, z: torch.Tensor, update_codebooks: bool = True
    ) -> dict:
        """
        z: (B, latent_dim) encoder output.
        Returns dict with quantized output, per-level code indices, and losses.
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_codes = []
        commitment_loss = 0.0

        for level in range(self.num_levels):
            quantized, code_idx = self._quantize_level(residual, level)

            # straight-through estimator: forward uses quantized value,
            # backward passes gradient straight through as if this were identity
            quantized_st = residual + (quantized - residual).detach()

            commitment_loss = commitment_loss + F.mse_loss(residual, quantized.detach())

            if update_codebooks and self.training:
                # EMA update uses the *un-quantized* residual assigned to each code
                self._ema_update(level, residual.detach(), code_idx)

            quantized_sum = quantized_sum + quantized_st
            residual = residual - quantized_st.detach()  # residual for next level
            all_codes.append(code_idx)

        codes = torch.stack(all_codes, dim=1)  # (B, num_levels)
        return {
            "quantized": quantized_sum,
            "codes": codes,
            "commitment_loss": commitment_loss / self.num_levels,
        }

    def codebook_utilization(self) -> torch.Tensor:
        """Fraction of codewords with at least one genuine (non-reset) real
        assignment, per level. Low values indicate codebook collapse.
        Deliberately based on `genuine_usage`, not `ema_cluster_size` --
        the latter gets forced to 1.0 on dead-code reset to keep the
        codebook position math stable, which would otherwise make this
        metric report high utilization even when a codeword has never
        actually been chosen by real data."""
        used = (self.genuine_usage > 0).float().mean(dim=1)
        return used  # (num_levels,)


class RQVAE(nn.Module):
    def __init__(self, cfg: RQVAEConfig):
        super().__init__()
        self.cfg = cfg
        enc_dims = [cfg.input_dim, *cfg.hidden_dims, cfg.latent_dim]
        dec_dims = [cfg.latent_dim, *reversed(cfg.hidden_dims), cfg.input_dim]

        self.encoder = MLP(enc_dims)
        self.decoder = MLP(dec_dims)
        self.quantizer = ResidualVectorQuantizer(cfg)

    def forward(self, x: torch.Tensor, update_codebooks: bool = True) -> dict:
        z = self.encoder(x)
        q_out = self.quantizer(z, update_codebooks=update_codebooks)
        recon = self.decoder(q_out["quantized"])

        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + self.cfg.commitment_weight * q_out["commitment_loss"]

        return {
            "recon": recon,
            "codes": q_out["codes"],          # (B, num_levels) -- the semantic ID
            "recon_loss": recon_loss,
            "commitment_loss": q_out["commitment_loss"],
            "loss": loss,
        }

    @torch.no_grad()
    def encode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: embedding -> semantic ID tuple, no codebook updates."""
        self.eval()
        z = self.encoder(x)
        q_out = self.quantizer(z, update_codebooks=False)
        return q_out["codes"]
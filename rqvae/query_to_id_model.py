"""
Query -> Semantic ID generative model.

Encoder: a small PRETRAINED sentence transformer (all-MiniLM-L6-v2) turns
the query into per-token hidden states. Pretrained, not trained from
scratch -- with only ~1940 query examples, we want language understanding
already baked in, and only fine-tune (or freeze) it for the ID-generation
task specifically.

Decoder: a small transformer decoder, trained from scratch, autoregressively
generates the ID digit sequence (e.g. [coarse, fine]) conditioned on the
query via cross-attention. Digit 2 is predicted conditioned on digit 1 --
this is what makes it genuinely sequence-to-sequence rather than two
independent classifiers.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class QueryToIDConfig:
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_dim: int = 384          # MiniLM hidden size
    decoder_dim: int = 128          # kept small -- little training data
    num_decoder_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    num_levels: int = 2             # must match Stage 1 RQVAEConfig.num_levels
    codebook_size: int = 16         # must match Stage 1 RQVAEConfig.codebook_size
    freeze_encoder: bool = True     # see rationale in QueryEncoder docstring
    max_query_len: int = 32


class QueryEncoder(nn.Module):
    """Wraps a pretrained sentence transformer and projects its per-token
    hidden states down to decoder_dim, so the decoder can cross-attend over
    them.

    freeze_encoder=True by default: with ~1940 examples, fine-tuning all of
    MiniLM risks overfitting/catastrophic forgetting of its pretrained
    language understanding. Freezing it and only training the projection +
    decoder is the safer default; unfreezing is easy to try later if the
    frozen version underfits.
    """

    def __init__(self, cfg: QueryToIDConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
        self.backbone = AutoModel.from_pretrained(cfg.encoder_name)

        if cfg.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)

    def forward(self, queries: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory, key_padding_mask):
        memory: (B, seq_len, decoder_dim) -- per-token encoder states, projected
        key_padding_mask: (B, seq_len) bool, True where padded (for attention masking)
        """
        tokenized = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_query_len,
            return_tensors="pt",
        ).to(device)

        if self.cfg.freeze_encoder:
            with torch.no_grad():
                out = self.backbone(**tokenized)
        else:
            out = self.backbone(**tokenized)

        hidden = out.last_hidden_state  # (B, seq_len, encoder_dim)
        memory = self.projection(hidden)  # (B, seq_len, decoder_dim)
        key_padding_mask = tokenized["attention_mask"] == 0  # True = padding
        return memory, key_padding_mask


class IDDecoder(nn.Module):
    """Autoregressively generates num_levels ID digits, each in
    [0, codebook_size). Uses a shared token embedding (BOS + all digit
    values) and a separate output head per level, since coarse/fine digit
    spaces are semantically distinct even though they're the same size.
    """

    def __init__(self, cfg: QueryToIDConfig):
        super().__init__()
        self.cfg = cfg
        self.bos_id = cfg.codebook_size  # reserve one extra token id for BOS

        self.token_embed = nn.Embedding(cfg.codebook_size + 1, cfg.decoder_dim)
        self.pos_embed = nn.Embedding(cfg.num_levels, cfg.decoder_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.decoder_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers)

        # separate output head per level -- coarse and fine are distinct
        # digit spaces despite sharing a vocab size
        self.output_heads = nn.ModuleList(
            [nn.Linear(cfg.decoder_dim, cfg.codebook_size) for _ in range(cfg.num_levels)]
        )

    def _causal_mask(self, size: int, device: str) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced training pass.
        target_ids: (B, num_levels) ground-truth digits.
        Returns logits: (B, num_levels, codebook_size).
        """
        B, device = target_ids.size(0), target_ids.device

        bos = torch.full((B, 1), self.bos_id, device=device, dtype=torch.long)
        decoder_input = torch.cat([bos, target_ids[:, :-1]], dim=1)  # shift right

        positions = torch.arange(self.cfg.num_levels, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(decoder_input) + self.pos_embed(positions)

        causal_mask = self._causal_mask(self.cfg.num_levels, device)
        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, num_levels, decoder_dim)

        logits = torch.stack(
            [self.output_heads[i](decoded[:, i, :]) for i in range(self.cfg.num_levels)], dim=1
        )  # (B, num_levels, codebook_size)
        return logits

    @torch.no_grad()
    def generate_beam(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        beam_size: int = 5,
    ) -> list[tuple[list[int], float]]:
        """Beam search decode, one query at a time (memory batch size 1 per call
        from the caller). Returns ALL final beams as (digit_sequence, score)
        tuples, sorted best-first -- not just the top one. Retrieval needs
        multiple candidate ID buckets, not just the single best guess, since
        a wrong first digit sends lookup to the wrong branch entirely and
        having fallback candidates matters for building a ranked result list."""
        device = memory.device
        beams = [([], 0.0)]  # (sequence_so_far, cumulative_log_prob)

        for level in range(self.cfg.num_levels):
            candidates = []
            for seq, score in beams:
                bos = torch.tensor([[self.bos_id]], device=device)
                prefix = torch.tensor([seq], device=device, dtype=torch.long) if seq else None
                decoder_input = bos if prefix is None else torch.cat([bos, prefix], dim=1)

                positions = torch.arange(decoder_input.size(1), device=device).unsqueeze(0)
                x = self.token_embed(decoder_input) + self.pos_embed(positions)
                causal_mask = self._causal_mask(decoder_input.size(1), device)

                decoded = self.decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                logits = self.output_heads[level](decoded[:, -1, :])  # (1, codebook_size)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (codebook_size,)

                topk = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
                for value, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((seq + [idx], score + value))

            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = candidates[:beam_size]

        beams.sort(key=lambda c: c[1], reverse=True)
        return beams


class QueryToIDModel(nn.Module):
    def __init__(self, cfg: QueryToIDConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = QueryEncoder(cfg)
        self.decoder = IDDecoder(cfg)

    def forward(self, queries: list[str], target_ids: torch.Tensor) -> dict:
        device = target_ids.device
        memory, key_padding_mask = self.encoder(queries, device)
        logits = self.decoder(memory, key_padding_mask, target_ids)  # (B, num_levels, codebook_size)

        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.codebook_size), target_ids.reshape(-1)
        )

        preds = logits.argmax(dim=-1)  # (B, num_levels)
        per_digit_acc = (preds == target_ids).float().mean().item()
        exact_match_acc = (preds == target_ids).all(dim=1).float().mean().item()

        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
            "per_digit_acc": per_digit_acc,
            "exact_match_acc": exact_match_acc,
        }

    @torch.no_grad()
    def predict(self, query: str, device: str, beam_size: int = 5) -> list[int]:
        """Inference on a single query string -> top predicted ID digit sequence."""
        self.eval()
        memory, key_padding_mask = self.encoder([query], device)
        beams = self.decoder.generate_beam(memory, key_padding_mask, beam_size=beam_size)
        return beams[0][0]  # top beam's sequence

    @torch.no_grad()
    def predict_topk(
        self, query: str, device: str, beam_size: int = 10
    ) -> list[tuple[list[int], float]]:
        """Inference on a single query string -> ALL beam candidates
        (id_sequence, score), best-first. Used by retrieval to pull papers
        from multiple candidate ID buckets, not just the single top guess."""
        self.eval()
        memory, key_padding_mask = self.encoder([query], device)
        return self.decoder.generate_beam(memory, key_padding_mask, beam_size=beam_size)
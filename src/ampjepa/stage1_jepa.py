"""Stage 1 AMP-JEPA utilities.

This module implements a compact, true JEPA-style peptide encoder:

- context encoder sees masked peptide sequences;
- target encoder sees the full peptide sequence;
- predictor maps context hidden states to target hidden states;
- target encoder is updated by exponential moving average, not backprop;
- optional MLM loss can be added as a stabilizer.

The implementation is intentionally small and auditable so Stage 1 can run before
larger ESM2-conditioned variants are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"

VOCAB: Dict[str, int] = {PAD_TOKEN: 0, MASK_TOKEN: 1, UNK_TOKEN: 2}
VOCAB.update({aa: i + 3 for i, aa in enumerate(CANONICAL_AA)})
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
PAD_ID = VOCAB[PAD_TOKEN]
MASK_ID = VOCAB[MASK_TOKEN]
UNK_ID = VOCAB[UNK_TOKEN]
VOCAB_SIZE = len(VOCAB)


@dataclass
class Stage1Config:
    max_len: int = 100
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    ff_dim: int = 512
    dropout: float = 0.1
    ema_decay: float = 0.996
    mlm_weight: float = 0.1


def clean_sequence(seq: str) -> str:
    """Return uppercase canonical amino-acid sequence with whitespace removed."""
    seq = "".join(str(seq).upper().split())
    return seq


def is_canonical_sequence(seq: str) -> bool:
    seq = clean_sequence(seq)
    return len(seq) > 0 and all(aa in CANONICAL_AA for aa in seq)


def tokenize_sequence(seq: str, max_len: int, mask_start: int | None = None, mask_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one peptide and optionally replace a contiguous span with MASK.

    Returns token ids and a boolean mask marking real, non-padding positions.
    """
    seq = clean_sequence(seq)[:max_len]
    ids: List[int] = []
    for i, aa in enumerate(seq):
        if mask_start is not None and mask_start <= i < mask_start + mask_len:
            ids.append(MASK_ID)
        else:
            ids.append(VOCAB.get(aa, UNK_ID))
    real_len = len(ids)
    if real_len < max_len:
        ids.extend([PAD_ID] * (max_len - real_len))
    token_ids = torch.tensor(ids, dtype=torch.long)
    real_mask = torch.zeros(max_len, dtype=torch.bool)
    real_mask[:real_len] = True
    return token_ids, real_mask


def masked_region_mask(max_len: int, seq_len: int, mask_start: int, mask_len: int) -> torch.Tensor:
    out = torch.zeros(max_len, dtype=torch.bool)
    start = max(0, int(mask_start))
    end = min(int(seq_len), start + max(1, int(mask_len)), max_len)
    if start < end:
        out[start:end] = True
    return out


class PeptideEncoder(nn.Module):
    """Small Transformer encoder for short peptide sequences."""

    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        padding_mask = tokens.eq(PAD_ID)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


class LatentPredictor(nn.Module):
    """Predict target latent vectors from context latent vectors."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class MaskedTokenHead(nn.Module):
    """Optional MLM stabilizer head."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


def copy_weights(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    for param in target.parameters():
        param.requires_grad_(False)


@torch.no_grad()
def ema_update(source: nn.Module, target: nn.Module, decay: float) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in target_state.items():
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(source_state[name].detach(), alpha=1.0 - decay)
        else:
            value.copy_(source_state[name])


def masked_latent_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and target latent vectors only at masked sites."""
    if mask.sum() == 0:
        return predicted.sum() * 0.0
    return torch.mean((predicted[mask] - target[mask]) ** 2)


def mean_pool(hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    real = tokens.ne(PAD_ID).float().unsqueeze(-1)
    pooled = (hidden * real).sum(dim=1) / real.sum(dim=1).clamp_min(1.0)
    return pooled


def save_checkpoint(
    path: str,
    config: Stage1Config,
    context_encoder: PeptideEncoder,
    target_encoder: PeptideEncoder,
    predictor: LatentPredictor,
    token_head: MaskedTokenHead | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": config.__dict__,
        "vocab": VOCAB,
        "context_encoder": context_encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "token_head": token_head.state_dict() if token_head is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_encoder_from_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Tuple[Stage1Config, PeptideEncoder, dict]:
    payload = torch.load(path, map_location=map_location)
    config = Stage1Config(**payload["config"])
    encoder = PeptideEncoder(config)
    encoder.load_state_dict(payload["context_encoder"])
    encoder.eval()
    return config, encoder, payload

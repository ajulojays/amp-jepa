"""AMP-JEPA-Hybrid v3 core model and scoring utilities.

This is the improved version of the original candidate-generation track:

- sequence VAE for AMP latent generation;
- JEPA-inspired masked-view latent consistency;
- physicochemical feature prediction as a weak biological regularizer;
- controlled decoding and candidate filtering utilities.

The model is intentionally compact so it can train on ~20k curated AMP sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS, MASK, UNK = "<pad>", "<bos>", "<eos>", "<mask>", "<unk>"
VOCAB: Dict[str, int] = {PAD: 0, BOS: 1, EOS: 2, MASK: 3, UNK: 4}
VOCAB.update({aa: i + 5 for i, aa in enumerate(CANONICAL_AA)})
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
PAD_ID, BOS_ID, EOS_ID, MASK_ID, UNK_ID = VOCAB[PAD], VOCAB[BOS], VOCAB[EOS], VOCAB[MASK], VOCAB[UNK]
AA_IDS = [VOCAB[aa] for aa in CANONICAL_AA]
VOCAB_SIZE = len(VOCAB)
HYDROPHOBIC = set("AILMFWVY")
POSITIVE = set("KR")
NEGATIVE = set("DE")


@dataclass
class V3Config:
    max_len: int = 64
    d_model: int = 192
    latent_dim: int = 64
    n_layers: int = 4
    n_heads: int = 6
    ff_dim: int = 512
    dropout: float = 0.1
    beta_kl: float = 0.02
    jepa_weight: float = 0.25
    property_weight: float = 0.10


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def is_valid_sequence(seq: str, min_len: int = 8, max_len: int = 64) -> bool:
    seq = clean_sequence(seq)
    return min_len <= len(seq) <= max_len and all(aa in CANONICAL_AA for aa in seq)


def peptide_features(seq: str, max_len: int = 64) -> np.ndarray:
    seq = clean_sequence(seq)
    n = max(len(seq), 1)
    charge = sum(aa in POSITIVE for aa in seq) - sum(aa in NEGATIVE for aa in seq)
    hydro = sum(aa in HYDROPHOBIC for aa in seq) / n
    gly_pro = (seq.count("G") + seq.count("P")) / n
    aromatic = sum(aa in set("FWY") for aa in seq) / n
    return np.array([len(seq) / max_len, charge / 12.0, hydro, gly_pro, aromatic], dtype=np.float32)


def tokenize(seq: str, max_len: int, mask_rate: float = 0.0) -> torch.Tensor:
    seq = clean_sequence(seq)[:max_len]
    ids: List[int] = []
    for aa in seq:
        ids.append(VOCAB.get(aa, UNK_ID))
    if mask_rate > 0:
        for i in range(len(ids)):
            if np.random.random() < mask_rate:
                ids[i] = MASK_ID
    if len(ids) < max_len:
        ids.extend([PAD_ID] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids: Iterable[int]) -> str:
    chars = []
    for idx in ids:
        token = ID_TO_TOKEN.get(int(idx), "")
        if token in {PAD, BOS, EOS, MASK, UNK}:
            continue
        if token in CANONICAL_AA:
            chars.append(token)
    return "".join(chars)


def sequence_identity(a: str, b: str) -> float:
    a, b = clean_sequence(a), clean_sequence(b)
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    matches = sum(x == y for x, y in zip(a[:m], b[:m]))
    return matches / max(len(a), len(b))


def max_identity_to_training(seq: str, training_sequences: List[str], sample_limit: int = 5000) -> float:
    if not training_sequences:
        return 0.0
    if len(training_sequences) > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(training_sequences), size=sample_limit, replace=False)
        subset = [training_sequences[i] for i in idx]
    else:
        subset = training_sequences
    return max(sequence_identity(seq, ref) for ref in subset)


def candidate_score(seq: str, max_train_identity: float = 0.0) -> Dict[str, float]:
    seq = clean_sequence(seq)
    length = len(seq)
    charge = sum(aa in POSITIVE for aa in seq) - sum(aa in NEGATIVE for aa in seq)
    hydro = sum(aa in HYDROPHOBIC for aa in seq) / max(length, 1)
    valid_len = 10 <= length <= 40
    charge_ok = 3 <= charge <= 10
    hydro_ok = 0.30 <= hydro <= 0.65
    novelty = 1.0 - max_train_identity
    developability = float(valid_len) + float(charge_ok) + float(hydro_ok)
    score = 2.0 * novelty + developability + 0.15 * charge - 1.5 * max(0.0, hydro - 0.65)
    return {
        "length": float(length),
        "net_charge_KR_minus_DE": float(charge),
        "hydrophobic_fraction": float(hydro),
        "max_train_identity": float(max_train_identity),
        "novelty_score": float(novelty),
        "developability_score": float(developability),
        "v3_rank_score": float(score),
        "passes_v3_filters": float(valid_len and charge_ok and hydro_ok and novelty >= 0.20),
    }


class HybridVAEJEPA(nn.Module):
    def __init__(self, config: V3Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.to_mu = nn.Linear(config.d_model, config.latent_dim)
        self.to_logvar = nn.Linear(config.d_model, config.latent_dim)
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.decoder = nn.Linear(config.d_model, config.max_len * VOCAB_SIZE)
        self.length_head = nn.Linear(config.d_model, 1)
        self.view_predictor = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim * 2),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )
        self.property_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length = tokens.shape
        pos = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
        x = self.token_embedding(tokens) + self.position_embedding(pos)
        pad_mask = tokens.eq(PAD_ID)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        real = tokens.ne(PAD_ID).float().unsqueeze(-1)
        pooled = (h * real).sum(dim=1) / real.sum(dim=1).clamp_min(1.0)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-8.0, 8.0)
        return mu, logvar, pooled

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_logits(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.latent_to_hidden(z)
        logits = self.decoder(h).view(-1, self.config.max_len, VOCAB_SIZE)
        length_norm = torch.sigmoid(self.length_head(h)).squeeze(-1)
        return logits, length_norm

    def forward(self, tokens: torch.Tensor, masked_tokens: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, _ = self.encode(tokens)
        z = self.reparameterize(mu, logvar)
        logits, length_norm = self.decode_logits(z)
        masked_mu, _, _ = self.encode(masked_tokens)
        predicted_full_mu = self.view_predictor(masked_mu)
        prop_pred = self.property_head(mu)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "logits": logits,
            "length_norm": length_norm,
            "predicted_full_mu": predicted_full_mu,
            "target_full_mu": mu.detach(),
            "property_pred": prop_pred,
            "property_target": features,
        }


def save_v3_checkpoint(path: str, model: HybridVAEJEPA, config: V3Config, extra: dict | None = None) -> None:
    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "vocab": VOCAB, "extra": extra or {}}, path)


def load_v3_checkpoint(path: str, map_location="cpu") -> Tuple[HybridVAEJEPA, V3Config, dict]:
    payload = torch.load(path, map_location=map_location)
    config = V3Config(**payload["config"])
    model = HybridVAEJEPA(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload


def sample_sequences(model: HybridVAEJEPA, n: int, temperature: float = 1.0, device: str | torch.device = "cpu") -> List[str]:
    model.eval()
    device = torch.device(device)
    model.to(device)
    sequences: List[str] = []
    with torch.no_grad():
        for _ in range(n):
            z = torch.randn(1, model.config.latent_dim, device=device)
            logits, length_norm = model.decode_logits(z)
            length = int(round(float(length_norm.item()) * model.config.max_len))
            length = max(8, min(model.config.max_len, length))
            probs = torch.softmax(logits[0, :length, :] / max(temperature, 1e-6), dim=-1)
            probs[:, :5] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ids = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().tolist()
            seq = decode_ids(ids)
            if seq:
                sequences.append(seq)
    return sequences

#!/usr/bin/env python3
"""Train AMP-JEPA-Hybrid v3 on a curated AMP corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ampjepa_hybrid_v3 import (
    PAD_ID,
    V3Config,
    HybridVAEJEPA,
    peptide_features,
    save_v3_checkpoint,
    tokenize,
)


class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: V3Config, mask_rate: float):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mask_rate = mask_rate

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        seq = str(self.df.iloc[idx]["sequence"])
        return {
            "tokens": tokenize(seq, self.config.max_len, mask_rate=0.0),
            "masked_tokens": tokenize(seq, self.config.max_len, mask_rate=self.mask_rate),
            "features": torch.tensor(peptide_features(seq, self.config.max_len), dtype=torch.float32),
        }


def collate(batch):
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "masked_tokens": torch.stack([b["masked_tokens"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
    }


def loss_fn(model: HybridVAEJEPA, batch, device):
    tokens = batch["tokens"].to(device)
    masked = batch["masked_tokens"].to(device)
    features = batch["features"].to(device)
    out = model(tokens, masked, features)

    ce = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    recon = ce(out["logits"].reshape(-1, out["logits"].size(-1)), tokens.reshape(-1))
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    jepa = torch.mean((out["predicted_full_mu"] - out["target_full_mu"]) ** 2)
    prop = torch.mean((out["property_pred"] - out["property_target"]) ** 2)
    loss = recon + model.config.beta_kl * kl + model.config.jepa_weight * jepa + model.config.property_weight * prop
    return loss, {"recon": recon.item(), "kl": kl.item(), "jepa": jepa.item(), "prop": prop.item(), "loss": loss.item()}


def run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "jepa": 0.0, "prop": 0.0}
    n = 0
    for batch in loader:
        with torch.set_grad_enabled(train):
            loss, metrics = loss_fn(model, batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        b = batch["tokens"].size(0)
        for k in totals:
            totals[k] += metrics[k] * b
        n += b
    return {k: v / max(n, 1) for k, v in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="v3/data/processed/peptide_corpus_v3.csv")
    p.add_argument("--checkpoint", default="v3/checkpoints/amp_jepa_hybrid_v3.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--mask-rate", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    corpus = pd.read_csv(args.corpus)
    if "sequence" not in corpus.columns:
        raise SystemExit("[ERROR] corpus must contain a sequence column")

    config = V3Config(max_len=args.max_len, latent_dim=args.latent_dim, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    dataset = PeptideDataset(corpus, config, args.mask_rate)
    val_size = max(1, int(0.1 * len(dataset))) if len(dataset) > 20 else 0
    train_size = len(dataset) - val_size
    if val_size:
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if val_ds is not None else None

    model = HybridVAEJEPA(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, opt, device)
        if val_loader is not None:
            va = run_epoch(model, val_loader, None, device)
            print(f"[EPOCH {epoch:03d}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} recon={tr['recon']:.4f} jepa={tr['jepa']:.4f}")
        else:
            va = {}
            print(f"[EPOCH {epoch:03d}] train_loss={tr['loss']:.4f} recon={tr['recon']:.4f} jepa={tr['jepa']:.4f}")
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_v3_checkpoint(str(ckpt), model, config, extra={"history": history, "n_sequences": len(corpus)})
    pd.DataFrame(history).to_csv(ckpt.with_suffix(".training_history.csv"), index=False)
    print(f"[DONE] Wrote {ckpt}")


if __name__ == "__main__":
    main()

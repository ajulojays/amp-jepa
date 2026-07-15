#!/usr/bin/env python3
"""Stage 1C: train the AMP-JEPA encoder.

This is a compact true JEPA training loop:

context sequence with masked span -> context encoder -> predictor
full sequence                        -> EMA target encoder -> target latent

The loss is computed only on the masked target positions. An optional masked-token
loss is included as a stabilizer, following the lesson that JEPA-only protein
training can collapse in small settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ampjepa.stage1_jepa import (  # noqa: E402
    MASK_ID,
    PAD_ID,
    Stage1Config,
    PeptideEncoder,
    LatentPredictor,
    MaskedTokenHead,
    copy_weights,
    ema_update,
    masked_latent_loss,
    masked_region_mask,
    save_checkpoint,
    tokenize_sequence,
)


class JEPAPairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, max_len: int):
        self.pairs = pairs.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.pairs.iloc[idx]
        seq = str(row["sequence"])
        start = int(row["mask_start"])
        length = int(row["mask_len"])
        context_tokens, _ = tokenize_sequence(seq, self.max_len, start, length)
        target_tokens, real_mask = tokenize_sequence(seq, self.max_len)
        target_mask = masked_region_mask(self.max_len, len(seq), start, length) & real_mask
        return {
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "target_mask": target_mask,
            "peptide_id": str(row["peptide_id"]),
            "pair_id": str(row["pair_id"]),
        }


def collate(batch):
    return {
        "context_tokens": torch.stack([x["context_tokens"] for x in batch]),
        "target_tokens": torch.stack([x["target_tokens"] for x in batch]),
        "target_mask": torch.stack([x["target_mask"] for x in batch]),
        "peptide_id": [x["peptide_id"] for x in batch],
        "pair_id": [x["pair_id"] for x in batch],
    }


def run_epoch(
    loader: DataLoader,
    context_encoder: PeptideEncoder,
    target_encoder: PeptideEncoder,
    predictor: LatentPredictor,
    token_head: MaskedTokenHead,
    optimizer: torch.optim.Optimizer | None,
    config: Stage1Config,
    device: torch.device,
) -> Dict[str, float]:
    train = optimizer is not None
    context_encoder.train(train)
    predictor.train(train)
    token_head.train(train)
    target_encoder.eval()

    total = 0.0
    latent_total = 0.0
    mlm_total = 0.0
    n = 0
    ce = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for batch in loader:
        context_tokens = batch["context_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        target_mask = batch["target_mask"].to(device)

        with torch.set_grad_enabled(train):
            context_hidden = context_encoder(context_tokens)
            pred_target = predictor(context_hidden)
            with torch.no_grad():
                target_hidden = target_encoder(target_tokens)
            latent_loss = masked_latent_loss(pred_target, target_hidden.detach(), target_mask)

            logits = token_head(context_hidden)
            mlm_labels = target_tokens.masked_fill(~target_mask, PAD_ID)
            mlm_loss = ce(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
            loss = latent_loss + config.mlm_weight * mlm_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(context_encoder.parameters()) + list(predictor.parameters()) + list(token_head.parameters()), 1.0)
                optimizer.step()
                ema_update(context_encoder, target_encoder, config.ema_decay)

        batch_n = context_tokens.size(0)
        total += float(loss.detach().cpu()) * batch_n
        latent_total += float(latent_loss.detach().cpu()) * batch_n
        mlm_total += float(mlm_loss.detach().cpu()) * batch_n
        n += batch_n

    return {
        "loss": total / max(n, 1),
        "latent_loss": latent_total / max(n, 1),
        "mlm_loss": mlm_total / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/processed/stage1/jepa_pairs.csv")
    parser.add_argument("--checkpoint", default="checkpoints/stage1/amp_jepa_stage1.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--mlm-weight", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"[ERROR] Missing JEPA pairs: {pairs_path}")
    pairs = pd.read_csv(pairs_path)
    required = {"pair_id", "peptide_id", "sequence", "mask_start", "mask_len"}
    missing = required - set(pairs.columns)
    if missing:
        raise SystemExit(f"[ERROR] Pair table missing columns: {sorted(missing)}")

    config = Stage1Config(
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        ema_decay=args.ema_decay,
        mlm_weight=args.mlm_weight,
    )

    dataset = JEPAPairDataset(pairs, config.max_len)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if val_ds is not None else None

    context_encoder = PeptideEncoder(config).to(device)
    target_encoder = PeptideEncoder(config).to(device)
    copy_weights(context_encoder, target_encoder)
    predictor = LatentPredictor(config.d_model).to(device)
    token_head = MaskedTokenHead(config.d_model).to(device)

    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()) + list(token_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, context_encoder, target_encoder, predictor, token_head, optimizer, config, device)
        if val_loader is not None:
            val_metrics = run_epoch(val_loader, context_encoder, target_encoder, predictor, token_head, None, config, device)
            print(
                f"[EPOCH {epoch:03d}] train_loss={train_metrics['loss']:.4f} "
                f"train_latent={train_metrics['latent_loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_latent={val_metrics['latent_loss']:.4f}"
            )
        else:
            val_metrics = {}
            print(f"[EPOCH {epoch:03d}] train_loss={train_metrics['loss']:.4f} train_latent={train_metrics['latent_loss']:.4f}")
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        str(checkpoint_path),
        config,
        context_encoder,
        target_encoder,
        predictor,
        token_head,
        extra={"history": history, "n_pairs": len(dataset)},
    )
    print(f"[DONE] Wrote {checkpoint_path}")


if __name__ == "__main__":
    main()

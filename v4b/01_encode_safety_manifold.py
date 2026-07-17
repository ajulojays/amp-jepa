#!/usr/bin/env python3
"""Encode safety-reference peptides with the frozen V3 encoder and save latent μ/logvar."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v3.ampjepa_hybrid_v3 import load_v3_checkpoint, tokenize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="v4b/results/safety_manifold_pilot/safety_reference_clean.csv")
    ap.add_argument("--checkpoint", default="v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt")
    ap.add_argument("--outdir", default="v4b/results/safety_manifold_pilot")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    input_path = Path(args.input)
    checkpoint = Path(args.checkpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    df = pd.read_csv(input_path, low_memory=False)
    if "sequence" not in df.columns:
        raise SystemExit("Input must contain sequence column")

    device = torch.device(args.device)
    model, config, _ = load_v3_checkpoint(str(checkpoint), map_location=device)
    model.to(device).eval()

    mu_parts: list[np.ndarray] = []
    logvar_parts: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            seqs = df["sequence"].iloc[start : start + args.batch_size].astype(str).tolist()
            tokens = torch.stack([tokenize(s, config.max_len) for s in seqs]).to(device)
            mu, logvar, _ = model.encode(tokens)
            mu_parts.append(mu.cpu().numpy().astype(np.float32))
            logvar_parts.append(logvar.cpu().numpy().astype(np.float32))
            print(f"Encoded {min(start + args.batch_size, len(df))}/{len(df)}")

    mu = np.concatenate(mu_parts, axis=0) if mu_parts else np.empty((0, config.latent_dim), dtype=np.float32)
    logvar = np.concatenate(logvar_parts, axis=0) if logvar_parts else np.empty((0, config.latent_dim), dtype=np.float32)

    np.save(outdir / "latent_mu.npy", mu)
    np.save(outdir / "latent_logvar.npy", logvar)
    df.to_csv(outdir / "latent_metadata.csv", index=False)

    print(f"Saved latent μ: {outdir / 'latent_mu.npy'} shape={mu.shape}")
    print(f"Saved logvar: {outdir / 'latent_logvar.npy'} shape={logvar.shape}")
    print(f"Saved metadata: {outdir / 'latent_metadata.csv'}")


if __name__ == "__main__":
    main()

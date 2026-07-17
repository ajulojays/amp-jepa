#!/usr/bin/env python3
"""Encode V4B Generation 0 candidates with the frozen AMP-JEPA V3 encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="v4b/results/generation_00/generation_00_candidates.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt",
    )
    parser.add_argument(
        "--outdir",
        default="v4b/results/generation_00",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "v3" / "src"
    if not package_root.exists():
        raise FileNotFoundError(f"AMP-JEPA package directory not found: {package_root}")
    sys.path.insert(0, str(package_root))

    from ampjepa.model import load_v3_checkpoint, tokenize  # noqa: E402

    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    outdir = Path(args.outdir)
    latent_path = outdir / "latent_vectors.npz"
    metadata_path = outdir / "latent_metadata.csv"
    summary_path = outdir / "latent_encoding_summary.json"

    for required in (input_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    existing = [path for path in (latent_path, metadata_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Latent outputs already exist ({names}). Use --overwrite deliberately.")

    frame = pd.read_csv(input_path, low_memory=False)
    required_columns = {"candidate_id", "sequence", "generation"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Generation 0 table is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Generation 0 candidate table is empty.")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique before latent encoding.")

    device = resolve_device(args.device)
    model, config, payload = load_v3_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    sequences = frame["sequence"].astype(str).tolist()
    candidate_ids = frame["candidate_id"].astype(str).to_numpy(dtype=str)
    all_mu: list[np.ndarray] = []
    all_logvar: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch_sequences = sequences[start : start + args.batch_size]
            tokens = torch.stack(
                [tokenize(sequence, max_len=config.max_len) for sequence in batch_sequences]
            ).to(device, non_blocking=device.type == "cuda")
            mu, logvar, _ = model.encode(tokens)
            all_mu.append(mu.detach().cpu().numpy().astype(np.float32, copy=False))
            all_logvar.append(logvar.detach().cpu().numpy().astype(np.float32, copy=False))

    mu_array = np.concatenate(all_mu, axis=0)
    logvar_array = np.concatenate(all_logvar, axis=0)
    if mu_array.shape[0] != len(frame):
        raise RuntimeError("Latent vector count does not match candidate count.")

    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        latent_path,
        candidate_id=candidate_ids,
        mu=mu_array,
        logvar=logvar_array,
        generation=np.zeros(len(frame), dtype=np.int16),
    )

    metadata = frame.copy()
    metadata["latent_row"] = np.arange(len(frame), dtype=np.int64)
    metadata["latent_mu_l2_norm"] = np.linalg.norm(mu_array, axis=1)
    metadata["latent_posterior_mean_variance"] = np.exp(logvar_array).mean(axis=1)
    metadata.to_csv(metadata_path, index=False)

    checkpoint_extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
    summary = {
        "schema_version": "1.0",
        "v4b_stage": "generation_00_latent_encoding",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation": 0,
        "candidate_count": int(len(frame)),
        "latent_dimension": int(mu_array.shape[1]),
        "max_sequence_length": int(config.max_len),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "extra": checkpoint_extra,
        },
        "outputs": {
            "latent_vectors": str(latent_path),
            "latent_vectors_sha256": sha256_file(latent_path),
            "latent_metadata": str(metadata_path),
            "latent_metadata_sha256": sha256_file(metadata_path),
        },
        "latent_statistics": {
            "mu_global_mean": float(mu_array.mean()),
            "mu_global_std": float(mu_array.std()),
            "mean_mu_l2_norm": float(np.linalg.norm(mu_array, axis=1).mean()),
            "mean_posterior_variance": float(np.exp(logvar_array).mean()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

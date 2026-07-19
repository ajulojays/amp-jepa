#!/usr/bin/env python3
"""Initialize the isolated V4C Generation 0 baseline from frozen V4B inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "generation_00_candidates.csv",
    "latent_metadata.csv",
    "latent_vectors.npz",
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def install_file(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        relative = os.path.relpath(source.resolve(), start=target.parent.resolve())
        target.symlink_to(relative)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="v4b/results/generation_00",
        help="Frozen V4B Generation 0 directory.",
    )
    parser.add_argument(
        "--target-root",
        default="v4c/results/generation_00",
        help="V4C Generation 0 destination directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to install the frozen baseline files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing V4C Generation 0 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    missing = [name for name in REQUIRED_FILES if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Frozen V4B Generation 0 is incomplete. Missing: " + ", ".join(missing)
        )

    target_root.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_FILES:
        target = target_root / name
        if target.exists() or target.is_symlink():
            if not args.force:
                raise FileExistsError(
                    f"Target already exists: {target}. Use --force only when replacement is intended."
                )
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        install_file(source_root / name, target, args.mode)

    candidates_path = target_root / "generation_00_candidates.csv"
    metadata_path = target_root / "latent_metadata.csv"
    latent_path = target_root / "latent_vectors.npz"

    candidates = pd.read_csv(candidates_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    latent = np.load(latent_path)

    for frame_name, frame in (("candidates", candidates), ("metadata", metadata)):
        if "candidate_id" not in frame.columns:
            raise ValueError(f"{frame_name} table lacks candidate_id")
        if frame["candidate_id"].astype(str).duplicated().any():
            raise ValueError(f"{frame_name} table contains duplicate candidate_id values")

    if "candidate_id" not in latent or "mu" not in latent:
        raise ValueError("latent_vectors.npz must contain candidate_id and mu arrays")

    latent_ids = latent["candidate_id"].astype(str)
    candidate_ids = candidates["candidate_id"].astype(str).to_numpy()
    metadata_ids = metadata["candidate_id"].astype(str).to_numpy()

    if len(candidates) != len(metadata) or len(candidates) != len(latent_ids):
        raise ValueError(
            "Generation 0 row-count mismatch: "
            f"candidates={len(candidates)}, metadata={len(metadata)}, latents={len(latent_ids)}"
        )
    if not np.array_equal(metadata_ids, latent_ids):
        raise ValueError("latent metadata and latent archive candidate IDs are not aligned")
    if set(candidate_ids) != set(latent_ids):
        raise ValueError("candidate table and latent archive contain different candidate IDs")
    if np.isnan(latent["mu"]).any() or np.isinf(latent["mu"]).any():
        raise ValueError("latent archive contains NaN or infinite values")

    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "installation_mode": args.mode,
        "n_generation_00_candidates": int(len(candidates)),
        "latent_shape": list(latent["mu"].shape),
        "source_files": {
            name: {
                "path": str(source_root / name),
                "size_bytes": int((source_root / name).stat().st_size),
                "sha256": sha256(source_root / name),
            }
            for name in REQUIRED_FILES
        },
        "validation": {
            "candidate_ids_unique": True,
            "row_counts_aligned": True,
            "metadata_latent_order_aligned": True,
            "candidate_latent_id_sets_equal": True,
            "latent_values_finite": True,
        },
    }

    manifest_path = target_root / "v4c_generation_00_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nV4C GENERATION 0 INITIALIZED")
    print(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()

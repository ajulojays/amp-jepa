#!/usr/bin/env python3
"""Run a tiny v3 training job for debugging only."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable, "v3/01_train_v3_hybrid.py",
        "--corpus", "v3/data/processed/peptide_corpus_v3.csv",
        "--checkpoint", "v3/checkpoints/amp_jepa_hybrid_v3_tiny_debug.pt",
        "--epochs", "1",
        "--batch-size", "4",
        "--max-len", "40",
        "--d-model", "64",
        "--latent-dim", "16",
        "--n-layers", "1",
        "--n-heads", "4",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

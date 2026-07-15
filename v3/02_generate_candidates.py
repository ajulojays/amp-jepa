#!/usr/bin/env python3
"""Generate candidate peptides from AMP-JEPA-Hybrid v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ampjepa_hybrid_v3 import clean_sequence, is_valid_sequence, load_v3_checkpoint, sample_sequences


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="v3/checkpoints/amp_jepa_hybrid_v3.pt")
    p.add_argument("--output", default="v3/results/raw_candidates_v3.csv")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--max-len", type=int, default=40)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_v3_checkpoint(args.checkpoint, map_location=device)
    seqs = sample_sequences(model, args.n, temperature=args.temperature, device=device)

    clean = []
    seen = set()
    for seq in seqs:
        seq = clean_sequence(seq)
        if seq in seen:
            continue
        if not is_valid_sequence(seq, args.min_len, args.max_len):
            continue
        seen.add(seq)
        clean.append(seq)

    out = pd.DataFrame({"candidate_id": [f"v3_candidate_{i:06d}" for i in range(1, len(clean) + 1)], "sequence": clean})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[INFO] Requested samples: {args.n:,}")
    print(f"[INFO] Valid unique candidates: {len(out):,}")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()

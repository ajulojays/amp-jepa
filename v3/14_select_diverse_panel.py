#!/usr/bin/env python3
"""Select a simple diverse candidate panel from a ranked v3 table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import sequence_identity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="v3/results/ranked_candidates_v3.csv")
    p.add_argument("--output", default="v3/results/diverse_panel_v3.csv")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--max-pairwise-identity", type=float, default=0.80)
    args = p.parse_args()

    df = pd.read_csv(args.input).sort_values("v3_rank_score", ascending=False)
    if "passes_v3_filters" in df.columns:
        df = df.loc[df["passes_v3_filters"].astype(float) >= 1]
    selected = []
    selected_seqs = []
    for _, row in df.iterrows():
        seq = str(row["sequence"])
        if all(sequence_identity(seq, s) <= args.max_pairwise_identity for s in selected_seqs):
            selected.append(row)
            selected_seqs.append(seq)
        if len(selected) >= args.top:
            break
    out = pd.DataFrame(selected)
    if not out.empty:
        out.insert(0, "diverse_panel_rank", range(1, len(out) + 1))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[DONE] Wrote {path} rows={len(out):,}")


if __name__ == "__main__":
    main()

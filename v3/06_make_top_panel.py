#!/usr/bin/env python3
"""Create a compact top candidate panel from v3-ranked outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranked", default="v3/results/ranked_candidates_v3.csv")
    p.add_argument("--output", default="v3/results/top_panel_v3.csv")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--min-novelty", type=float, default=0.20)
    args = p.parse_args()

    df = pd.read_csv(args.ranked)
    keep = df.copy()
    if "passes_v3_filters" in keep.columns:
        keep = keep.loc[keep["passes_v3_filters"].astype(float) >= 1]
    if "novelty_score" in keep.columns:
        keep = keep.loc[keep["novelty_score"].astype(float) >= args.min_novelty]
    keep = keep.sort_values("v3_rank_score", ascending=False).head(args.top)

    cols = [c for c in [
        "v3_rank", "candidate_id", "sequence", "length", "net_charge_KR_minus_DE",
        "hydrophobic_fraction", "max_train_identity", "novelty_score", "developability_score", "v3_rank_score"
    ] if c in keep.columns]
    keep = keep[cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out, index=False)
    print(keep.to_string(index=False))
    print(f"[DONE] Wrote {out}")


if __name__ == "__main__":
    main()

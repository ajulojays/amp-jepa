#!/usr/bin/env python3
"""Deduplicate a ranked v3 candidate table by exact sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="v3/results/ranked_candidates_v3.csv")
    p.add_argument("--output", default="v3/results/ranked_candidates_v3_dedup.csv")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    sort_col = "v3_rank_score" if "v3_rank_score" in df.columns else df.columns[0]
    out = df.sort_values(sort_col, ascending=False).drop_duplicates("sequence")
    out.insert(0, "dedup_rank", range(1, len(out) + 1))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[DONE] Wrote {path} rows={len(out):,}")


if __name__ == "__main__":
    main()

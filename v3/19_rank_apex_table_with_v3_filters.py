#!/usr/bin/env python3
"""Rank the bundled APEX table with v3 heuristic filters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import candidate_score, clean_sequence


def main() -> None:
    src = Path("v3/data/external/apex_oracle_ranked_summary.csv")
    out = Path("v3/results/apex_table_v3_filter_rank.csv")
    df = pd.read_csv(src)
    rows = []
    for _, row in df.iterrows():
        seq = clean_sequence(row["PeptideSequence"])
        rows.append({**row.to_dict(), "sequence": seq, **candidate_score(seq)})
    ranked = pd.DataFrame(rows).sort_values(["passes_v3_filters", "v3_rank_score"], ascending=[False, False])
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    print(ranked.head(20).to_string(index=False))
    print(f"[DONE] Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Small smoke test for v3 utilities using the bundled APEX table only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import candidate_score, clean_sequence


def main() -> None:
    path = Path("v3/data/external/apex_oracle_ranked_summary.csv")
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing {path}")
    df = pd.read_csv(path)
    seq_col = "PeptideSequence" if "PeptideSequence" in df.columns else "sequence"
    rows = []
    for _, row in df.iterrows():
        seq = clean_sequence(row[seq_col])
        rows.append({"candidate_id": row.get("Unnamed: 0", "apex_candidate"), "sequence": seq, **candidate_score(seq)})
    out = pd.DataFrame(rows).sort_values("v3_rank_score", ascending=False)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

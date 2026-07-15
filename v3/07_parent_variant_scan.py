#!/usr/bin/env python3
"""Generate simple parent-peptide variants for v3 follow-up optimization.

This is a safe computational design utility: it proposes small in silico variants
for ranking by v3 filters and external predictors. It does not imply biological
validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import CANONICAL_AA, candidate_score, clean_sequence

CONSERVATIVE = {
    "K": "R", "R": "K", "L": "I", "I": "L", "V": "I", "F": "W", "W": "F",
    "D": "E", "E": "D", "S": "T", "T": "S", "N": "Q", "Q": "N",
}


def variants(seq: str, max_variants: int = 500):
    seq = clean_sequence(seq)
    out = []
    # Conservative substitutions first.
    for i, aa in enumerate(seq):
        if aa in CONSERVATIVE:
            out.append((f"{aa}{i+1}{CONSERVATIVE[aa]}", seq[:i] + CONSERVATIVE[aa] + seq[i+1:]))
    # Charge-enhancing substitutions at neutral polar positions.
    for i, aa in enumerate(seq):
        if aa in "STNQG":
            out.append((f"{aa}{i+1}K", seq[:i] + "K" + seq[i+1:]))
    # Mild hydrophobic tuning.
    for i, aa in enumerate(seq):
        if aa in "AILV" and seq.count("K") + seq.count("R") >= 4:
            out.append((f"{aa}{i+1}A", seq[:i] + "A" + seq[i+1:]))
    seen = set()
    clean_out = []
    for mut, var in out:
        if var != seq and var not in seen:
            clean_out.append((mut, var))
            seen.add(var)
        if len(clean_out) >= max_variants:
            break
    return clean_out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True, help="Parent peptide sequence")
    p.add_argument("--output", default="v3/results/parent_variant_scan_v3.csv")
    p.add_argument("--max-variants", type=int, default=500)
    args = p.parse_args()

    parent = clean_sequence(args.parent)
    rows = []
    rows.append({"variant_id": "parent", "mutation": "parent", "sequence": parent, **candidate_score(parent)})
    for i, (mut, seq) in enumerate(variants(parent, args.max_variants), start=1):
        rows.append({"variant_id": f"variant_{i:05d}", "mutation": mut, "sequence": seq, **candidate_score(seq)})
    out = pd.DataFrame(rows).sort_values("v3_rank_score", ascending=False)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.head(50).to_string(index=False))
    print(f"[DONE] Wrote {path}")


if __name__ == "__main__":
    main()

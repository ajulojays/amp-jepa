#!/usr/bin/env python3
"""Prepare the AMP-JEPA-Hybrid V4A seed pool from existing v3 outputs.

This script scans v3/results for CSV files containing peptide sequences, cleans and
merges them, deduplicates by sequence, computes simple peptide features, and writes
an APEX-ready candidate table with a `sequence` column.

No APEX inference is performed here. The output is scored by
v3/25_score_v3_candidates_with_apex.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDRO = set("AILMFWYV")
AROM = set("FWY")


def clean_sequence(x: object) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch in AA)


def longest_homopolymer(seq: str) -> int:
    best = cur = 0
    prev = None
    for ch in seq:
        cur = cur + 1 if ch == prev else 1
        best = max(best, cur)
        prev = ch
    return best


def entropy(seq: str) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    e = 0.0
    for aa in set(seq):
        p = seq.count(aa) / n
        e -= p * math.log2(p)
    return e


def features(seq: str) -> dict:
    n = len(seq)
    return {
        "length": n,
        "net_charge_KR_minus_DE": seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"),
        "hydrophobic_fraction": sum(ch in HYDRO for ch in seq) / max(n, 1),
        "cysteine_count": seq.count("C"),
        "tryptophan_count": seq.count("W"),
        "aromatic_fraction": sum(ch in AROM for ch in seq) / max(n, 1),
        "maximum_residue_fraction": max((seq.count(ch) / max(n, 1) for ch in set(seq)), default=0.0),
        "longest_homopolymer": longest_homopolymer(seq),
        "entropy": entropy(seq),
    }


def find_sequence_column(df: pd.DataFrame) -> str | None:
    for c in ["sequence", "Sequence", "PeptideSequence", "peptide", "Peptide", "peptide_sequence"]:
        if c in df.columns:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-results", default="v3/results")
    ap.add_argument("--output", default="v4/results/seed_pool/v4a_seed_candidates.csv")
    ap.add_argument("--summary", default="v4/results/seed_pool/v4a_seed_summary.json")
    ap.add_argument("--max-seeds", type=int, default=20000, help="0 means keep all deduplicated seeds")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.v3_results)
    rows = []
    scanned = skipped = 0

    for path in sorted(root.glob("**/*.csv")):
        scanned += 1
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            skipped += 1
            continue
        seq_col = find_sequence_column(df)
        if seq_col is None:
            skipped += 1
            continue
        keep_cols = [c for c in [
            seq_col,
            "candidate_id", "apex_candidate_id", "v3_rank", "v3_rank_score",
            "APEX_mean_MIC", "APEX_median_MIC", "APEX_worst_MIC", "organisms_MIC_le_64",
            "max_train_identity", "novelty_score", "developability_score",
            "source_file", "run", "global_rank"
        ] if c in df.columns]
        sub = df[keep_cols].copy()
        sub = sub.rename(columns={seq_col: "sequence"})
        sub["sequence"] = sub["sequence"].map(clean_sequence)
        sub = sub[sub["sequence"].str.len().between(args.min_len, args.max_len)]
        sub["source_file"] = str(path)
        rows.append(sub)

    if not rows:
        raise RuntimeError(f"No sequence-bearing CSV files found under {root}")

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out[out["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")].copy()

    # Prefer already-scored / globally ranked rows when duplicate sequences appear.
    out["_has_apex"] = out.get("APEX_median_MIC", pd.Series(index=out.index, dtype=float)).notna()
    out["_rank_key"] = pd.to_numeric(out.get("global_rank", pd.Series(index=out.index)), errors="coerce").fillna(1e9)
    out = out.sort_values(["_has_apex", "_rank_key"], ascending=[False, True])
    out = out.drop_duplicates("sequence", keep="first").reset_index(drop=True)
    out = out.drop(columns=[c for c in ["_has_apex", "_rank_key"] if c in out.columns])

    feat = pd.DataFrame([features(s) for s in out["sequence"]])
    for c in feat.columns:
        if c not in out.columns:
            out[c] = feat[c]

    if "candidate_id" not in out.columns:
        out.insert(0, "candidate_id", [f"V4A_seed_{i+1:06d}" for i in range(len(out))])
    else:
        out["candidate_id"] = out["candidate_id"].fillna(pd.Series([f"V4A_seed_{i+1:06d}" for i in range(len(out))]))

    out.insert(0, "v4a_seed_rank", range(1, len(out) + 1))
    out["v4a_source"] = "v3_results_scan"

    if args.max_seeds and args.max_seeds > 0:
        # Keep all already strong rows first, then broad coverage of the rest.
        scored = out[out.get("APEX_median_MIC", pd.Series(index=out.index, dtype=float)).notna()].copy()
        unscored = out[~out.index.isin(scored.index)].copy()
        out = pd.concat([scored, unscored], ignore_index=True).head(args.max_seeds).copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary = {
        "v3_results_root": str(root),
        "csv_files_scanned": scanned,
        "csv_files_skipped_or_without_sequences": skipped,
        "unique_seed_sequences": int(len(out)),
        "output": str(out_path),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "max_seeds": args.max_seeds,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("V4A seed pool prepared")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

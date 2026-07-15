#!/usr/bin/env python3
"""Map the full V4A candidate landscape after APEX scoring.

Input is usually:
  v4/results/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv

Outputs:
  v4/results/landscape/candidate_landscape.csv
  v4/results/landscape/candidate_class_assignments.csv
  v4/results/rescue/g_rescue_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
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
    out = 0.0
    for aa in set(seq):
        p = seq.count(aa) / n
        out -= p * math.log2(p)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sequence"] = df["sequence"].map(clean_sequence)
    seqs = df["sequence"].fillna("")
    df["length"] = seqs.map(len)
    df["net_charge_KR_minus_DE"] = seqs.map(lambda s: s.count("K") + s.count("R") - s.count("D") - s.count("E"))
    df["hydrophobic_fraction"] = seqs.map(lambda s: sum(ch in HYDRO for ch in s) / max(len(s), 1))
    df["cysteine_count"] = seqs.map(lambda s: s.count("C"))
    df["tryptophan_count"] = seqs.map(lambda s: s.count("W"))
    df["aromatic_fraction"] = seqs.map(lambda s: sum(ch in AROM for ch in s) / max(len(s), 1))
    df["maximum_residue_fraction"] = seqs.map(lambda s: max((s.count(ch) / max(len(s), 1) for ch in set(s)), default=0.0))
    df["longest_homopolymer"] = seqs.map(longest_homopolymer)
    df["entropy"] = seqs.map(entropy)
    return df


def clip01(x):
    return np.clip(x, 0, 1)


def normalize_inverse(series: pd.Series, good: float, bad: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(clip01((bad - x) / (bad - good)), index=series.index).fillna(0)


def normalize_direct(series: pd.Series, low: float, high: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(clip01((x - low) / (high - low)), index=series.index).fillna(0)


def property_ok(row) -> bool:
    return (
        8 <= row.length <= 35
        and 2 <= row.net_charge_KR_minus_DE <= 10
        and 0.25 <= row.hydrophobic_fraction <= 0.62
        and row.cysteine_count <= 2
        and row.tryptophan_count <= 3
        and row.aromatic_fraction <= 0.30
        and row.maximum_residue_fraction <= 0.42
        and row.longest_homopolymer <= 4
    )


def failure_modes(row) -> list[str]:
    modes = []
    if row.length < 8: modes.append("too_short")
    if row.length > 35: modes.append("too_long")
    if row.net_charge_KR_minus_DE < 2: modes.append("low_charge")
    if row.net_charge_KR_minus_DE > 10: modes.append("excess_charge")
    if row.hydrophobic_fraction < 0.25: modes.append("low_hydrophobicity")
    if row.hydrophobic_fraction > 0.62: modes.append("excess_hydrophobicity")
    if row.cysteine_count > 2: modes.append("excess_cysteine")
    if row.tryptophan_count > 3 or row.aromatic_fraction > 0.30: modes.append("excess_aromatic")
    if row.maximum_residue_fraction > 0.42 or row.longest_homopolymer > 4: modes.append("low_complexity")
    if pd.notna(row.APEX_median_MIC) and row.APEX_median_MIC > 80: modes.append("weak_median_MIC")
    if pd.notna(row.APEX_worst_MIC) and row.APEX_worst_MIC > 500: modes.append("weak_worst_case")
    if pd.notna(row.organisms_MIC_le_64) and row.organisms_MIC_le_64 < 12: modes.append("weak_breadth")
    return modes or ["balanced_or_unknown_failure"]


def assign_class(row) -> str:
    junk = (
        row.length < 5 or row.length > 64
        or row.net_charge_KR_minus_DE < -2
        or row.hydrophobic_fraction < 0.12
        or row.hydrophobic_fraction > 0.82
        or row.maximum_residue_fraction > 0.55
        or row.longest_homopolymer >= 6
    )
    if junk:
        return "G5_true_junk"

    prop = property_ok(row)
    median = row.APEX_median_MIC
    worst = row.APEX_worst_MIC
    org64 = row.organisms_MIC_le_64
    novelty = row.novelty_score if pd.notna(row.novelty_score) else 0.0

    if prop and median <= 40 and worst <= 350 and org64 >= 24:
        return "A_elite"
    if prop and median <= 80 and org64 >= 15:
        return "B_near_pass"
    if novelty >= 0.60 and median <= 128:
        return "C_high_novelty"
    if org64 >= 24:
        return "D_broad_spectrum"
    if worst <= 250:
        return "E_worst_case_robust"
    if prop and median <= 128:
        return "F_developable"
    return "G_rescue"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="v4/results/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv")
    ap.add_argument("--outdir", default="v4/results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    landscape_dir = outdir / "landscape"
    rescue_dir = outdir / "rescue"
    landscape_dir.mkdir(parents=True, exist_ok=True)
    rescue_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    if "sequence" not in df.columns:
        raise ValueError("Input needs a sequence column")
    df = add_features(df)

    for c in ["APEX_mean_MIC", "APEX_median_MIC", "APEX_worst_MIC", "organisms_MIC_le_64", "novelty_score"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "novelty_score" not in df.columns or df["novelty_score"].isna().all():
        if "max_train_identity" in df.columns:
            df["novelty_score"] = 1 - pd.to_numeric(df["max_train_identity"], errors="coerce")
        else:
            df["novelty_score"] = 0.50

    df["potency_component"] = normalize_inverse(df["APEX_median_MIC"], 20, 160)
    df["mean_component"] = normalize_inverse(df["APEX_mean_MIC"], 40, 220)
    df["worst_component"] = normalize_inverse(df["APEX_worst_MIC"], 180, 900)
    df["breadth_component"] = normalize_direct(df["organisms_MIC_le_64"], 8, 30)
    df["novelty_component"] = clip01(df["novelty_score"].fillna(0.5))
    df["developability_ok"] = df.apply(property_ok, axis=1)
    df["developability_component"] = df["developability_ok"].astype(float)

    df["v4a_landscape_score"] = (
        0.27 * df["potency_component"]
        + 0.18 * df["breadth_component"]
        + 0.18 * df["worst_component"]
        + 0.12 * df["mean_component"]
        + 0.12 * df["novelty_component"]
        + 0.13 * df["developability_component"]
    )

    df["v4a_class"] = df.apply(assign_class, axis=1)
    df["failure_modes"] = df.apply(lambda r: ";".join(failure_modes(r)), axis=1)
    df = df.sort_values(["v4a_landscape_score", "APEX_median_MIC", "APEX_worst_MIC"], ascending=[False, True, True]).reset_index(drop=True)
    df.insert(0, "v4a_landscape_rank", range(1, len(df) + 1))

    landscape = landscape_dir / "candidate_landscape.csv"
    classes = landscape_dir / "candidate_class_assignments.csv"
    rescue = rescue_dir / "g_rescue_candidates.csv"
    summary = landscape_dir / "candidate_landscape_summary.json"

    df.to_csv(landscape, index=False)
    df[["v4a_landscape_rank", "candidate_id", "sequence", "v4a_class", "failure_modes", "v4a_landscape_score"]].to_csv(classes, index=False)
    df[df["v4a_class"].eq("G_rescue")].to_csv(rescue, index=False)

    stats = {
        "input": args.input,
        "total_candidates": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["v4a_class"].value_counts().to_dict().items()},
        "outputs": {"landscape": str(landscape), "classes": str(classes), "g_rescue": str(rescue)},
    }
    summary.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

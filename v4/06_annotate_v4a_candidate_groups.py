#!/usr/bin/env python3
"""Annotate V4A candidates with Elite, Pareto, potency, and spectrum labels.

This script can be run after an existing V4A pipeline without repeating APEX
scoring or optimization. It merges seed and optimized candidate tables, imports
Pareto membership from the existing Pareto front, applies absolute Elite
criteria, and writes dedicated candidate-group CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDRO = set("AILMFWYV")
AROM = set("FWY")


def clean_sequence(value: object) -> str:
    return "".join(x for x in str(value).strip().upper() if x in AA)


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sequence"] = out["sequence"].map(clean_sequence)
    seq = out["sequence"]
    out["length"] = seq.map(len)
    out["net_charge_KR_minus_DE"] = seq.map(
        lambda s: s.count("K") + s.count("R") - s.count("D") - s.count("E")
    )
    out["hydrophobic_fraction"] = seq.map(
        lambda s: sum(a in HYDRO for a in s) / max(len(s), 1)
    )
    out["cysteine_count"] = seq.map(lambda s: s.count("C"))
    out["tryptophan_count"] = seq.map(lambda s: s.count("W"))
    out["aromatic_fraction"] = seq.map(
        lambda s: sum(a in AROM for a in s) / max(len(s), 1)
    )
    return out


def developability(row: pd.Series) -> float:
    score = 1.0
    if not 8 <= row["length"] <= 35:
        score -= 0.25
    if not 2 <= row["net_charge_KR_minus_DE"] <= 10:
        score -= 0.20
    if not 0.25 <= row["hydrophobic_fraction"] <= 0.62:
        score -= 0.20
    if row["cysteine_count"] > 2:
        score -= 0.15
    if row["tryptophan_count"] > 3:
        score -= 0.10
    if row["aromatic_fraction"] > 0.30:
        score -= 0.10
    return max(score, 0.0)


def load_candidates(seed_path: Path, variant_path: Path) -> pd.DataFrame:
    seed = pd.read_csv(seed_path, low_memory=False)
    seed["v4a_record_type"] = "seed"
    variants = pd.read_csv(variant_path, low_memory=False)
    variants["v4a_record_type"] = "optimized_or_rescued_variant"
    df = pd.concat([seed, variants], ignore_index=True, sort=False)
    df = add_sequence_features(df)
    return df[df["sequence"].str.len() > 0].drop_duplicates("sequence").copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-scored",
        default="v4/results/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv",
    )
    parser.add_argument(
        "--variant-gains",
        default="v4/results/optimization/optimized_variants_with_gains.csv",
    )
    parser.add_argument(
        "--pareto",
        default="v4/results/final_panel/v4a_pareto_front.csv",
    )
    parser.add_argument("--outdir", default="v4/results/final_panel")
    parser.add_argument("--elite-max-best-mic", type=float, default=20.0)
    parser.add_argument("--elite-max-mean-mic", type=float, default=80.0)
    parser.add_argument("--elite-max-median-mic", type=float, default=32.0)
    parser.add_argument("--elite-max-worst-mic", type=float, default=512.0)
    parser.add_argument("--elite-min-fraction-mic-le64", type=float, default=0.60)
    parser.add_argument("--elite-min-developability", type=float, default=0.70)
    args = parser.parse_args()

    seed_path = Path(args.seed_scored)
    variant_path = Path(args.variant_gains)
    pareto_path = Path(args.pareto)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_candidates(seed_path, variant_path)

    numeric = [
        "APEX_best_MIC",
        "APEX_mean_MIC",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "fraction_MIC_le_64",
    ]
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["fraction_MIC_le_64"].isna().all():
        df["fraction_MIC_le_64"] = (
            df["organisms_MIC_le_64"] / 40.0
        ).clip(0, 1)

    df["developability_component"] = df.apply(developability, axis=1)

    # Requested independent candidate labels.
    df["is_potent_any_organism"] = df["APEX_best_MIC"] < 5
    df["is_narrow_spectrum_specialist"] = (
        (df["APEX_best_MIC"] < 5)
        & (df["fraction_MIC_le_64"] <= 0.35)
    )
    df["is_broad_spectrum"] = df["fraction_MIC_le_64"] >= 0.70

    # Absolute Elite status, independent of Pareto status.
    df["is_elite"] = (
        (df["APEX_best_MIC"] <= args.elite_max_best_mic)
        & (df["APEX_mean_MIC"] <= args.elite_max_mean_mic)
        & (df["APEX_median_MIC"] <= args.elite_max_median_mic)
        & (df["APEX_worst_MIC"] <= args.elite_max_worst_mic)
        & (df["fraction_MIC_le_64"] >= args.elite_min_fraction_mic_le64)
        & (df["developability_component"] >= args.elite_min_developability)
    )

    pareto_sequences: set[str] = set()
    if pareto_path.exists():
        pareto = pd.read_csv(pareto_path, low_memory=False)
        if "sequence" in pareto.columns:
            pareto_sequences = set(pareto["sequence"].map(clean_sequence))

    df["is_pareto"] = df["sequence"].isin(pareto_sequences)
    df["is_elite_pareto"] = df["is_elite"] & df["is_pareto"]

    # Sanity-oriented reporting pool; group labels are retained independently.
    sane = df[
        df["length"].between(8, 35)
        & df["net_charge_KR_minus_DE"].between(1, 12)
        & df["hydrophobic_fraction"].between(0.18, 0.70)
        & (df["cysteine_count"] <= 3)
        & (df["tryptophan_count"] <= 4)
    ].copy()

    outputs = {
        "all": outdir / "v4a_candidate_groups_all.csv",
        "elite": outdir / "v4a_elite_candidates.csv",
        "pareto": outdir / "v4a_pareto_candidates_annotated.csv",
        "elite_pareto": outdir / "v4a_elite_pareto_candidates.csv",
        "potent_any": outdir / "v4a_potent_any_organism.csv",
        "narrow": outdir / "v4a_narrow_spectrum_specialists.csv",
        "broad": outdir / "v4a_broad_spectrum_candidates.csv",
    }

    sane.to_csv(outputs["all"], index=False)
    sane[sane["is_elite"]].to_csv(outputs["elite"], index=False)
    sane[sane["is_pareto"]].to_csv(outputs["pareto"], index=False)
    sane[sane["is_elite_pareto"]].to_csv(outputs["elite_pareto"], index=False)
    sane[sane["is_potent_any_organism"]].to_csv(outputs["potent_any"], index=False)
    sane[sane["is_narrow_spectrum_specialist"]].to_csv(outputs["narrow"], index=False)
    sane[sane["is_broad_spectrum"]].to_csv(outputs["broad"], index=False)

    summary = {
        "candidate_pool": int(len(sane)),
        "elite_candidates": int(sane["is_elite"].sum()),
        "pareto_candidates": int(sane["is_pareto"].sum()),
        "elite_pareto_candidates": int(sane["is_elite_pareto"].sum()),
        "potent_any_organism": int(sane["is_potent_any_organism"].sum()),
        "narrow_spectrum_specialists": int(
            sane["is_narrow_spectrum_specialist"].sum()
        ),
        "broad_spectrum_candidates": int(sane["is_broad_spectrum"].sum()),
        "definitions": {
            "potent_any": "APEX_best_MIC < 5",
            "narrow_specialist": (
                "APEX_best_MIC < 5 and fraction_MIC_le_64 <= 0.35"
            ),
            "broad_spectrum": "fraction_MIC_le_64 >= 0.70",
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }

    summary_path = outdir / "v4a_candidate_group_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

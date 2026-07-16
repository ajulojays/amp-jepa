#!/usr/bin/env python3
"""Select the final V4A Pareto panel and annotate biological candidate groups.

Independent labels added to every candidate:
- is_potent_any_organism: APEX_best_MIC < 5
- is_narrow_spectrum_specialist: APEX_best_MIC < 5 and fraction_MIC_le_64 <= 0.35
- is_broad_spectrum: fraction_MIC_le_64 >= 0.70
- is_elite: passes absolute V4A quality thresholds
- is_pareto: non-dominated within the Pareto prefilter
- is_elite_pareto: elite and Pareto

Elite and Pareto remain parallel candidate labels. Potent specialist and spectrum
labels are also independent so narrow-spectrum biological opportunities are retained.
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


def clean_sequence(x):
    return "".join(ch for ch in str(x).strip().upper() if ch in AA)


def add_features(df):
    df = df.copy()
    df["sequence"] = df["sequence"].map(clean_sequence)
    seqs = df["sequence"]
    df["length"] = seqs.map(len)
    df["net_charge_KR_minus_DE"] = seqs.map(
        lambda s: s.count("K") + s.count("R") - s.count("D") - s.count("E")
    )
    df["hydrophobic_fraction"] = seqs.map(
        lambda s: sum(ch in HYDRO for ch in s) / max(len(s), 1)
    )
    df["cysteine_count"] = seqs.map(lambda s: s.count("C"))
    df["tryptophan_count"] = seqs.map(lambda s: s.count("W"))
    df["aromatic_fraction"] = seqs.map(
        lambda s: sum(ch in AROM for ch in s) / max(len(s), 1)
    )
    return df


def clip01(x):
    return np.clip(x, 0, 1)


def inv(series, good, bad):
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(clip01((bad - x) / (bad - good)), index=series.index).fillna(0)


def direct(series, low, high):
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(clip01((x - low) / (high - low)), index=series.index).fillna(0)


def developability_score(row):
    score = 1.0
    if not (8 <= row.length <= 35):
        score -= 0.25
    if not (2 <= row.net_charge_KR_minus_DE <= 10):
        score -= 0.20
    if not (0.25 <= row.hydrophobic_fraction <= 0.62):
        score -= 0.20
    if row.cysteine_count > 2:
        score -= 0.15
    if row.tryptophan_count > 3:
        score -= 0.10
    if row.aromatic_fraction > 0.30:
        score -= 0.10
    return max(score, 0.0)


def pareto_mask(df, cols_min, cols_max):
    """Return a Boolean mask for non-dominated rows."""
    vals_min = df[cols_min].apply(pd.to_numeric, errors="coerce").fillna(np.inf).to_numpy()
    vals_max = df[cols_max].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy()
    n = len(df)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        better_or_equal_min = vals_min <= vals_min[i]
        better_or_equal_max = vals_max >= vals_max[i]
        strictly_better = (
            (vals_min < vals_min[i]).any(axis=1)
            | (vals_max > vals_max[i]).any(axis=1)
        )
        dominates_i = (
            better_or_equal_min.all(axis=1)
            & better_or_equal_max.all(axis=1)
            & strictly_better
        )
        dominates_i[i] = False
        if dominates_i.any():
            keep[i] = False

    return keep


def write_fasta(df, path, n=50):
    with open(path, "w", encoding="utf-8") as handle:
        for _, row in df.head(n).iterrows():
            handle.write(
                f">V4A_rank_{int(row['v4a_final_rank']):03d}"
                f"|source={row.get('v4a_record_type', 'NA')}"
                f"|elite={bool(row.get('is_elite', False))}"
                f"|pareto={bool(row.get('is_pareto', False))}"
                f"|potent_any={bool(row.get('is_potent_any_organism', False))}"
                f"|narrow_specialist={bool(row.get('is_narrow_spectrum_specialist', False))}"
                f"|broad={bool(row.get('is_broad_spectrum', False))}"
                f"|median={row.get('APEX_median_MIC', np.nan):.3f}"
                f"|worst={row.get('APEX_worst_MIC', np.nan):.3f}"
                f"|org64={int(row.get('organisms_MIC_le_64', 0))}"
                f"|score={row.get('v4a_final_score', np.nan):.3f}\n"
            )
            handle.write(str(row["sequence"]) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-scored",
        default="v4/results/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv",
    )
    parser.add_argument(
        "--variant-gains",
        default="v4/results/optimization/optimized_variants_with_gains.csv",
    )
    parser.add_argument("--outdir", default="v4/results/final_panel")
    parser.add_argument("--pareto-prefilter", type=int, default=5000)

    # Absolute elite thresholds. These are independent of Pareto status.
    parser.add_argument("--elite-max-best-mic", type=float, default=20.0)
    parser.add_argument("--elite-max-mean-mic", type=float, default=80.0)
    parser.add_argument("--elite-max-median-mic", type=float, default=32.0)
    parser.add_argument("--elite-max-worst-mic", type=float, default=512.0)
    parser.add_argument("--elite-min-fraction-mic-le64", type=float, default=0.60)
    parser.add_argument("--elite-min-developability", type=float, default=0.70)

    args = parser.parse_args()

    seed = pd.read_csv(args.seed_scored, low_memory=False)
    seed["v4a_record_type"] = "seed"

    variants = pd.read_csv(args.variant_gains, low_memory=False)
    variants["v4a_record_type"] = "optimized_or_rescued_variant"

    df = pd.concat([seed, variants], ignore_index=True, sort=False)
    df = add_features(df)
    df = df[df["sequence"].str.len() > 0].drop_duplicates("sequence").copy()

    numeric_columns = [
        "APEX_best_MIC",
        "APEX_mean_MIC",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "fraction_MIC_le_64",
        "novelty_score",
        "delta_median_MIC",
        "delta_worst_MIC",
        "delta_organisms_MIC_le_64",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Fallback for older scoring outputs without fraction_MIC_le_64.
    if df["fraction_MIC_le_64"].isna().all():
        organism_count = 0
        if "APEX_scored_organism_count" in df.columns:
            organism_count = pd.to_numeric(
                df["APEX_scored_organism_count"], errors="coerce"
            )
        else:
            # Current full APEX panel has 40 organism/model outputs.
            organism_count = 40.0
        df["fraction_MIC_le_64"] = (
            df["organisms_MIC_le_64"] / organism_count
        ).clip(0, 1)

    if df["novelty_score"].isna().all():
        if "max_train_identity" in df.columns:
            df["novelty_score"] = 1 - pd.to_numeric(
                df["max_train_identity"], errors="coerce"
            )
        else:
            df["novelty_score"] = 0.50

    df["developability_component"] = df.apply(developability_score, axis=1)
    df["potency_component"] = inv(df["APEX_median_MIC"], 20, 160)
    df["mean_component"] = inv(df["APEX_mean_MIC"], 40, 220)
    df["worst_component"] = inv(df["APEX_worst_MIC"], 180, 900)
    df["breadth_component"] = direct(df["organisms_MIC_le_64"], 8, 30)
    df["novelty_component"] = clip01(df["novelty_score"].fillna(0.5))
    df["gain_component"] = (
        df["delta_median_MIC"].fillna(0).clip(-100, 100) / 100
        + df["delta_worst_MIC"].fillna(0).clip(-300, 300) / 300
        + df["delta_organisms_MIC_le_64"].fillna(0).clip(-10, 10) / 10
    ).clip(lower=-1, upper=1)

    df["v4a_final_score"] = (
        0.25 * df["potency_component"]
        + 0.18 * df["breadth_component"]
        + 0.17 * df["worst_component"]
        + 0.10 * df["mean_component"]
        + 0.10 * df["novelty_component"]
        + 0.12 * df["developability_component"]
        + 0.08 * df["gain_component"].clip(lower=0)
    )

    # -------------------------------------------------------------------------
    # Independent biological candidate labels requested for V4A.
    # -------------------------------------------------------------------------
    df["is_potent_any_organism"] = df["APEX_best_MIC"] < 5

    df["is_narrow_spectrum_specialist"] = (
        (df["APEX_best_MIC"] < 5)
        & (df["fraction_MIC_le_64"] <= 0.35)
    )

    df["is_broad_spectrum"] = df["fraction_MIC_le_64"] >= 0.70

    # Elite is an absolute-quality label and remains independent of Pareto.
    df["is_elite"] = (
        (df["APEX_best_MIC"] <= args.elite_max_best_mic)
        & (df["APEX_mean_MIC"] <= args.elite_max_mean_mic)
        & (df["APEX_median_MIC"] <= args.elite_max_median_mic)
        & (df["APEX_worst_MIC"] <= args.elite_max_worst_mic)
        & (df["fraction_MIC_le_64"] >= args.elite_min_fraction_mic_le64)
        & (df["developability_component"] >= args.elite_min_developability)
    )

    # Hard sanity filter for synthesis-oriented candidate reporting.
    final = df[
        df["length"].between(8, 35)
        & df["net_charge_KR_minus_DE"].between(1, 12)
        & df["hydrophobic_fraction"].between(0.18, 0.70)
        & (df["cysteine_count"] <= 3)
        & (df["tryptophan_count"] <= 4)
    ].copy()

    final = final.sort_values(
        [
            "v4a_final_score",
            "APEX_median_MIC",
            "APEX_worst_MIC",
            "organisms_MIC_le_64",
        ],
        ascending=[False, True, True, False],
    )

    pre = final.head(args.pareto_prefilter).copy().reset_index(drop=True)
    pre["is_pareto"] = pareto_mask(
        pre,
        cols_min=["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"],
        cols_max=[
            "organisms_MIC_le_64",
            "novelty_component",
            "developability_component",
        ],
    )

    # Candidates outside the Pareto prefilter are explicitly non-Pareto.
    final["is_pareto"] = False
    final.loc[pre.index, "is_pareto"] = pre["is_pareto"].to_numpy()
    final["is_elite_pareto"] = final["is_elite"] & final["is_pareto"]

    pareto = final[final["is_pareto"]].copy()
    pareto = pareto.sort_values(
        ["v4a_final_score", "APEX_median_MIC", "APEX_worst_MIC"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    pareto.insert(0, "v4a_final_rank", range(1, len(pareto) + 1))

    elite = final[final["is_elite"]].copy().reset_index(drop=True)
    elite_pareto = final[final["is_elite_pareto"]].copy().reset_index(drop=True)
    potent_any = final[final["is_potent_any_organism"]].copy().reset_index(drop=True)
    narrow_specialists = final[
        final["is_narrow_spectrum_specialist"]
    ].copy().reset_index(drop=True)
    broad_spectrum = final[final["is_broad_spectrum"]].copy().reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    final.to_csv(outdir / "v4a_all_sanity_filtered_candidates.csv", index=False)
    elite.to_csv(outdir / "v4a_elite_candidates.csv", index=False)
    pareto.to_csv(outdir / "v4a_pareto_front.csv", index=False)
    elite_pareto.to_csv(outdir / "v4a_elite_pareto_candidates.csv", index=False)
    potent_any.to_csv(outdir / "v4a_potent_any_organism.csv", index=False)
    narrow_specialists.to_csv(
        outdir / "v4a_narrow_spectrum_specialists.csv", index=False
    )
    broad_spectrum.to_csv(outdir / "v4a_broad_spectrum_candidates.csv", index=False)

    pareto.head(20).to_csv(outdir / "v4a_top20_panel.csv", index=False)
    pareto.head(50).to_csv(outdir / "v4a_top50_panel.csv", index=False)
    write_fasta(pareto, outdir / "v4a_top50_panel.fasta", n=50)

    summary = {
        "total_seed_plus_variant_candidates": int(len(df)),
        "after_final_sanity_filter": int(len(final)),
        "elite_candidates": int(final["is_elite"].sum()),
        "pareto_front_size": int(final["is_pareto"].sum()),
        "elite_pareto_candidates": int(final["is_elite_pareto"].sum()),
        "potent_any_organism_candidates": int(
            final["is_potent_any_organism"].sum()
        ),
        "narrow_spectrum_specialists": int(
            final["is_narrow_spectrum_specialist"].sum()
        ),
        "broad_spectrum_candidates": int(final["is_broad_spectrum"].sum()),
        "definitions": {
            "is_potent_any_organism": "APEX_best_MIC < 5",
            "is_narrow_spectrum_specialist": (
                "APEX_best_MIC < 5 and fraction_MIC_le_64 <= 0.35"
            ),
            "is_broad_spectrum": "fraction_MIC_le_64 >= 0.70",
            "is_elite": {
                "APEX_best_MIC_max": args.elite_max_best_mic,
                "APEX_mean_MIC_max": args.elite_max_mean_mic,
                "APEX_median_MIC_max": args.elite_max_median_mic,
                "APEX_worst_MIC_max": args.elite_max_worst_mic,
                "fraction_MIC_le_64_min": args.elite_min_fraction_mic_le64,
                "developability_component_min": args.elite_min_developability,
            },
        },
        "top_sequence": str(pareto.iloc[0]["sequence"]) if len(pareto) else None,
        "top_score": (
            float(pareto.iloc[0]["v4a_final_score"]) if len(pareto) else None
        ),
        "outputs": {
            "all_sanity_filtered": str(
                outdir / "v4a_all_sanity_filtered_candidates.csv"
            ),
            "elite": str(outdir / "v4a_elite_candidates.csv"),
            "pareto": str(outdir / "v4a_pareto_front.csv"),
            "elite_pareto": str(outdir / "v4a_elite_pareto_candidates.csv"),
            "potent_any": str(outdir / "v4a_potent_any_organism.csv"),
            "narrow_specialists": str(
                outdir / "v4a_narrow_spectrum_specialists.csv"
            ),
            "broad_spectrum": str(
                outdir / "v4a_broad_spectrum_candidates.csv"
            ),
            "top20": str(outdir / "v4a_top20_panel.csv"),
            "top50": str(outdir / "v4a_top50_panel.csv"),
            "fasta": str(outdir / "v4a_top50_panel.fasta"),
        },
    }

    (outdir / "v4a_final_panel_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

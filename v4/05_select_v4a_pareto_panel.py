#!/usr/bin/env python3
"""Select the final V4A Pareto panel from seed and optimized candidates."""

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
    df["net_charge_KR_minus_DE"] = seqs.map(lambda s: s.count("K") + s.count("R") - s.count("D") - s.count("E"))
    df["hydrophobic_fraction"] = seqs.map(lambda s: sum(ch in HYDRO for ch in s) / max(len(s), 1))
    df["cysteine_count"] = seqs.map(lambda s: s.count("C"))
    df["tryptophan_count"] = seqs.map(lambda s: s.count("W"))
    df["aromatic_fraction"] = seqs.map(lambda s: sum(ch in AROM for ch in s) / max(len(s), 1))
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
    if not (8 <= row.length <= 35): score -= 0.25
    if not (2 <= row.net_charge_KR_minus_DE <= 10): score -= 0.20
    if not (0.25 <= row.hydrophobic_fraction <= 0.62): score -= 0.20
    if row.cysteine_count > 2: score -= 0.15
    if row.tryptophan_count > 3: score -= 0.10
    if row.aromatic_fraction > 0.30: score -= 0.10
    return max(score, 0.0)


def pareto_mask(df, cols_min, cols_max):
    # O(n^2), intentionally applied after prefiltering/sorting.
    vals_min = df[cols_min].apply(pd.to_numeric, errors="coerce").fillna(np.inf).to_numpy()
    vals_max = df[cols_max].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy()
    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        better_or_equal_min = vals_min <= vals_min[i]
        better_or_equal_max = vals_max >= vals_max[i]
        strictly_better = (vals_min < vals_min[i]).any(axis=1) | (vals_max > vals_max[i]).any(axis=1)
        dominates_i = better_or_equal_min.all(axis=1) & better_or_equal_max.all(axis=1) & strictly_better
        dominates_i[i] = False
        if dominates_i.any():
            keep[i] = False
    return keep


def write_fasta(df, path, n=50):
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.head(n).iterrows():
            f.write(
                f">V4A_rank_{int(r['v4a_final_rank']):03d}|source={r.get('v4a_record_type','NA')}|median={r.get('APEX_median_MIC', np.nan):.3f}|worst={r.get('APEX_worst_MIC', np.nan):.3f}|org64={int(r.get('organisms_MIC_le_64', 0))}|score={r.get('v4a_final_score', np.nan):.3f}\n"
            )
            f.write(str(r["sequence"]) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-scored", default="v4/results/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv")
    ap.add_argument("--variant-gains", default="v4/results/optimization/optimized_variants_with_gains.csv")
    ap.add_argument("--outdir", default="v4/results/final_panel")
    ap.add_argument("--pareto-prefilter", type=int, default=5000)
    args = ap.parse_args()

    seed = pd.read_csv(args.seed_scored, low_memory=False)
    seed["v4a_record_type"] = "seed"
    var = pd.read_csv(args.variant_gains, low_memory=False)
    var["v4a_record_type"] = "optimized_or_rescued_variant"

    df = pd.concat([seed, var], ignore_index=True, sort=False)
    df = add_features(df)
    df = df[df["sequence"].str.len() > 0].drop_duplicates("sequence").copy()

    for c in ["APEX_mean_MIC", "APEX_median_MIC", "APEX_worst_MIC", "organisms_MIC_le_64", "novelty_score", "delta_median_MIC", "delta_worst_MIC", "delta_organisms_MIC_le_64"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["novelty_score"].isna().all():
        if "max_train_identity" in df.columns:
            df["novelty_score"] = 1 - pd.to_numeric(df["max_train_identity"], errors="coerce")
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

    # Hard sanity filter for final panel, but still keep broad biological range.
    final = df[
        df["length"].between(8, 35)
        & df["net_charge_KR_minus_DE"].between(1, 12)
        & df["hydrophobic_fraction"].between(0.18, 0.70)
        & (df["cysteine_count"] <= 3)
        & (df["tryptophan_count"] <= 4)
    ].copy()

    final = final.sort_values(["v4a_final_score", "APEX_median_MIC", "APEX_worst_MIC", "organisms_MIC_le_64"], ascending=[False, True, True, False])
    pre = final.head(args.pareto_prefilter).copy().reset_index(drop=True)
    pre["is_pareto"] = pareto_mask(
        pre,
        cols_min=["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"],
        cols_max=["organisms_MIC_le_64", "novelty_component", "developability_component"],
    )
    pareto = pre[pre["is_pareto"]].copy()
    pareto = pareto.sort_values(["v4a_final_score", "APEX_median_MIC", "APEX_worst_MIC"], ascending=[False, True, True]).reset_index(drop=True)
    pareto.insert(0, "v4a_final_rank", range(1, len(pareto) + 1))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(outdir / "v4a_pareto_front.csv", index=False)
    pareto.head(20).to_csv(outdir / "v4a_top20_panel.csv", index=False)
    pareto.head(50).to_csv(outdir / "v4a_top50_panel.csv", index=False)
    write_fasta(pareto, outdir / "v4a_top50_panel.fasta", n=50)

    summary = {
        "total_seed_plus_variant_candidates": int(len(df)),
        "after_final_sanity_filter": int(len(final)),
        "pareto_front_size": int(len(pareto)),
        "top_sequence": str(pareto.iloc[0]["sequence"]) if len(pareto) else None,
        "top_score": float(pareto.iloc[0]["v4a_final_score"]) if len(pareto) else None,
        "outputs": {
            "pareto": str(outdir / "v4a_pareto_front.csv"),
            "top20": str(outdir / "v4a_top20_panel.csv"),
            "top50": str(outdir / "v4a_top50_panel.csv"),
            "fasta": str(outdir / "v4a_top50_panel.fasta"),
        },
    }
    (outdir / "v4a_final_panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

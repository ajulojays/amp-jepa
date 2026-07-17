#!/usr/bin/env python3
"""Summarize V4B generations and select a final hydrophobicity-balanced panel.

This script answers two questions after the closed-loop V4B run:

1. Across generations 1..10, how many candidates meet the original generation
   constraints?
2. Among APEX-scored candidates, which candidates are best after adding a
   soft hydrophobicity/developability preference?

The original constraints here match the V4B generator defaults:
- canonical amino acids only
- length 10..40
- net charge 2..12
- hydrophobic_fraction 0.20..0.70

The final panel is selected with hydrophobicity as a soft preference, not as a
single hard filter. This prevents collapse into very hydrophobic short peptides
while preserving strong APEX potency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).upper().split())


def minmax_good_high(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    x = x.fillna(x.median())
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return pd.Series(np.ones(len(x)), index=x.index, dtype=float)
    return (x - lo) / (hi - lo)


def minmax_good_low(x: pd.Series) -> pd.Series:
    return 1.0 - minmax_good_high(x)


def load_scored_generations(results_dir: Path, start: int, end: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for g in range(start, end + 1):
        gtag = f"generation_{g:02d}"
        path = results_dir / gtag / f"{gtag}_candidates_scored.csv"
        if not path.exists():
            print(f"[WARN] Missing scored file, skipping: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["source_generation"] = g
        df["source_file"] = str(path)
        pieces.append(df)
    if not pieces:
        raise FileNotFoundError(f"No scored generation files found in {results_dir}")
    all_df = pd.concat(pieces, ignore_index=True, sort=False)
    all_df["sequence"] = all_df["sequence"].map(clean_sequence)
    return all_df


def add_criteria_flags(
    df: pd.DataFrame,
    min_len: int,
    max_len: int,
    min_charge: float,
    max_charge: float,
    min_hydro: float,
    max_hydro: float,
    preferred_hydro_min: float,
    preferred_hydro_max: float,
    preferred_charge_min: float,
    preferred_charge_max: float,
    preferred_length_min: int,
    preferred_length_max: int,
    apex_median_cutoff: float,
    apex_worst_cutoff: float,
    min_organisms_le64: int,
) -> pd.DataFrame:
    out = df.copy()
    seq = out["sequence"].fillna("").astype(str)
    out["criteria_canonical"] = seq.map(lambda s: bool(s) and all(aa in CANONICAL for aa in s))

    for col in ["length", "net_charge_KR_minus_DE", "hydrophobic_fraction"]:
        if col not in out.columns:
            raise ValueError(f"Required column missing: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["criteria_length_initial"] = out["length"].between(min_len, max_len, inclusive="both")
    out["criteria_charge_initial"] = out["net_charge_KR_minus_DE"].between(min_charge, max_charge, inclusive="both")
    out["criteria_hydro_initial"] = out["hydrophobic_fraction"].between(min_hydro, max_hydro, inclusive="both")
    out["meets_initial_generation_criteria"] = (
        out["criteria_canonical"]
        & out["criteria_length_initial"]
        & out["criteria_charge_initial"]
        & out["criteria_hydro_initial"]
    )

    out["preferred_hydro_window"] = out["hydrophobic_fraction"].between(preferred_hydro_min, preferred_hydro_max, inclusive="both")
    out["preferred_charge_window"] = out["net_charge_KR_minus_DE"].between(preferred_charge_min, preferred_charge_max, inclusive="both")
    out["preferred_length_window"] = out["length"].between(preferred_length_min, preferred_length_max, inclusive="both")

    if "APEX_median_MIC" in out.columns:
        out["APEX_median_MIC"] = pd.to_numeric(out["APEX_median_MIC"], errors="coerce")
        out["apex_median_pass"] = out["APEX_median_MIC"] <= apex_median_cutoff
    else:
        out["apex_median_pass"] = False

    if "APEX_worst_MIC" in out.columns:
        out["APEX_worst_MIC"] = pd.to_numeric(out["APEX_worst_MIC"], errors="coerce")
        out["apex_worst_pass"] = out["APEX_worst_MIC"] <= apex_worst_cutoff
    else:
        out["apex_worst_pass"] = False

    if "organisms_MIC_le_64" in out.columns:
        out["organisms_MIC_le_64"] = pd.to_numeric(out["organisms_MIC_le_64"], errors="coerce")
        out["breadth_pass"] = out["organisms_MIC_le_64"] >= min_organisms_le64
    else:
        out["breadth_pass"] = False

    out["balanced_lead_like"] = (
        out["meets_initial_generation_criteria"]
        & out["preferred_hydro_window"]
        & out["preferred_charge_window"]
        & out["preferred_length_window"]
        & out["apex_median_pass"]
        & out["breadth_pass"]
    )
    return out


def add_scores(df: pd.DataFrame, hydro_target: float) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    out["hydro_distance_from_target"] = (out["hydrophobic_fraction"] - hydro_target).abs()
    out["hydro_balance_score"] = (1.0 - (out["hydro_distance_from_target"] / 0.30)).clip(lower=0.0, upper=1.0)

    if "APEX_median_MIC" in out.columns:
        median_score = minmax_good_low(out["APEX_median_MIC"])
    else:
        median_score = pd.Series(np.zeros(n), index=out.index, dtype=float)
    if "APEX_worst_MIC" in out.columns:
        worst_score = minmax_good_low(out["APEX_worst_MIC"])
    else:
        worst_score = pd.Series(np.zeros(n), index=out.index, dtype=float)
    if "APEX_mean_MIC" in out.columns:
        mean_score = minmax_good_low(out["APEX_mean_MIC"])
    else:
        mean_score = pd.Series(np.zeros(n), index=out.index, dtype=float)
    if "organisms_MIC_le_64" in out.columns:
        breadth_score = minmax_good_high(out["organisms_MIC_le_64"])
    else:
        breadth_score = pd.Series(np.zeros(n), index=out.index, dtype=float)

    charge = pd.to_numeric(out["net_charge_KR_minus_DE"], errors="coerce").fillna(0)
    out["charge_balance_score"] = (1.0 - ((charge - 5.5).abs() / 5.5)).clip(lower=0.0, upper=1.0)

    length = pd.to_numeric(out["length"], errors="coerce").fillna(0)
    out["length_balance_score"] = (1.0 - ((length - 16.0).abs() / 16.0)).clip(lower=0.0, upper=1.0)

    out["v4b_final_balanced_score"] = (
        0.30 * median_score
        + 0.20 * worst_score
        + 0.10 * mean_score
        + 0.20 * breadth_score
        + 0.12 * out["hydro_balance_score"]
        + 0.05 * out["charge_balance_score"]
        + 0.03 * out["length_balance_score"]
    )

    out["v4b_potency_score"] = (
        0.45 * median_score
        + 0.25 * worst_score
        + 0.15 * mean_score
        + 0.15 * breadth_score
    )
    return out


def dedupe_by_sequence(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["v4b_final_balanced_score"]
    ascending = [False]
    if "APEX_median_MIC" in df.columns:
        sort_cols.append("APEX_median_MIC")
        ascending.append(True)
    ranked = df.sort_values(sort_cols, ascending=ascending, na_position="last").copy()
    ranked = ranked.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    return ranked


def pick_panel(df: pd.DataFrame, n_final: int) -> pd.DataFrame:
    """Pick a final panel across potency, balanced, and diversity buckets."""
    selected: list[pd.DataFrame] = []
    used: set[str] = set()

    def add_bucket(label: str, pool: pd.DataFrame, k: int, sort_cols: list[str], ascending: list[bool]) -> None:
        nonlocal selected, used
        if k <= 0 or pool.empty:
            return
        p = pool.copy()
        p = p[~p["sequence"].isin(used)]
        if p.empty:
            return
        p = p.sort_values(sort_cols, ascending=ascending, na_position="last").head(k).copy()
        p["final_panel_bucket"] = label
        used.update(p["sequence"].astype(str))
        selected.append(p)

    n_balanced = int(round(n_final * 0.50))
    n_potency = int(round(n_final * 0.25))
    n_low_hydro = int(round(n_final * 0.10))
    n_moderate_hydro = int(round(n_final * 0.10))
    n_frontier = max(0, n_final - n_balanced - n_potency - n_low_hydro - n_moderate_hydro)

    balanced_pool = df[df["balanced_lead_like"]].copy()
    add_bucket(
        "balanced_developability",
        balanced_pool,
        n_balanced,
        ["v4b_final_balanced_score", "APEX_median_MIC"],
        [False, True],
    )

    add_bucket(
        "best_apex_potency",
        df[df["meets_initial_generation_criteria"]].copy(),
        n_potency,
        ["v4b_potency_score", "APEX_median_MIC"],
        [False, True],
    )

    low_hydro_pool = df[
        df["meets_initial_generation_criteria"]
        & df["hydrophobic_fraction"].between(0.45, 0.55, inclusive="both")
    ].copy()
    add_bucket(
        "lower_hydrophobicity_reserve",
        low_hydro_pool,
        n_low_hydro,
        ["v4b_final_balanced_score", "APEX_median_MIC"],
        [False, True],
    )

    moderate_hydro_pool = df[
        df["meets_initial_generation_criteria"]
        & df["hydrophobic_fraction"].between(0.55, 0.62, inclusive="both")
    ].copy()
    add_bucket(
        "moderate_hydrophobicity_core",
        moderate_hydro_pool,
        n_moderate_hydro,
        ["v4b_final_balanced_score", "APEX_median_MIC"],
        [False, True],
    )

    frontier_pool = df[
        df["meets_initial_generation_criteria"]
        & df["hydrophobic_fraction"].between(0.62, 0.68, inclusive="both")
        & (df["net_charge_KR_minus_DE"] >= 4)
    ].copy()
    add_bucket(
        "potency_risk_frontier",
        frontier_pool,
        n_frontier,
        ["v4b_potency_score", "APEX_median_MIC"],
        [False, True],
    )

    if selected:
        panel = pd.concat(selected, ignore_index=True, sort=False)
    else:
        panel = pd.DataFrame()

    if len(panel) < n_final:
        remainder = df[~df["sequence"].isin(used)].copy()
        add_bucket(
            "score_backfill",
            remainder,
            n_final - len(panel),
            ["v4b_final_balanced_score", "APEX_median_MIC"],
            [False, True],
        )
        panel = pd.concat(selected, ignore_index=True, sort=False) if selected else panel

    panel = panel.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    panel.insert(0, "final_panel_rank", range(1, len(panel) + 1))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="v4b/results")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--end-generation", type=int, default=10)
    parser.add_argument("--outdir", default="v4b/results/final_selection")
    parser.add_argument("--n-final", type=int, default=96)

    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--min-charge", type=float, default=2)
    parser.add_argument("--max-charge", type=float, default=12)
    parser.add_argument("--min-hydro", type=float, default=0.20)
    parser.add_argument("--max-hydro", type=float, default=0.70)

    parser.add_argument("--preferred-hydro-min", type=float, default=0.45)
    parser.add_argument("--preferred-hydro-max", type=float, default=0.60)
    parser.add_argument("--hydro-target", type=float, default=0.54)
    parser.add_argument("--preferred-charge-min", type=float, default=4)
    parser.add_argument("--preferred-charge-max", type=float, default=8)
    parser.add_argument("--preferred-length-min", type=int, default=12)
    parser.add_argument("--preferred-length-max", type=int, default=25)

    parser.add_argument("--apex-median-cutoff", type=float, default=16.0)
    parser.add_argument("--apex-worst-cutoff", type=float, default=350.0)
    parser.add_argument("--min-organisms-le64", type=int, default=28)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_scored_generations(results_dir, args.start_generation, args.end_generation)
    df = add_criteria_flags(
        df,
        min_len=args.min_len,
        max_len=args.max_len,
        min_charge=args.min_charge,
        max_charge=args.max_charge,
        min_hydro=args.min_hydro,
        max_hydro=args.max_hydro,
        preferred_hydro_min=args.preferred_hydro_min,
        preferred_hydro_max=args.preferred_hydro_max,
        preferred_charge_min=args.preferred_charge_min,
        preferred_charge_max=args.preferred_charge_max,
        preferred_length_min=args.preferred_length_min,
        preferred_length_max=args.preferred_length_max,
        apex_median_cutoff=args.apex_median_cutoff,
        apex_worst_cutoff=args.apex_worst_cutoff,
        min_organisms_le64=args.min_organisms_le64,
    )
    df = add_scores(df, hydro_target=args.hydro_target)
    deduped = dedupe_by_sequence(df)
    panel = pick_panel(deduped, n_final=args.n_final)

    all_out = outdir / "v4b_all_generations_scored_with_flags.csv"
    dedupe_out = outdir / "v4b_all_generations_unique_ranked.csv"
    panel_out = outdir / "v4b_final_panel_hydrophobicity_balanced.csv"
    summary_out = outdir / "v4b_generation_criteria_summary.csv"
    json_out = outdir / "v4b_final_selection_summary.json"
    fasta_out = outdir / "v4b_final_panel_hydrophobicity_balanced.fasta"

    df.to_csv(all_out, index=False)
    deduped.to_csv(dedupe_out, index=False)
    panel.to_csv(panel_out, index=False)

    rows = []
    for g, sub in df.groupby("source_generation"):
        rows.append({
            "generation": int(g),
            "scored_candidates": int(len(sub)),
            "unique_sequences": int(sub["sequence"].nunique()),
            "meet_initial_generation_criteria": int(sub["meets_initial_generation_criteria"].sum()),
            "meet_initial_generation_criteria_fraction": float(sub["meets_initial_generation_criteria"].mean()),
            "preferred_hydro_0.45_0.60": int(sub["preferred_hydro_window"].sum()),
            "preferred_hydro_fraction": float(sub["preferred_hydro_window"].mean()),
            "balanced_lead_like": int(sub["balanced_lead_like"].sum()),
            "best_APEX_median_MIC": float(pd.to_numeric(sub.get("APEX_median_MIC"), errors="coerce").min()),
            "median_APEX_median_MIC": float(pd.to_numeric(sub.get("APEX_median_MIC"), errors="coerce").median()),
            "mean_hydrophobic_fraction": float(pd.to_numeric(sub["hydrophobic_fraction"], errors="coerce").mean()),
            "median_hydrophobic_fraction": float(pd.to_numeric(sub["hydrophobic_fraction"], errors="coerce").median()),
            "hydro_gt_0.65": int((sub["hydrophobic_fraction"] > 0.65).sum()),
            "hydro_gt_0.65_fraction": float((sub["hydrophobic_fraction"] > 0.65).mean()),
        })
    summary = pd.DataFrame(rows).sort_values("generation")
    summary.to_csv(summary_out, index=False)

    with fasta_out.open("w", encoding="utf-8") as handle:
        for _, row in panel.iterrows():
            handle.write(
                f">{row['candidate_id']}|rank={int(row['final_panel_rank'])}"
                f"|bucket={row['final_panel_bucket']}"
                f"|gen={int(row['source_generation'])}"
                f"|medianMIC={float(row.get('APEX_median_MIC', np.nan)):.3f}"
                f"|worstMIC={float(row.get('APEX_worst_MIC', np.nan)):.3f}"
                f"|hydro={float(row['hydrophobic_fraction']):.3f}"
                f"|charge={float(row['net_charge_KR_minus_DE']):.1f}\n"
            )
            handle.write(f"{row['sequence']}\n")

    payload = {
        "input_generations": [args.start_generation, args.end_generation],
        "total_scored_rows": int(len(df)),
        "total_unique_sequences": int(df["sequence"].nunique()),
        "total_meet_initial_generation_criteria": int(df["meets_initial_generation_criteria"].sum()),
        "total_preferred_hydro_0.45_0.60": int(df["preferred_hydro_window"].sum()),
        "total_balanced_lead_like": int(df["balanced_lead_like"].sum()),
        "unique_ranked_candidates": int(len(deduped)),
        "final_panel_size": int(len(panel)),
        "parameters": vars(args),
        "outputs": {
            "all_scored_with_flags": str(all_out),
            "unique_ranked": str(dedupe_out),
            "final_panel_csv": str(panel_out),
            "final_panel_fasta": str(fasta_out),
            "generation_summary": str(summary_out),
        },
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nGENERATION SUMMARY")
    print(summary.round(3).to_string(index=False))
    print("\nFINAL PANEL BUCKETS")
    if len(panel):
        print(panel["final_panel_bucket"].value_counts().to_string())
        display_cols = [
            "final_panel_rank", "final_panel_bucket", "source_generation", "candidate_id", "sequence",
            "length", "net_charge_KR_minus_DE", "hydrophobic_fraction",
            "APEX_mean_MIC", "APEX_median_MIC", "APEX_worst_MIC", "organisms_MIC_le_64",
            "v4b_final_balanced_score",
        ]
        display_cols = [c for c in display_cols if c in panel.columns]
        print("\nTOP FINAL PANEL")
        print(panel[display_cols].head(30).round(3).to_string(index=False))
    print("\nOutputs:")
    for p in [summary_out, panel_out, fasta_out, dedupe_out, all_out, json_out]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

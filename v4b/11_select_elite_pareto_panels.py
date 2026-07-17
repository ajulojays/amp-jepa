#!/usr/bin/env python3
"""Select V4B AMP-JEPA candidates into elite/Pareto/broad/narrow panels.

This script starts from all scored candidates from generations 1..10, applies the
original G1/V4B generation criteria, deduplicates by sequence, computes a
composite APEX/developability score, and writes selection buckets:

- apex_elite: strongest overall APEX/developability candidates
- pareto_broad: non-dominated candidates from a broad initial-pass pool
- pareto_narrow: non-dominated candidates after stricter developability filters
- hydrophobic_balanced: strong candidates in the preferred hydrophobic window
- broad_spectrum: candidates active across many organisms
- worst_case_robust: candidates with low worst-case predicted MIC
- potency_rescue_high_hydro: very potent but hydrophobic-risk candidates
- generation_diverse: top representatives spread across generations

Hydrophobicity is treated as a developability balance signal, not a hard-only
criterion, after the initial G1 criteria are applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVY")


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).upper().split())


def sequence_features(seq: object) -> tuple[int, int, float, bool]:
    seq = clean_sequence(seq)
    n = max(len(seq), 1)
    charge = sum(aa in POSITIVE for aa in seq) - sum(aa in NEGATIVE for aa in seq)
    hydro = sum(aa in HYDROPHOBIC for aa in seq) / n
    canonical = bool(seq) and all(aa in CANONICAL for aa in seq)
    return len(seq), charge, hydro, canonical


def to_numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return s.fillna(float(s.median()))


def minmax_score(values: Iterable[float], lower_is_better: bool) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    x[~finite] = float(np.nanmedian(x[finite]))
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.ones_like(x, dtype=np.float32)
    score = (x - lo) / (hi - lo)
    if lower_is_better:
        score = 1.0 - score
    return score.astype(np.float32)


def triangular_balance(values: Iterable[float], target: float, half_width: float) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    x = np.nan_to_num(x, nan=target, posinf=target, neginf=target)
    return np.clip(1.0 - (np.abs(x - target) / max(half_width, 1e-9)), 0.0, 1.0).astype(np.float32)


def add_scores(df: pd.DataFrame, hydro_target: float) -> pd.DataFrame:
    out = df.copy()
    median_mic = to_numeric(out, "APEX_median_MIC", default=9999.0)
    worst_mic = to_numeric(out, "APEX_worst_MIC", default=9999.0)
    mean_mic = to_numeric(out, "APEX_mean_MIC", default=9999.0)
    breadth = to_numeric(out, "organisms_MIC_le_64", default=0.0)

    out["score_median_MIC"] = minmax_score(median_mic, lower_is_better=True)
    out["score_worst_MIC"] = minmax_score(worst_mic, lower_is_better=True)
    out["score_mean_MIC"] = minmax_score(mean_mic, lower_is_better=True)
    out["score_breadth"] = minmax_score(breadth, lower_is_better=False)
    out["score_hydro_balance"] = triangular_balance(out["criteria_hydrophobic_fraction"], hydro_target, 0.18)
    out["score_charge_balance"] = triangular_balance(out["criteria_charge"], 5.5, 4.0)
    out["score_length_balance"] = triangular_balance(out["criteria_length"], 17.0, 12.0)

    out["v4b_elite_composite_score"] = (
        0.30 * out["score_median_MIC"]
        + 0.20 * out["score_worst_MIC"]
        + 0.15 * out["score_mean_MIC"]
        + 0.15 * out["score_breadth"]
        + 0.10 * out["score_hydro_balance"]
        + 0.05 * out["score_charge_balance"]
        + 0.05 * out["score_length_balance"]
    )

    out["hydrophobicity_zone"] = pd.cut(
        out["criteria_hydrophobic_fraction"],
        bins=[-0.001, 0.30, 0.45, 0.60, 0.65, 0.70, 1.0],
        labels=[
            "very_low_<0.30",
            "low_0.30_0.45",
            "preferred_0.45_0.60",
            "caution_0.60_0.65",
            "high_risk_0.65_0.70",
            "outside_>0.70",
        ],
    )

    return out


def pareto_front(df: pd.DataFrame, objective_cols: list[str], maximize_cols: set[str]) -> pd.DataFrame:
    """Exact Pareto front for a prefiltered pool.

    Internally converts every objective to minimization, then marks a point as
    dominated if another point is <= in all objectives and < in at least one.
    This is intended for a top pool, not the full 100k table.
    """
    if df.empty:
        return df.copy()

    values = []
    for col in objective_cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        finite = np.isfinite(x)
        if finite.any():
            fill = float(np.nanmedian(x[finite]))
        else:
            fill = 0.0
        x[~finite] = fill
        if col in maximize_cols:
            x = -x
        values.append(x)
    arr = np.vstack(values).T.astype(np.float64)

    n = arr.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        # Vectorized comparison against all points.
        le_all = np.all(arr <= arr[i], axis=1)
        lt_any = np.any(arr < arr[i], axis=1)
        if np.any(le_all & lt_any):
            dominated[i] = True

    return df.loc[~dominated].copy()


def select_head(df: pd.DataFrame, n: int, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(sort_cols, ascending=ascending, na_position="last").head(n).copy()


def add_bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    out = df.copy()
    out["selection_bucket"] = bucket
    return out


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            cid = row.get("candidate_id", "candidate")
            seq = clean_sequence(row.get("sequence", ""))
            bucket = row.get("selection_bucket_union", row.get("selection_bucket", "selected"))
            med = row.get("APEX_median_MIC", "NA")
            worst = row.get("APEX_worst_MIC", "NA")
            hydro = row.get("criteria_hydrophobic_fraction", "NA")
            charge = row.get("criteria_charge", "NA")
            gen = row.get("generation_source", row.get("generation", "NA"))
            handle.write(
                f">{cid}|bucket={bucket}|G={gen}|median_MIC={med}|worst_MIC={worst}|hydro={hydro}|charge={charge}\n"
            )
            handle.write(seq + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="v4b/results")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--end-generation", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hydro-target", type=float, default=0.52)
    parser.add_argument("--pareto-pool", type=int, default=10000)
    parser.add_argument("--n-elite", type=int, default=512)
    parser.add_argument("--n-hydro-balanced", type=int, default=512)
    parser.add_argument("--n-broad-spectrum", type=int, default=512)
    parser.add_argument("--n-worst-case", type=int, default=512)
    parser.add_argument("--n-potency-rescue", type=int, default=256)
    parser.add_argument("--n-generation-diverse-per-generation", type=int, default=64)
    parser.add_argument("--n-final-preview", type=int, default=192)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir / "elite_pareto_selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[pd.DataFrame] = []
    for g in range(args.start_generation, args.end_generation + 1):
        path = results_dir / f"generation_{g:02d}" / f"generation_{g:02d}_candidates_scored.csv"
        if not path.exists():
            print(f"[WARN] Missing scored file for generation {g}: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["generation_source"] = g
        pieces.append(df)

    if not pieces:
        raise SystemExit("No scored generation files found.")

    all_df = pd.concat(pieces, ignore_index=True, sort=False)
    feats = all_df["sequence"].map(sequence_features)
    all_df["criteria_length"] = [x[0] for x in feats]
    all_df["criteria_charge"] = [x[1] for x in feats]
    all_df["criteria_hydrophobic_fraction"] = [x[2] for x in feats]
    all_df["criteria_canonical_aa"] = [x[3] for x in feats]
    all_df["sequence_clean"] = all_df["sequence"].map(clean_sequence)

    all_df["passes_initial_G1_criteria"] = (
        all_df["criteria_canonical_aa"]
        & all_df["criteria_length"].between(10, 40, inclusive="both")
        & all_df["criteria_charge"].between(2, 12, inclusive="both")
        & all_df["criteria_hydrophobic_fraction"].between(0.20, 0.70, inclusive="both")
    )

    initial_pass = all_df.loc[all_df["passes_initial_G1_criteria"]].copy()
    initial_pass = add_scores(initial_pass, hydro_target=args.hydro_target)

    # Keep one copy per sequence, choosing by composite score, median MIC, worst MIC, breadth.
    dedup_sort = [
        "v4b_elite_composite_score",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
    ]
    dedup_ascending = [False, True, True, False]
    unique = initial_pass.sort_values(dedup_sort, ascending=dedup_ascending, na_position="last")
    unique = unique.drop_duplicates("sequence_clean", keep="first").reset_index(drop=True)

    summary_by_generation = (
        all_df.groupby("generation_source")
        .agg(
            total_candidates=("candidate_id", "count"),
            pass_initial_G1_criteria=("passes_initial_G1_criteria", "sum"),
            mean_length=("criteria_length", "mean"),
            mean_charge=("criteria_charge", "mean"),
            mean_hydrophobic_fraction=("criteria_hydrophobic_fraction", "mean"),
        )
        .reset_index()
    )
    summary_by_generation["fraction_pass_initial_G1_criteria"] = (
        summary_by_generation["pass_initial_G1_criteria"] / summary_by_generation["total_candidates"]
    )

    sort_main = [
        "v4b_elite_composite_score",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
    ]
    asc_main = [False, True, True, False]

    elite = add_bucket(select_head(unique, args.n_elite, sort_main, asc_main), "apex_elite")

    broad_pool = select_head(
        unique,
        min(args.pareto_pool, len(unique)),
        ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64"],
        [True, True, True, False],
    )
    pareto_broad = pareto_front(
        broad_pool,
        ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64"],
        maximize_cols={"organisms_MIC_le_64"},
    )
    pareto_broad = add_bucket(
        pareto_broad.sort_values(sort_main, ascending=asc_main, na_position="last"),
        "pareto_broad_initial_pass",
    )

    narrow_pool = unique.loc[
        unique["criteria_hydrophobic_fraction"].between(0.45, 0.60, inclusive="both")
        & unique["criteria_charge"].between(4, 8, inclusive="both")
        & unique["criteria_length"].between(12, 25, inclusive="both")
        & (to_numeric(unique, "APEX_median_MIC", 9999.0) <= 16.0)
        & (to_numeric(unique, "organisms_MIC_le_64", 0.0) >= 28.0)
    ].copy()
    narrow_pool = select_head(
        narrow_pool,
        min(args.pareto_pool, len(narrow_pool)),
        ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64", "score_hydro_balance"],
        [True, True, True, False, False],
    )
    pareto_narrow = pareto_front(
        narrow_pool,
        ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64", "score_hydro_balance"],
        maximize_cols={"organisms_MIC_le_64", "score_hydro_balance"},
    )
    pareto_narrow = add_bucket(
        pareto_narrow.sort_values(sort_main, ascending=asc_main, na_position="last"),
        "pareto_narrow_developability",
    )

    hydro_balanced_pool = unique.loc[
        unique["criteria_hydrophobic_fraction"].between(0.45, 0.60, inclusive="both")
        & unique["criteria_charge"].between(4, 8, inclusive="both")
        & unique["criteria_length"].between(12, 25, inclusive="both")
    ].copy()
    hydro_balanced = add_bucket(
        select_head(hydro_balanced_pool, args.n_hydro_balanced, sort_main, asc_main),
        "hydrophobicity_balanced",
    )

    broad_spectrum_pool = unique.loc[to_numeric(unique, "organisms_MIC_le_64", 0.0) >= 30.0].copy()
    broad_spectrum = add_bucket(
        select_head(
            broad_spectrum_pool,
            args.n_broad_spectrum,
            ["organisms_MIC_le_64", "APEX_median_MIC", "APEX_worst_MIC", "score_hydro_balance"],
            [False, True, True, False],
        ),
        "broad_spectrum_MIC_le_64",
    )

    worst_case = add_bucket(
        select_head(
            unique,
            args.n_worst_case,
            ["APEX_worst_MIC", "APEX_median_MIC", "organisms_MIC_le_64", "score_hydro_balance"],
            [True, True, False, False],
        ),
        "worst_case_robust",
    )

    rescue_pool = unique.loc[
        unique["criteria_hydrophobic_fraction"].between(0.60, 0.70, inclusive="both")
        & (to_numeric(unique, "APEX_median_MIC", 9999.0) <= 10.0)
        & (to_numeric(unique, "organisms_MIC_le_64", 0.0) >= 28.0)
    ].copy()
    potency_rescue = add_bucket(
        select_head(
            rescue_pool,
            args.n_potency_rescue,
            ["APEX_median_MIC", "APEX_worst_MIC", "criteria_hydrophobic_fraction"],
            [True, True, True],
        ),
        "potency_rescue_high_hydro",
    )

    generation_diverse_pieces = []
    for g, group in unique.groupby("generation_source"):
        part = select_head(group, args.n_generation_diverse_per_generation, sort_main, asc_main)
        generation_diverse_pieces.append(part)
    generation_diverse = pd.concat(generation_diverse_pieces, ignore_index=True, sort=False) if generation_diverse_pieces else unique.head(0)
    generation_diverse = add_bucket(generation_diverse, "generation_diverse")

    buckets = {
        "apex_elite": elite,
        "pareto_broad_initial_pass": pareto_broad,
        "pareto_narrow_developability": pareto_narrow,
        "hydrophobicity_balanced": hydro_balanced,
        "broad_spectrum_MIC_le_64": broad_spectrum,
        "worst_case_robust": worst_case,
        "potency_rescue_high_hydro": potency_rescue,
        "generation_diverse": generation_diverse,
    }

    for name, table in buckets.items():
        table.to_csv(output_dir / f"v4b_selection_{name}.csv", index=False)

    stacked = pd.concat([t for t in buckets.values() if not t.empty], ignore_index=True, sort=False)
    if stacked.empty:
        union = unique.head(0).copy()
    else:
        bucket_labels = stacked.groupby("sequence_clean")["selection_bucket"].apply(lambda s: ";".join(sorted(set(s)))).reset_index()
        union = stacked.sort_values(sort_main, ascending=asc_main, na_position="last")
        union = union.drop_duplicates("sequence_clean", keep="first").merge(bucket_labels, on="sequence_clean", how="left")
        union = union.rename(columns={"selection_bucket_y": "selection_bucket_union"})
        if "selection_bucket_x" in union.columns:
            union = union.drop(columns=["selection_bucket_x"])
        union = union.sort_values(sort_main, ascending=asc_main, na_position="last").reset_index(drop=True)

    final_preview = union.head(args.n_final_preview).copy()

    all_df.to_csv(output_dir / "v4b_all_G01_to_G10_with_initial_flags.csv", index=False)
    initial_pass.to_csv(output_dir / "v4b_all_candidates_passing_initial_G1_criteria.csv", index=False)
    unique.to_csv(output_dir / "v4b_unique_sequences_passing_initial_G1_criteria.csv", index=False)
    summary_by_generation.to_csv(output_dir / "v4b_initial_G1_criteria_summary_by_generation.csv", index=False)
    union.to_csv(output_dir / "v4b_elite_pareto_bucket_union.csv", index=False)
    final_preview.to_csv(output_dir / f"v4b_final_preview_top_{args.n_final_preview}.csv", index=False)
    write_fasta(union, output_dir / "v4b_elite_pareto_bucket_union.fasta")
    write_fasta(final_preview, output_dir / f"v4b_final_preview_top_{args.n_final_preview}.fasta")

    hydro_counts = initial_pass["hydrophobicity_zone"].value_counts(dropna=False).rename_axis("hydrophobicity_zone").reset_index(name="count")
    hydro_counts.to_csv(output_dir / "v4b_initial_pass_hydrophobicity_zone_counts.csv", index=False)

    bucket_summary_rows = []
    for name, table in buckets.items():
        row = {
            "bucket": name,
            "rows": int(len(table)),
            "unique_sequences": int(table["sequence_clean"].nunique()) if len(table) else 0,
            "best_median_MIC": float(pd.to_numeric(table.get("APEX_median_MIC", pd.Series(dtype=float)), errors="coerce").min()) if len(table) and "APEX_median_MIC" in table else None,
            "median_of_median_MIC": float(pd.to_numeric(table.get("APEX_median_MIC", pd.Series(dtype=float)), errors="coerce").median()) if len(table) and "APEX_median_MIC" in table else None,
            "mean_hydrophobic_fraction": float(table["criteria_hydrophobic_fraction"].mean()) if len(table) else None,
            "mean_charge": float(table["criteria_charge"].mean()) if len(table) else None,
            "mean_length": float(table["criteria_length"].mean()) if len(table) else None,
        }
        bucket_summary_rows.append(row)
    bucket_summary = pd.DataFrame(bucket_summary_rows)
    bucket_summary.to_csv(output_dir / "v4b_elite_pareto_bucket_summary.csv", index=False)

    payload = {
        "total_scored_candidates": int(len(all_df)),
        "total_passing_initial_G1_criteria": int(len(initial_pass)),
        "unique_sequences_passing_initial_G1_criteria": int(len(unique)),
        "hydro_target": float(args.hydro_target),
        "pareto_pool": int(args.pareto_pool),
        "bucket_counts": {k: int(len(v)) for k, v in buckets.items()},
        "bucket_unique_sequence_counts": {k: int(v["sequence_clean"].nunique()) if len(v) else 0 for k, v in buckets.items()},
        "union_unique_sequences": int(len(union)),
        "final_preview_count": int(len(final_preview)),
        "outputs": {
            "all_with_flags": str(output_dir / "v4b_all_G01_to_G10_with_initial_flags.csv"),
            "all_initial_pass": str(output_dir / "v4b_all_candidates_passing_initial_G1_criteria.csv"),
            "unique_initial_pass": str(output_dir / "v4b_unique_sequences_passing_initial_G1_criteria.csv"),
            "bucket_union": str(output_dir / "v4b_elite_pareto_bucket_union.csv"),
            "bucket_union_fasta": str(output_dir / "v4b_elite_pareto_bucket_union.fasta"),
            "final_preview": str(output_dir / f"v4b_final_preview_top_{args.n_final_preview}.csv"),
            "final_preview_fasta": str(output_dir / f"v4b_final_preview_top_{args.n_final_preview}.fasta"),
        },
    }
    (output_dir / "v4b_elite_pareto_selection_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\nINITIAL G1 CRITERIA SUMMARY BY GENERATION")
    print(summary_by_generation.round(4).to_string(index=False))
    print("\nHYDROPHOBICITY ZONES AMONG INITIAL-PASS CANDIDATES")
    print(hydro_counts.to_string(index=False))
    print("\nSELECTION BUCKET SUMMARY")
    print(bucket_summary.round(3).to_string(index=False))
    print("\nTOTALS")
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nTop preview saved: {output_dir / f'v4b_final_preview_top_{args.n_final_preview}.csv'}")


if __name__ == "__main__":
    main()

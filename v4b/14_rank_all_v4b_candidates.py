#!/usr/bin/env python3
"""Create a master ranking of all V4B AMP-JEPA candidates.

This script is intentionally not a panel builder and not a challenge-submission
builder. It consolidates all scored generations, validates the original V4B/G1
candidate criteria, deduplicates by sequence, and ranks every candidate using
APEX potency, breadth, worst-case robustness, and physicochemical balance.

Outputs include complete ranked tables plus top-N ranked FASTA/CSV exports.
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


def numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return s.fillna(float(s.median()))


def minmax(values: Iterable[float], lower_is_better: bool) -> np.ndarray:
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


def balance(values: Iterable[float], target: float, half_width: float) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    x = np.nan_to_num(x, nan=target, posinf=target, neginf=target)
    return np.clip(1.0 - np.abs(x - target) / max(half_width, 1e-9), 0.0, 1.0).astype(np.float32)


def add_master_scores(df: pd.DataFrame, hydro_target: float) -> pd.DataFrame:
    out = df.copy()

    median_mic = numeric(out, "APEX_median_MIC", 9999.0)
    worst_mic = numeric(out, "APEX_worst_MIC", 9999.0)
    mean_mic = numeric(out, "APEX_mean_MIC", 9999.0)
    breadth = numeric(out, "organisms_MIC_le_64", 0.0)

    out["score_apex_median"] = minmax(median_mic, lower_is_better=True)
    out["score_apex_worst"] = minmax(worst_mic, lower_is_better=True)
    out["score_apex_mean"] = minmax(mean_mic, lower_is_better=True)
    out["score_breadth"] = minmax(breadth, lower_is_better=False)
    out["score_hydro_balance"] = balance(out["criteria_hydrophobic_fraction"], hydro_target, 0.18)
    out["score_charge_balance"] = balance(out["criteria_charge"], 5.5, 4.0)
    out["score_length_balance"] = balance(out["criteria_length"], 17.0, 12.0)

    # Main rank: useful before toxicity/hemolysis predictors are added.
    # Lower MIC is important, but high breadth, lower worst-case MIC, and sane
    # hydro/charge/length prevent collapse to very hydrophobic edge cases.
    out["ampjepa_master_score"] = (
        0.32 * out["score_apex_median"]
        + 0.22 * out["score_apex_worst"]
        + 0.12 * out["score_apex_mean"]
        + 0.16 * out["score_breadth"]
        + 0.10 * out["score_hydro_balance"]
        + 0.04 * out["score_charge_balance"]
        + 0.04 * out["score_length_balance"]
    )

    # Alternate ranks for interpretation, not final panel assignment.
    out["apex_potency_score"] = (
        0.55 * out["score_apex_median"]
        + 0.25 * out["score_apex_mean"]
        + 0.20 * out["score_apex_worst"]
    )
    out["apex_breadth_robustness_score"] = (
        0.35 * out["score_breadth"]
        + 0.35 * out["score_apex_worst"]
        + 0.20 * out["score_apex_median"]
        + 0.10 * out["score_hydro_balance"]
    )
    out["developability_balance_score"] = (
        0.45 * out["score_hydro_balance"]
        + 0.30 * out["score_charge_balance"]
        + 0.25 * out["score_length_balance"]
    )

    hydro = out["criteria_hydrophobic_fraction"]
    out["hydrophobicity_zone"] = pd.cut(
        hydro,
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

    out["ranking_note"] = "standard"
    out.loc[hydro.between(0.45, 0.60, inclusive="both"), "ranking_note"] = "hydrophobicity_preferred"
    out.loc[hydro.between(0.60, 0.65, inclusive="right"), "ranking_note"] = "hydrophobicity_caution"
    out.loc[hydro.gt(0.65), "ranking_note"] = "hydrophobicity_high_risk"

    return out


def write_fasta(df: pd.DataFrame, path: Path, rank_col: str = "ampjepa_master_rank") -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            cid = row.get("candidate_id", "candidate")
            seq = clean_sequence(row.get("sequence", ""))
            rank = row.get(rank_col, "NA")
            score = row.get("ampjepa_master_score", "NA")
            med = row.get("APEX_median_MIC", "NA")
            worst = row.get("APEX_worst_MIC", "NA")
            org = row.get("organisms_MIC_le_64", "NA")
            hydro = row.get("criteria_hydrophobic_fraction", "NA")
            charge = row.get("criteria_charge", "NA")
            gen = row.get("generation_source", row.get("generation", "NA"))
            handle.write(
                f">rank={rank}|{cid}|G={gen}|score={score}|median_MIC={med}|worst_MIC={worst}|org_le64={org}|hydro={hydro}|charge={charge}\n"
            )
            handle.write(seq + "\n")


def export_top(df: pd.DataFrame, outdir: Path, n: int) -> None:
    top = df.head(min(n, len(df))).copy()
    top.to_csv(outdir / f"ampjepa_v4b_top{n}_master_ranked.csv", index=False)
    write_fasta(top, outdir / f"ampjepa_v4b_top{n}_master_ranked.fasta")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="v4b/results")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--end-generation", type=int, default=10)
    parser.add_argument("--output-dir", default="v4b/results/master_ranking")
    parser.add_argument("--hydro-target", type=float, default=0.52)
    parser.add_argument("--top-n", type=int, nargs="+", default=[100, 500, 1000, 5000, 10000])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pieces: list[pd.DataFrame] = []
    for g in range(args.start_generation, args.end_generation + 1):
        path = results_dir / f"generation_{g:02d}" / f"generation_{g:02d}_candidates_scored.csv"
        if not path.exists():
            print(f"[WARN] Missing scored generation file: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["generation_source"] = g
        pieces.append(df)

    if not pieces:
        raise SystemExit("No scored generation files found.")

    all_df = pd.concat(pieces, ignore_index=True, sort=False)
    all_df["sequence_clean"] = all_df["sequence"].map(clean_sequence)

    feats = all_df["sequence_clean"].map(sequence_features)
    all_df["criteria_length"] = [x[0] for x in feats]
    all_df["criteria_charge"] = [x[1] for x in feats]
    all_df["criteria_hydrophobic_fraction"] = [x[2] for x in feats]
    all_df["criteria_canonical_aa"] = [x[3] for x in feats]

    all_df["passes_initial_G1_criteria"] = (
        all_df["criteria_canonical_aa"]
        & all_df["criteria_length"].between(10, 40, inclusive="both")
        & all_df["criteria_charge"].between(2, 12, inclusive="both")
        & all_df["criteria_hydrophobic_fraction"].between(0.20, 0.70, inclusive="both")
    )

    passing = all_df.loc[all_df["passes_initial_G1_criteria"]].copy()
    passing = add_master_scores(passing, hydro_target=args.hydro_target)

    # Rank all passing candidates, then keep best copy of each sequence.
    sort_cols = [
        "ampjepa_master_score",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "score_hydro_balance",
    ]
    ascending = [False, True, True, False, False]
    ranked_all = passing.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    ranked_all.insert(0, "ampjepa_all_candidate_rank", np.arange(1, len(ranked_all) + 1))

    ranked_unique = ranked_all.drop_duplicates("sequence_clean", keep="first").reset_index(drop=True)
    ranked_unique.insert(0, "ampjepa_master_rank", np.arange(1, len(ranked_unique) + 1))

    # Extra interpretable ranks.
    ranked_unique["rank_apex_potency_only"] = ranked_unique["apex_potency_score"].rank(ascending=False, method="first").astype(int)
    ranked_unique["rank_breadth_robustness"] = ranked_unique["apex_breadth_robustness_score"].rank(ascending=False, method="first").astype(int)
    ranked_unique["rank_developability_balance"] = ranked_unique["developability_balance_score"].rank(ascending=False, method="first").astype(int)

    ranked_all.to_csv(outdir / "ampjepa_v4b_all_100k_ranked_with_duplicates.csv", index=False)
    ranked_unique.to_csv(outdir / "ampjepa_v4b_all_unique_master_ranked.csv", index=False)
    write_fasta(ranked_unique, outdir / "ampjepa_v4b_all_unique_master_ranked.fasta")

    for n in args.top_n:
        export_top(ranked_unique, outdir, int(n))

    generation_summary = (
        ranked_unique.groupby("generation_source")
        .agg(
            unique_ranked_candidates=("candidate_id", "count"),
            best_master_rank=("ampjepa_master_rank", "min"),
            median_master_rank=("ampjepa_master_rank", "median"),
            best_median_MIC=("APEX_median_MIC", "min"),
            median_of_median_MIC=("APEX_median_MIC", "median"),
            mean_hydrophobic_fraction=("criteria_hydrophobic_fraction", "mean"),
            mean_charge=("criteria_charge", "mean"),
            mean_length=("criteria_length", "mean"),
        )
        .reset_index()
    )
    generation_summary.to_csv(outdir / "ampjepa_v4b_master_ranking_summary_by_generation.csv", index=False)

    hydro_summary = (
        ranked_unique["hydrophobicity_zone"]
        .value_counts(dropna=False)
        .rename_axis("hydrophobicity_zone")
        .reset_index(name="count")
    )
    hydro_summary.to_csv(outdir / "ampjepa_v4b_master_ranking_hydrophobicity_zones.csv", index=False)

    top100 = ranked_unique.head(100)
    top100_summary = {
        "n": int(len(top100)),
        "best_median_MIC": float(pd.to_numeric(top100["APEX_median_MIC"], errors="coerce").min()),
        "median_of_median_MIC": float(pd.to_numeric(top100["APEX_median_MIC"], errors="coerce").median()),
        "best_worst_MIC": float(pd.to_numeric(top100["APEX_worst_MIC"], errors="coerce").min()),
        "median_worst_MIC": float(pd.to_numeric(top100["APEX_worst_MIC"], errors="coerce").median()),
        "mean_hydrophobic_fraction": float(top100["criteria_hydrophobic_fraction"].mean()),
        "mean_charge": float(top100["criteria_charge"].mean()),
        "mean_length": float(top100["criteria_length"].mean()),
        "generation_counts": {str(k): int(v) for k, v in top100["generation_source"].value_counts().sort_index().items()},
        "hydrophobicity_zone_counts": {str(k): int(v) for k, v in top100["hydrophobicity_zone"].value_counts().items()},
    }

    payload = {
        "total_loaded_candidates": int(len(all_df)),
        "passing_initial_G1_criteria": int(len(passing)),
        "unique_sequences_ranked": int(len(ranked_unique)),
        "hydro_target": float(args.hydro_target),
        "score_formula": {
            "ampjepa_master_score": {
                "score_apex_median": 0.32,
                "score_apex_worst": 0.22,
                "score_apex_mean": 0.12,
                "score_breadth": 0.16,
                "score_hydro_balance": 0.10,
                "score_charge_balance": 0.04,
                "score_length_balance": 0.04,
            }
        },
        "top100_summary": top100_summary,
        "outputs": {
            "all_ranked_with_duplicates": str(outdir / "ampjepa_v4b_all_100k_ranked_with_duplicates.csv"),
            "all_unique_master_ranked": str(outdir / "ampjepa_v4b_all_unique_master_ranked.csv"),
            "all_unique_fasta": str(outdir / "ampjepa_v4b_all_unique_master_ranked.fasta"),
            "top100_csv": str(outdir / "ampjepa_v4b_top100_master_ranked.csv"),
            "top100_fasta": str(outdir / "ampjepa_v4b_top100_master_ranked.fasta"),
        },
    }
    (outdir / "ampjepa_v4b_master_ranking_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lean_cols = [
        "ampjepa_master_rank",
        "candidate_id",
        "generation_source",
        "sequence",
        "criteria_length",
        "criteria_charge",
        "criteria_hydrophobic_fraction",
        "hydrophobicity_zone",
        "APEX_mean_MIC",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "ampjepa_master_score",
        "apex_potency_score",
        "apex_breadth_robustness_score",
        "developability_balance_score",
        "rank_apex_potency_only",
        "rank_breadth_robustness",
        "rank_developability_balance",
        "ranking_note",
    ]
    lean_cols = [c for c in lean_cols if c in ranked_unique.columns]
    ranked_unique[lean_cols].to_csv(outdir / "ampjepa_v4b_all_unique_master_ranked_lean.csv", index=False)
    ranked_unique.head(100)[lean_cols].to_csv(outdir / "ampjepa_v4b_top100_master_ranked_lean.csv", index=False)

    print("\nMASTER RANKING SUMMARY")
    print(json.dumps(payload, indent=2, default=str))
    print("\nSUMMARY BY GENERATION")
    print(generation_summary.round(3).to_string(index=False))
    print("\nTOP 25 MASTER-RANKED CANDIDATES")
    print(ranked_unique[lean_cols].head(25).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

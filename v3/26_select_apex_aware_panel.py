#!/usr/bin/env python3
"""Select an APEX-aware final panel from MIC-scored v3 candidates.

This step is meant to run after v3/25_score_v3_candidates_with_apex.py.
It converts raw APEX MIC predictions into a decision-oriented ranking that
balances predicted potency, breadth, worst-case weakness, novelty, and simple
sequence developability constraints.

The output is not experimental validation. It is a computational triage layer
for choosing which v3 candidates deserve deeper analysis or wet-lab review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWYC")
AROMATIC = set("FWY")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def clean_sequence(value: object) -> str:
    sequence = str(value).strip().upper()
    return "".join(residue for residue in sequence if residue in AMINO_ACIDS)


def residue_fraction(sequence: str, residues: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(residue in residues for residue in sequence) / len(sequence)


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sequence"] = out["sequence"].map(clean_sequence)
    out["length"] = safe_numeric(out, "length", np.nan).fillna(out["sequence"].str.len())

    if "net_charge_KR_minus_DE" not in out.columns:
        out["net_charge_KR_minus_DE"] = out["sequence"].map(
            lambda s: s.count("K") + s.count("R") - s.count("D") - s.count("E")
        )

    if "hydrophobic_fraction" not in out.columns:
        out["hydrophobic_fraction"] = out["sequence"].map(lambda s: residue_fraction(s, HYDROPHOBIC))

    out["cysteine_count"] = out["sequence"].str.count("C")
    out["tryptophan_count"] = out["sequence"].str.count("W")
    out["aromatic_fraction"] = out["sequence"].map(lambda s: residue_fraction(s, AROMATIC))
    out["positive_charge_density"] = (
        safe_numeric(out, "net_charge_KR_minus_DE", 0.0).clip(lower=0)
        / safe_numeric(out, "length", 1.0).replace(0, np.nan)
    )

    return out


def normalize_inverse(values: pd.Series, high_value: float) -> pd.Series:
    """Return high score for low numeric values using a linear cap."""
    numeric = pd.to_numeric(values, errors="coerce")
    return (1.0 - (numeric / high_value)).clip(lower=0.0, upper=1.0).fillna(0.0)


def normalize_direct(values: pd.Series, high_value: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric / max(high_value, 1e-8)).clip(lower=0.0, upper=1.0).fillna(0.0)


def add_apex_aware_scores(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = add_sequence_features(df)

    median_mic = safe_numeric(out, "APEX_median_MIC")
    mean_mic = safe_numeric(out, "APEX_mean_MIC")
    worst_mic = safe_numeric(out, "APEX_worst_MIC")
    best_mic = safe_numeric(out, "APEX_best_MIC")
    hit64 = safe_numeric(out, "organisms_MIC_le_64", 0.0)
    hit32 = safe_numeric(out, "organisms_MIC_le_32", 0.0)
    v3_score = safe_numeric(out, "v3_rank_score", 0.0)
    novelty = safe_numeric(out, "novelty_score", 0.0)

    max_hit64 = max(float(hit64.max(skipna=True) or 1.0), 1.0)
    max_v3_score = max(float(v3_score.max(skipna=True) or 1.0), 1.0)

    out["apex_potency_component"] = normalize_inverse(median_mic, args.median_scale)
    out["apex_mean_component"] = normalize_inverse(mean_mic, args.mean_scale)
    out["apex_worst_component"] = normalize_inverse(np.log10(worst_mic.clip(lower=0) + 1.0), np.log10(args.worst_scale + 1.0))
    out["apex_hit64_component"] = normalize_direct(hit64, max_hit64)
    out["apex_hit32_component"] = normalize_direct(hit32, max_hit64)
    out["v3_internal_component"] = normalize_direct(v3_score, max_v3_score)
    out["novelty_component"] = novelty.clip(lower=0.0, upper=1.0).fillna(0.0)

    charge = safe_numeric(out, "net_charge_KR_minus_DE", 0.0)
    hydro = safe_numeric(out, "hydrophobic_fraction", 0.0)
    length = safe_numeric(out, "length", 0.0)
    cys = safe_numeric(out, "cysteine_count", 0.0)
    trp = safe_numeric(out, "tryptophan_count", 0.0)
    aromatic = safe_numeric(out, "aromatic_fraction", 0.0)

    out["length_ok"] = length.between(args.min_length, args.max_length)
    out["charge_ok"] = charge.between(args.min_charge, args.max_charge)
    out["hydrophobicity_ok"] = hydro.between(args.min_hydrophobic_fraction, args.max_hydrophobic_fraction)
    out["cysteine_ok"] = cys <= args.max_cysteines
    out["tryptophan_ok"] = trp <= args.max_tryptophans
    out["aromatic_ok"] = aromatic <= args.max_aromatic_fraction
    out["mic_median_ok"] = median_mic <= args.max_median_mic
    out["hit64_ok"] = hit64 >= args.min_organisms_mic_le_64
    out["worst_mic_ok"] = worst_mic <= args.max_worst_mic

    out["developability_penalty"] = 0.0
    out.loc[~out["length_ok"], "developability_penalty"] += 0.12
    out.loc[~out["charge_ok"], "developability_penalty"] += 0.12
    out.loc[~out["hydrophobicity_ok"], "developability_penalty"] += 0.12
    out.loc[~out["cysteine_ok"], "developability_penalty"] += 0.10
    out.loc[~out["tryptophan_ok"], "developability_penalty"] += 0.08
    out.loc[~out["aromatic_ok"], "developability_penalty"] += 0.08

    out["apex_aware_score"] = (
        0.30 * out["apex_potency_component"]
        + 0.22 * out["apex_hit64_component"]
        + 0.14 * out["apex_worst_component"]
        + 0.12 * out["apex_mean_component"]
        + 0.08 * out["apex_hit32_component"]
        + 0.07 * out["v3_internal_component"]
        + 0.07 * out["novelty_component"]
        - out["developability_penalty"]
    )

    out["passes_apex_aware_filters"] = (
        out["mic_median_ok"]
        & out["hit64_ok"]
        & out["worst_mic_ok"]
        & out["length_ok"]
        & out["charge_ok"]
        & out["hydrophobicity_ok"]
        & out["cysteine_ok"]
        & out["tryptophan_ok"]
        & out["aromatic_ok"]
    )

    out = out.sort_values(
        by=[
            "passes_apex_aware_filters",
            "apex_aware_score",
            "APEX_median_MIC",
            "APEX_worst_MIC",
            "APEX_mean_MIC",
        ],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    if "apex_aware_rank" in out.columns:
        out = out.drop(columns=["apex_aware_rank"])
    out.insert(0, "apex_aware_rank", range(1, len(out) + 1))

    return out


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            handle.write(
                f">{row.get('apex_candidate_id', row.get('candidate_id', 'v3_candidate'))}"
                f"|apex_aware_rank={int(row['apex_aware_rank'])}"
                f"|median_MIC={float(row.get('APEX_median_MIC', np.nan)):.2f}"
                f"|mean_MIC={float(row.get('APEX_mean_MIC', np.nan)):.2f}"
                f"|worst_MIC={float(row.get('APEX_worst_MIC', np.nan)):.2f}\n"
            )
            handle.write(f"{row['sequence']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-candidates", default="v3/results/apex_scored_v3/apex_scored_v3_candidates.csv")
    parser.add_argument("--output-dir", default="v3/results/apex_scored_v3")
    parser.add_argument("--top-n", type=int, default=20)

    parser.add_argument("--max-median-mic", type=float, default=80.0)
    parser.add_argument("--max-worst-mic", type=float, default=1200.0)
    parser.add_argument("--min-organisms-mic-le-64", type=float, default=10.0)

    parser.add_argument("--min-length", type=float, default=12.0)
    parser.add_argument("--max-length", type=float, default=32.0)
    parser.add_argument("--min-charge", type=float, default=3.0)
    parser.add_argument("--max-charge", type=float, default=9.0)
    parser.add_argument("--min-hydrophobic-fraction", type=float, default=0.30)
    parser.add_argument("--max-hydrophobic-fraction", type=float, default=0.55)
    parser.add_argument("--max-cysteines", type=float, default=2.0)
    parser.add_argument("--max-tryptophans", type=float, default=3.0)
    parser.add_argument("--max-aromatic-fraction", type=float, default=0.25)

    parser.add_argument("--median-scale", type=float, default=128.0)
    parser.add_argument("--mean-scale", type=float, default=256.0)
    parser.add_argument("--worst-scale", type=float, default=2048.0)
    args = parser.parse_args()

    scored_path = resolve_path(args.scored_candidates)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scored_path.exists():
        raise FileNotFoundError(f"MIC-scored candidate table not found: {scored_path}")

    df = pd.read_csv(scored_path)
    if "sequence" not in df.columns:
        raise ValueError("Input table must contain a sequence column.")
    if "APEX_median_MIC" not in df.columns:
        raise ValueError("Input table must contain APEX_median_MIC. Run v3/25_score_v3_candidates_with_apex.py first.")

    ranked = add_apex_aware_scores(df, args)
    top_panel = ranked.head(args.top_n).copy()

    ranked_path = output_dir / "apex_aware_ranked_v3.csv"
    top_path = output_dir / "apex_aware_top_panel_v3.csv"
    fasta_path = output_dir / "apex_aware_top_panel_v3.fasta"
    summary_path = output_dir / "apex_aware_selection_summary.json"

    ranked.to_csv(ranked_path, index=False)
    top_panel.to_csv(top_path, index=False)
    write_fasta(top_panel, fasta_path)

    summary = {
        "input_table": str(scored_path),
        "candidate_count": int(len(ranked)),
        "top_n": int(len(top_panel)),
        "passing_apex_aware_filters": int(ranked["passes_apex_aware_filters"].sum()),
        "best_median_MIC": float(ranked["APEX_median_MIC"].min()),
        "best_mean_MIC": float(ranked["APEX_mean_MIC"].min()),
        "best_worst_MIC": float(ranked["APEX_worst_MIC"].min()),
        "best_apex_aware_sequence": str(ranked.iloc[0]["sequence"]),
        "filter_parameters": {
            "max_median_mic": args.max_median_mic,
            "max_worst_mic": args.max_worst_mic,
            "min_organisms_mic_le_64": args.min_organisms_mic_le_64,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "min_charge": args.min_charge,
            "max_charge": args.max_charge,
            "min_hydrophobic_fraction": args.min_hydrophobic_fraction,
            "max_hydrophobic_fraction": args.max_hydrophobic_fraction,
            "max_cysteines": args.max_cysteines,
            "max_tryptophans": args.max_tryptophans,
            "max_aromatic_fraction": args.max_aromatic_fraction,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    display_cols = [
        "apex_aware_rank",
        "APEX_rank",
        "candidate_id",
        "sequence",
        "length",
        "net_charge_KR_minus_DE",
        "hydrophobic_fraction",
        "cysteine_count",
        "tryptophan_count",
        "APEX_mean_MIC",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "apex_aware_score",
        "passes_apex_aware_filters",
    ]
    display_cols = [column for column in display_cols if column in ranked.columns]

    print("\n" + "=" * 130)
    print("APEX-AWARE V3 FINAL PANEL")
    print("=" * 130)
    print(ranked[display_cols].head(args.top_n).round(3).to_string(index=False))

    print("\nOutput files:")
    print(f"  {ranked_path}")
    print(f"  {top_path}")
    print(f"  {fasta_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()

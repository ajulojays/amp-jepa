#!/usr/bin/env python3
"""Analyze one-digit predicted MICs against Salmonella enterica strains.

The script filters the final nonredundant AMP-EvoDesign portfolio for peptides
with a predicted MIC < 10 µM against at least one Salmonella enterica model,
exports strain-level and overlap summaries, and creates both a complete heatmap
and a presentation-friendly top-N heatmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Salmonella enterica one-digit MIC summaries and heatmaps."
    )
    parser.add_argument(
        "--input",
        default=(
            "v4b/results/novelty_nonredundant_mic32/"
            "v4b_novel_MIC32_self_nonredundant_lt75.csv"
        ),
        help="Final candidate CSV containing organism-specific APEX MIC columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="v4b/results/salmonella_one_digit_mic",
        help="Directory for summary tables and figures.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Exclusive MIC cutoff. Default: MIC < 10 µM.",
    )
    parser.add_argument(
        "--heatmap-top-n",
        type=int,
        default=100,
        help="Number of best peptides in the readable heatmap.",
    )
    return parser.parse_args()


def detect_salmonella_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.columns
        if "salmonella enterica" in str(column).lower()
    ]
    if not columns:
        raise ValueError(
            "No Salmonella enterica MIC columns were detected in the input table."
        )
    return columns


def short_strain_name(name: str) -> str:
    replacements = {
        "Salmonella enterica ": "S. enterica\n",
        "ATCC 9150 ": "ATCC 9150\n",
        "(BEIRES NR-": "NR-",
        ")": "",
    }
    output = name
    for old, new in replacements.items():
        output = output.replace(old, new)
    return output


def save_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    annotate: bool,
) -> None:
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    log_values = np.log2(values)

    figure_width = max(7.5, matrix.shape[1] * 2.0)
    figure_height = max(5.5, min(35.0, matrix.shape[0] * 0.22 + 2.5))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(log_values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [short_strain_name(column) for column in matrix.columns],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=5 if matrix.shape[0] > 100 else 7)
    ax.set_xlabel("S. enterica strain/model")
    ax.set_ylabel("Peptide candidate")
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Predicted MIC (log2 µM)")

    if annotate and matrix.shape[0] <= 100:
        midpoint = np.nanmedian(log_values)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if log_values[row_index, column_index] > midpoint else "black",
                    )

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    if "candidate_id" not in df.columns:
        raise ValueError("Input table must contain a candidate_id column.")

    salmonella_columns = detect_salmonella_columns(df)
    for column in salmonella_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    strain_values = df[salmonella_columns]
    one_digit = strain_values.lt(args.cutoff)

    df["salmonella_best_MIC"] = strain_values.min(axis=1, skipna=True)
    df["salmonella_median_MIC"] = strain_values.median(axis=1, skipna=True)
    df["n_salmonella_strains_MIC_lt_10"] = one_digit.sum(axis=1)
    df["any_salmonella_MIC_lt_10"] = one_digit.any(axis=1)
    df["all_salmonella_MIC_lt_10"] = one_digit.all(axis=1)

    qualifying = df[df["any_salmonella_MIC_lt_10"]].copy()
    qualifying = qualifying.sort_values(
        ["n_salmonella_strains_MIC_lt_10", "salmonella_best_MIC", "salmonella_median_MIC"],
        ascending=[False, True, True],
    )

    strain_summary_rows = []
    total_candidates = int(df["candidate_id"].nunique())
    for column in salmonella_columns:
        count = int((df[column] < args.cutoff).sum())
        valid = int(df[column].notna().sum())
        strain_summary_rows.append(
            {
                "salmonella_model": column,
                "n_candidates_with_MIC_lt_10": count,
                "n_candidates_with_prediction": valid,
                "fraction_of_all_candidates": count / total_candidates,
                "percentage_of_all_candidates": 100.0 * count / total_candidates,
                "median_MIC_all_candidates": float(df[column].median()),
                "best_MIC": float(df[column].min()),
            }
        )

    strain_summary = pd.DataFrame(strain_summary_rows).sort_values(
        "n_candidates_with_MIC_lt_10", ascending=False
    )

    overlap_summary = (
        df["n_salmonella_strains_MIC_lt_10"]
        .value_counts()
        .sort_index()
        .rename_axis("n_salmonella_models_with_MIC_lt_10")
        .reset_index(name="n_peptides")
    )
    overlap_summary["percentage_of_all_candidates"] = (
        100.0 * overlap_summary["n_peptides"] / total_candidates
    )

    metadata_columns = [
        column
        for column in [
            "candidate_id",
            "sequence_clean",
            "sequence",
            "lead_tier",
            "generation_source",
            "criteria_length",
            "criteria_charge",
            "criteria_hydrophobic_fraction",
            "APEX_median_MIC",
            "APEX_mean_MIC",
            "APEX_worst_MIC",
            "salmonella_best_MIC",
            "salmonella_median_MIC",
            "n_salmonella_strains_MIC_lt_10",
            "any_salmonella_MIC_lt_10",
            "all_salmonella_MIC_lt_10",
        ]
        if column in qualifying.columns
    ]

    qualifying_output = qualifying[metadata_columns + salmonella_columns]
    qualifying_output.to_csv(
        output_dir / "salmonella_one_digit_candidates.csv", index=False
    )
    strain_summary.to_csv(
        output_dir / "salmonella_one_digit_summary_by_strain.csv", index=False
    )
    overlap_summary.to_csv(
        output_dir / "salmonella_one_digit_overlap_summary.csv", index=False
    )

    binary_matrix = qualifying.set_index("candidate_id")[salmonella_columns].lt(args.cutoff)
    binary_matrix.astype(int).to_csv(
        output_dir / "salmonella_one_digit_binary_matrix.csv"
    )

    full_matrix = qualifying.set_index("candidate_id")[salmonella_columns]
    full_matrix.to_csv(output_dir / "salmonella_MIC_heatmap_matrix_all.csv")

    top_n = min(args.heatmap_top_n, len(qualifying))
    top_matrix = full_matrix.head(top_n)
    top_matrix.to_csv(output_dir / f"salmonella_MIC_heatmap_matrix_top{top_n}.csv")

    save_heatmap(
        full_matrix,
        output_dir / "01_salmonella_MIC_heatmap_all_one_digit_candidates.png",
        (
            f"Peptides with predicted MIC < {args.cutoff:g} µM against at least one "
            f"S. enterica model (n={len(qualifying)})"
        ),
        annotate=False,
    )

    save_heatmap(
        top_matrix,
        output_dir / f"02_salmonella_MIC_heatmap_top{top_n}.png",
        (
            f"Top {top_n} peptides by one-digit S. enterica coverage and potency"
        ),
        annotate=top_n <= 100,
    )

    summary = {
        "input_file": str(input_path),
        "mic_definition": f"predicted MIC < {args.cutoff:g} µM",
        "n_total_candidates": total_candidates,
        "n_salmonella_models": len(salmonella_columns),
        "salmonella_models": salmonella_columns,
        "n_candidates_one_digit_against_any_salmonella": int(len(qualifying)),
        "percentage_candidates_one_digit_against_any_salmonella": (
            100.0 * len(qualifying) / total_candidates
        ),
        "n_candidates_one_digit_against_all_salmonella": int(
            df["all_salmonella_MIC_lt_10"].sum()
        ),
        "percentage_candidates_one_digit_against_all_salmonella": (
            100.0 * df["all_salmonella_MIC_lt_10"].sum() / total_candidates
        ),
        "best_overall_salmonella_MIC": float(strain_values.min().min()),
        "best_candidate": str(
            df.loc[df["salmonella_best_MIC"].idxmin(), "candidate_id"]
        ),
    }

    with (output_dir / "salmonella_one_digit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\nSALMONELLA ONE-DIGIT MIC ANALYSIS")
    print(json.dumps(summary, indent=2))
    print("\nBY STRAIN")
    print(strain_summary.to_string(index=False))
    print("\nOVERLAP ACROSS SALMONELLA MODELS")
    print(overlap_summary.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

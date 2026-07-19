#!/usr/bin/env python3
"""Create a species-specific one-digit predicted-MIC panel for V4C.

Species are resolved only from the 34 organism/strain variables present in the frozen
APEX matrix. A canonical full species name is preferred, while the abbreviated source
name remains accepted for compatibility. All MIC values are computational predictions.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apex_model_catalog import (
    available_species,
    infer_species_label,
    match_species_columns,
    normalize_species_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "v4c/results/final_funnel/04_self_nonredundant_cdhit75/"
            "v4c_novel_MIC32_self_nonredundant_cdhit75.csv"
        ),
    )
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument(
        "--species",
        help=(
            "Canonical species name, for example 'Escherichia coli'. "
            "Abbreviations such as 'E. coli' are also accepted."
        ),
    )
    species_group.add_argument(
        "--species-regex",
        help="Backward-compatible raw/canonical species regular expression.",
    )
    parser.add_argument("--species-label", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--heatmap-top-n", type=int, default=100)
    parser.add_argument("--label-top-n", type=int, default=100)
    parser.add_argument("--minimum-numeric-fraction", type=float, default=0.95)
    return parser.parse_args()


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "species"


def save_heatmap(
    matrix: pd.DataFrame,
    path_stem: Path,
    title: str,
    label_top_n: int,
) -> None:
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    log_values = np.log2(np.clip(values, 1e-6, None))
    n_rows, n_cols = matrix.shape
    fig_width = max(7.5, n_cols * 2.0)
    fig_height = max(6.0, min(35.0, n_rows * 0.22 + 2.5))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(log_values, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)

    if n_rows <= label_top_n:
        ticks = np.arange(n_rows)
    else:
        ticks = np.unique(np.linspace(0, n_rows - 1, label_top_n, dtype=int))
    ax.set_yticks(ticks)
    ax.set_yticklabels([matrix.index[index] for index in ticks], fontsize=6)
    ax.set_xlabel("Matched species strain/model")
    ax.set_ylabel("V4C candidate")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Predicted MIC (log2 µM)")
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, low_memory=False)
    if "candidate_id" not in df.columns:
        raise ValueError("Input must contain candidate_id.")
    if df["candidate_id"].duplicated().any():
        raise ValueError("Input contains duplicate candidate IDs.")

    query = args.species if args.species is not None else args.species_regex
    assert query is not None
    species_columns, excluded_numeric = match_species_columns(
        df,
        query,
        minimum_numeric_fraction=args.minimum_numeric_fraction,
    )
    if not species_columns:
        choices = "\n  - ".join(
            available_species(
                df,
                minimum_numeric_fraction=args.minimum_numeric_fraction,
            )
        )
        raise ValueError(
            f"No frozen APEX organism/strain variables matched species query {query!r}.\n"
            f"Available canonical species:\n  - {choices}"
        )

    for column in species_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].notna().mean() < args.minimum_numeric_fraction:
            raise ValueError(
                f"Matched column {column!r} is not sufficiently numeric to be an MIC variable."
            )

    canonical_labels = sorted({infer_species_label(column) for column in species_columns})
    if len(canonical_labels) != 1:
        raise ValueError(
            "Species query matched more than one canonical species: "
            + " | ".join(canonical_labels)
        )

    canonical_species = canonical_labels[0]
    label = args.species_label or canonical_species
    slug = safe_slug(label)

    mic = df.set_index("candidate_id")[species_columns]
    tested = mic.notna().sum(axis=1)
    below = mic.lt(args.cutoff)

    metrics = pd.DataFrame(index=mic.index)
    metrics["n_species_models_tested"] = tested
    metrics["n_species_models_MIC_lt_10"] = below.sum(axis=1)
    metrics["fraction_species_models_MIC_lt_10"] = (
        metrics["n_species_models_MIC_lt_10"] / tested.replace(0, np.nan)
    )
    metrics["best_species_MIC"] = mic.min(axis=1)
    metrics["median_species_MIC"] = mic.median(axis=1)
    metrics["worst_species_MIC"] = mic.max(axis=1)
    metrics["one_digit_any_species_model"] = metrics[
        "n_species_models_MIC_lt_10"
    ].ge(1)
    metrics["one_digit_all_species_models"] = (
        tested.gt(0) & metrics["n_species_models_MIC_lt_10"].eq(tested)
    )

    metadata_columns = [
        column for column in [
            "candidate_id", "sequence_clean", "sequence", "generation_source",
            "APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC",
            "organisms_MIC_le_64", "criteria_length", "criteria_charge",
            "criteria_hydrophobic_fraction", "v4c_self_nonredundant_rank",
        ] if column in df.columns
    ]
    candidate_summary = (
        metrics.reset_index()
        .merge(
            df[metadata_columns].drop_duplicates("candidate_id"),
            on="candidate_id",
            how="left",
        )
    )
    candidate_summary = candidate_summary.sort_values(
        [
            "n_species_models_MIC_lt_10",
            "fraction_species_models_MIC_lt_10",
            "median_species_MIC",
            "worst_species_MIC",
            "best_species_MIC",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    candidate_summary["species_panel_rank"] = np.arange(
        1, len(candidate_summary) + 1
    )

    qualifying = candidate_summary[
        candidate_summary["one_digit_any_species_model"]
    ].copy()
    qualifying_ids = qualifying["candidate_id"].tolist()
    qualifying_full = qualifying.merge(
        df[["candidate_id"] + species_columns].drop_duplicates("candidate_id"),
        on="candidate_id",
        how="left",
    )

    model_rows = []
    for column in species_columns:
        values = mic[column].dropna()
        count = int(values.lt(args.cutoff).sum())
        model_rows.append(
            {
                "species_model": column,
                "canonical_species": canonical_species,
                "n_candidates_tested": int(len(values)),
                "n_candidates_MIC_lt_10": count,
                "percentage_candidates_MIC_lt_10": (
                    100.0 * count / max(len(values), 1)
                ),
                "best_predicted_MIC": float(values.min()),
                "median_predicted_MIC": float(values.median()),
                "worst_predicted_MIC": float(values.max()),
            }
        )
    model_summary = pd.DataFrame(model_rows).sort_values(
        ["n_candidates_MIC_lt_10", "median_predicted_MIC"],
        ascending=[False, True],
    )

    overlap_rows = []
    for count_models in range(1, len(species_columns) + 1):
        count = int(metrics["n_species_models_MIC_lt_10"].ge(count_models).sum())
        overlap_rows.append(
            {
                "criterion": (
                    f"predicted MIC < {args.cutoff:g} µM against at least "
                    f"{count_models} matched model(s)"
                ),
                "n_candidates": count,
                "percentage_portfolio": 100.0 * count / len(df),
            }
        )
    overlap_rows.append(
        {
            "criterion": (
                f"predicted MIC < {args.cutoff:g} µM against all matched models"
            ),
            "n_candidates": int(metrics["one_digit_all_species_models"].sum()),
            "percentage_portfolio": (
                100.0 * metrics["one_digit_all_species_models"].sum() / len(df)
            ),
        }
    )
    overlap_summary = pd.DataFrame(overlap_rows)

    candidate_summary.to_csv(
        output_dir / f"{slug}_all_candidate_summary.csv", index=False
    )
    qualifying_full.to_csv(
        output_dir / f"{slug}_one_digit_candidates.csv", index=False
    )
    model_summary.to_csv(
        output_dir / f"{slug}_model_summary.csv", index=False
    )
    overlap_summary.to_csv(
        output_dir / f"{slug}_overlap_summary.csv", index=False
    )

    full_matrix = mic.reindex(index=qualifying_ids, columns=species_columns)
    full_matrix.to_csv(output_dir / f"{slug}_one_digit_heatmap_matrix_all.csv")
    top_n = min(args.heatmap_top_n, len(full_matrix))
    top_matrix = full_matrix.head(top_n)
    top_matrix.to_csv(
        output_dir / f"{slug}_one_digit_heatmap_matrix_top{top_n}.csv"
    )

    save_heatmap(
        full_matrix,
        output_dir / f"01_{slug}_one_digit_heatmap_all",
        (
            f"{label}: candidates with predicted MIC < {args.cutoff:g} µM "
            f"against at least one model (n={len(full_matrix)})"
        ),
        args.label_top_n,
    )
    save_heatmap(
        top_matrix,
        output_dir / f"02_{slug}_one_digit_heatmap_top{top_n}",
        f"{label}: top {top_n} candidates by one-digit coverage and predicted potency",
        args.label_top_n,
    )

    summary = {
        "schema_version": "1.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "species_specific_one_digit_predicted_MIC_panel",
        "input_file": str(input_path),
        "species_query": query,
        "canonical_species": canonical_species,
        "species_label": label,
        "matched_model_columns": species_columns,
        "excluded_numeric_nonmodel_columns": excluded_numeric,
        "n_portfolio_candidates": int(len(df)),
        "n_matched_models": int(len(species_columns)),
        "one_digit_definition": f"predicted MIC < {args.cutoff:g} µM",
        "n_candidates_one_digit_any_model": int(len(qualifying)),
        "percentage_candidates_one_digit_any_model": float(
            100.0 * len(qualifying) / len(df)
        ),
        "n_candidates_one_digit_all_models": int(
            metrics["one_digit_all_species_models"].sum()
        ),
        "best_predicted_species_MIC": float(mic.min().min()),
        "best_candidate_by_species_breadth": (
            str(qualifying.iloc[0]["candidate_id"])
            if len(qualifying) else None
        ),
        "outputs": {
            "all_candidate_summary": str(
                output_dir / f"{slug}_all_candidate_summary.csv"
            ),
            "one_digit_candidates": str(
                output_dir / f"{slug}_one_digit_candidates.csv"
            ),
            "model_summary": str(output_dir / f"{slug}_model_summary.csv"),
            "overlap_summary": str(output_dir / f"{slug}_overlap_summary.csv"),
        },
    }
    with (output_dir / f"{slug}_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\nV4C SPECIES-SPECIFIC ONE-DIGIT SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nMATCHED MODEL SUMMARY")
    print(model_summary.round(3).to_string(index=False))
    print("\nOVERLAP SUMMARY")
    print(overlap_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

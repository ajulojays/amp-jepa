#!/usr/bin/env python3
"""Build the global one-digit predicted-MIC panel for the certified V4C portfolio.

The script detects organism/strain APEX MIC columns, ranks peptides by breadth of
predicted MIC <10 µM, exports model-level summaries, infers a species inventory,
and creates a presentation-ready top-N heatmap. All MIC values remain predictions.
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

SUMMARY_PREFIXES = (
    "APEX_", "mean_", "median_", "min_", "max_", "organisms_",
    "fraction_", "n_pathogens_", "criteria_", "score_", "rank_",
    "v4c_", "audit_", "self_", "broad_", "qc_", "layer1_",
)

FIXED_METADATA = {
    "candidate_id", "apex_candidate_id", "sequence", "sequence_clean",
    "generation", "generation_source", "lead_tier", "record_type",
    "parent_candidate_id", "second_parent_candidate_id", "generation_operator",
    "parent_selection_stratum", "hydrophobicity_zone", "ranking_note",
    "length", "sequence_length", "net_charge_KR_minus_DE",
    "hydrophobic_fraction", "proposal_index", "lineage_depth",
    "latent_sigma", "decode_temperature", "APEX_rank",
    "dual_novelty_pass", "median_MIC_le_32", "self_similarity_status",
    "v4c_pre_self_similarity_rank", "v4c_self_nonredundant_rank",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=(
            "v4c/results/final_funnel/04_self_nonredundant_cdhit75/"
            "v4c_novel_MIC32_self_nonredundant_cdhit75.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        default="v4c/results/final_funnel/05_global_one_digit",
    )
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--heatmap-top-n", type=int, default=100)
    p.add_argument("--minimum-numeric-fraction", type=float, default=0.95)
    return p.parse_args()


def infer_species_label(column: str) -> str:
    matches = re.findall(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z.-]{2,})\b", column)
    if not matches:
        return "Unresolved"
    genus, species = matches[0]
    return f"{genus} {species}"


def detect_model_columns(df: pd.DataFrame, minimum_numeric_fraction: float) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        name = str(column)
        if name in FIXED_METADATA or name.startswith(SUMMARY_PREFIXES):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        fraction = float(numeric.notna().mean())
        if fraction < minimum_numeric_fraction:
            continue
        valid = numeric.dropna()
        if valid.empty or not valid.gt(0).all():
            continue
        columns.append(name)
    if not columns:
        raise ValueError("No organism/strain MIC columns were detected.")
    return columns


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

    model_columns = detect_model_columns(df, args.minimum_numeric_fraction)
    for column in model_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    mic = df.set_index("candidate_id")[model_columns]
    tested = mic.notna().sum(axis=1)
    below = mic.lt(args.cutoff)

    metrics = pd.DataFrame(index=mic.index)
    metrics["n_models_tested"] = tested
    metrics["n_models_MIC_lt_10"] = below.sum(axis=1)
    metrics["fraction_models_MIC_lt_10"] = (
        metrics["n_models_MIC_lt_10"] / tested.replace(0, np.nan)
    )
    metrics["best_model_MIC"] = mic.min(axis=1)
    metrics["median_model_MIC"] = mic.median(axis=1)
    metrics["worst_model_MIC"] = mic.max(axis=1)
    metrics["global_one_digit_any_model"] = metrics["n_models_MIC_lt_10"].ge(1)
    metrics["global_one_digit_all_models"] = (
        tested.gt(0) & metrics["n_models_MIC_lt_10"].eq(tested)
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
        .merge(df[metadata_columns].drop_duplicates("candidate_id"), on="candidate_id", how="left")
    )
    candidate_summary = candidate_summary.sort_values(
        [
            "n_models_MIC_lt_10", "fraction_models_MIC_lt_10",
            "median_model_MIC", "worst_model_MIC", "best_model_MIC",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    candidate_summary["global_one_digit_rank"] = np.arange(1, len(candidate_summary) + 1)

    qualifying = candidate_summary[candidate_summary["global_one_digit_any_model"]].copy()

    model_rows = []
    inventory_rows = []
    for column in model_columns:
        values = mic[column].dropna()
        count = int(values.lt(args.cutoff).sum())
        species = infer_species_label(column)
        model_rows.append(
            {
                "model_column": column,
                "inferred_species": species,
                "n_candidates_tested": int(len(values)),
                "n_candidates_MIC_lt_10": count,
                "percentage_candidates_MIC_lt_10": 100.0 * count / max(len(values), 1),
                "best_predicted_MIC": float(values.min()),
                "median_predicted_MIC": float(values.median()),
                "worst_predicted_MIC": float(values.max()),
            }
        )
        inventory_rows.append({"inferred_species": species, "model_column": column})

    model_summary = pd.DataFrame(model_rows).sort_values(
        ["n_candidates_MIC_lt_10", "median_predicted_MIC"],
        ascending=[False, True],
    )
    species_inventory = (
        pd.DataFrame(inventory_rows)
        .groupby("inferred_species", dropna=False)
        .agg(
            n_models=("model_column", "count"),
            model_columns=("model_column", lambda x: " | ".join(map(str, x))),
        )
        .reset_index()
        .sort_values(["inferred_species"])
    )

    candidate_summary.to_csv(output_dir / "v4c_all_candidates_global_model_summary.csv", index=False)
    qualifying.to_csv(output_dir / "v4c_global_one_digit_candidates.csv", index=False)
    model_summary.to_csv(output_dir / "v4c_apex_model_summary.csv", index=False)
    species_inventory.to_csv(output_dir / "v4c_inferred_species_inventory.csv", index=False)
    (output_dir / "v4c_detected_apex_model_columns.txt").write_text(
        "\n".join(model_columns) + "\n", encoding="utf-8"
    )

    heatmap_ids = qualifying["candidate_id"].head(args.heatmap_top_n).tolist()
    heatmap = mic.reindex(index=heatmap_ids, columns=model_columns)
    heatmap.to_csv(output_dir / f"v4c_global_one_digit_heatmap_top{len(heatmap_ids)}_matrix.csv")

    if not heatmap.empty:
        log_values = np.log2(heatmap.clip(lower=1e-6))
        fig_width = max(12, len(model_columns) * 0.55)
        fig_height = max(7, len(heatmap) * 0.18 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(log_values.values, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(model_columns)))
        ax.set_xticklabels(model_columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(heatmap_ids)))
        ax.set_yticklabels(heatmap_ids, fontsize=6)
        ax.set_xlabel("APEX organism/strain model")
        ax.set_ylabel("V4C candidate")
        ax.set_title(
            f"Top {len(heatmap_ids)} V4C candidates by breadth of predicted MIC < {args.cutoff:g} µM"
        )
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Predicted MIC (log2 µM)")
        fig.tight_layout()
        fig.savefig(output_dir / "v4c_global_one_digit_heatmap_topN.png", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / "v4c_global_one_digit_heatmap_topN.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "global_one_digit_predicted_MIC_panel",
        "input_file": str(input_path),
        "n_portfolio_candidates": int(len(df)),
        "n_apex_models_detected": int(len(model_columns)),
        "one_digit_definition": f"predicted MIC < {args.cutoff:g} µM",
        "n_candidates_one_digit_any_model": int(len(qualifying)),
        "percentage_candidates_one_digit_any_model": float(100.0 * len(qualifying) / len(df)),
        "n_candidates_one_digit_all_models": int(metrics["global_one_digit_all_models"].sum()),
        "best_predicted_MIC_any_model": float(mic.min().min()),
        "best_candidate_by_global_breadth": (
            str(qualifying.iloc[0]["candidate_id"]) if len(qualifying) else None
        ),
        "outputs": {
            "all_candidate_summary": str(output_dir / "v4c_all_candidates_global_model_summary.csv"),
            "one_digit_candidates": str(output_dir / "v4c_global_one_digit_candidates.csv"),
            "model_summary": str(output_dir / "v4c_apex_model_summary.csv"),
            "species_inventory": str(output_dir / "v4c_inferred_species_inventory.csv"),
        },
    }
    with (output_dir / "v4c_global_one_digit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nV4C GLOBAL ONE-DIGIT PREDICTED-MIC SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nTOP APEX MODELS")
    print(model_summary.head(20).round(3).to_string(index=False))
    print("\nINFERRED SPECIES INVENTORY")
    print(species_inventory.to_string(index=False))


if __name__ == "__main__":
    main()

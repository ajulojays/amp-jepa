#!/usr/bin/env python3

"""Filter APEX MIC predictions for one species and create a peptide × strain heatmap.

Example
-------
python v4b/19_filter_species_mic_heatmap.py \
  --input v4b/results/novelty_nonredundant_mic32/v4b_novel_MIC32_self_nonredundant_lt75.csv \
  --species-regex "Salmonella enterica" \
  --one-digit-cutoff 10 \
  --output-dir v4b/results/species_mic_analysis/s_enterica
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_PREFIXES = (
    "APEX_", "mean_", "median_", "min_", "max_", "organisms_",
    "fraction_", "n_pathogens_", "criteria_", "score_", "rank_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count peptides below a MIC cutoff for a selected species and create a heatmap."
    )
    parser.add_argument("--input", required=True, help="Wide candidate × pathogen MIC CSV.")
    parser.add_argument(
        "--species-regex",
        required=True,
        help='Case-insensitive regular expression matched against column names, e.g. "Salmonella enterica".',
    )
    parser.add_argument(
        "--one-digit-cutoff",
        type=float,
        default=10.0,
        help="Exclusive cutoff for one-digit MIC. Default: MIC < 10 µM.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=50,
        help="Number of top peptide IDs labelled on the heatmap; all qualifying peptides are still plotted.",
    )
    return parser.parse_args()


def is_metadata_or_summary(column: str) -> bool:
    fixed = {
        "candidate_id", "apex_candidate_id", "sequence", "sequence_clean",
        "generation", "generation_source", "lead_tier", "record_type",
        "parent_candidate_id", "second_parent_candidate_id", "generation_operator",
        "parent_selection_stratum", "hydrophobicity_zone", "ranking_note",
        "length", "sequence_length", "net_charge_KR_minus_DE",
        "hydrophobic_fraction", "proposal_index", "lineage_depth",
        "latent_sigma", "decode_temperature", "APEX_rank",
    }
    return column in fixed or column.startswith(SUMMARY_PREFIXES)


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "species"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    if "candidate_id" not in df.columns:
        raise ValueError("Input must contain a candidate_id column.")

    pattern = re.compile(args.species_regex, flags=re.IGNORECASE)
    species_columns = [
        c for c in df.columns
        if pattern.search(str(c)) and not is_metadata_or_summary(str(c))
    ]

    if not species_columns:
        raise ValueError(
            f"No organism columns matched --species-regex {args.species_regex!r}."
        )

    for column in species_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    mic = df.set_index("candidate_id")[species_columns]
    valid_counts = mic.notna().sum(axis=1)
    one_digit_mask = mic.lt(args.one_digit_cutoff)

    per_candidate = pd.DataFrame(index=mic.index)
    per_candidate["n_species_strains_tested"] = valid_counts
    per_candidate["n_strains_MIC_lt_cutoff"] = one_digit_mask.sum(axis=1)
    per_candidate["fraction_strains_MIC_lt_cutoff"] = (
        per_candidate["n_strains_MIC_lt_cutoff"] / valid_counts.replace(0, np.nan)
    )
    per_candidate["best_species_MIC"] = mic.min(axis=1)
    per_candidate["median_species_MIC"] = mic.median(axis=1)
    per_candidate["worst_species_MIC"] = mic.max(axis=1)
    per_candidate["has_one_digit_MIC_any_strain"] = (
        per_candidate["n_strains_MIC_lt_cutoff"] >= 1
    )
    per_candidate["has_one_digit_MIC_all_strains"] = (
        (valid_counts > 0)
        & (per_candidate["n_strains_MIC_lt_cutoff"] == valid_counts)
    )

    metadata_columns = [
        c for c in [
            "candidate_id", "sequence_clean", "sequence", "lead_tier",
            "generation_source", "generation", "APEX_median_MIC",
            "APEX_mean_MIC", "APEX_worst_MIC",
        ] if c in df.columns
    ]
    per_candidate = (
        per_candidate.reset_index()
        .merge(df[metadata_columns].drop_duplicates("candidate_id"), on="candidate_id", how="left")
    )

    qualifying = per_candidate[per_candidate["has_one_digit_MIC_any_strain"]].copy()
    qualifying = qualifying.sort_values(
        ["n_strains_MIC_lt_cutoff", "best_species_MIC", "median_species_MIC"],
        ascending=[False, True, True],
    )

    strain_rows = []
    for column in species_columns:
        values = mic[column].dropna()
        n_below = int((values < args.one_digit_cutoff).sum())
        strain_rows.append({
            "strain": column,
            "n_peptides_tested": int(values.size),
            "n_peptides_MIC_lt_cutoff": n_below,
            "percentage_MIC_lt_cutoff": 100.0 * n_below / values.size if values.size else np.nan,
            "best_predicted_MIC": float(values.min()) if values.size else np.nan,
            "median_predicted_MIC": float(values.median()) if values.size else np.nan,
        })
    strain_summary = pd.DataFrame(strain_rows).sort_values("median_predicted_MIC")

    overlap_rows = []
    for k in range(1, len(species_columns) + 1):
        n = int((per_candidate["n_strains_MIC_lt_cutoff"] >= k).sum())
        overlap_rows.append({
            "criterion": f"MIC < {args.one_digit_cutoff:g} µM against at least {k} strain(s)",
            "n_peptides": n,
            "percentage_of_portfolio": 100.0 * n / len(per_candidate),
        })
    n_all = int(per_candidate["has_one_digit_MIC_all_strains"].sum())
    overlap_rows.append({
        "criterion": f"MIC < {args.one_digit_cutoff:g} µM against all matched strains",
        "n_peptides": n_all,
        "percentage_of_portfolio": 100.0 * n_all / len(per_candidate),
    })
    overlap_summary = pd.DataFrame(overlap_rows)

    slug = safe_slug(args.species_regex)
    per_candidate.to_csv(output_dir / f"{slug}_all_candidate_summary.csv", index=False)
    qualifying.to_csv(output_dir / f"{slug}_one_digit_candidates.csv", index=False)
    strain_summary.to_csv(output_dir / f"{slug}_strain_summary.csv", index=False)
    overlap_summary.to_csv(output_dir / f"{slug}_overlap_summary.csv", index=False)

    heatmap_ids = qualifying["candidate_id"].tolist()
    heatmap = mic.reindex(index=heatmap_ids, columns=species_columns)
    heatmap.to_csv(output_dir / f"{slug}_one_digit_heatmap_matrix.csv")

    if not heatmap.empty:
        log_heatmap = np.log2(heatmap.clip(lower=1e-6))
        n_rows = len(log_heatmap)
        fig_height = min(24, max(7, n_rows * 0.025))
        fig_width = max(7, len(species_columns) * 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(log_heatmap.values, aspect="auto", interpolation="nearest")

        ax.set_xticks(np.arange(len(species_columns)))
        ax.set_xticklabels(species_columns, rotation=35, ha="right", fontsize=9)

        label_top_n = min(args.label_top_n, n_rows)
        if n_rows <= label_top_n:
            ticks = np.arange(n_rows)
        else:
            ticks = np.unique(np.linspace(0, n_rows - 1, label_top_n, dtype=int))
        ax.set_yticks(ticks)
        ax.set_yticklabels([heatmap_ids[i] for i in ticks], fontsize=6)

        ax.set_xlabel("Matched strain/model")
        ax.set_ylabel(
            f"Peptides with MIC < {args.one_digit_cutoff:g} µM against at least one matched strain"
        )
        ax.set_title(
            f"Species-filtered predicted MIC heatmap\n"
            f"{len(qualifying):,}/{len(per_candidate):,} peptides; regex: {args.species_regex}"
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Predicted MIC (log2 µM)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{slug}_one_digit_heatmap.png", dpi=600, bbox_inches="tight")
        fig.savefig(output_dir / f"{slug}_one_digit_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)

    summary = {
        "input_file": str(input_path),
        "species_regex": args.species_regex,
        "matched_columns": species_columns,
        "n_portfolio_peptides": int(len(per_candidate)),
        "n_matched_strains": int(len(species_columns)),
        "one_digit_definition": f"MIC < {args.one_digit_cutoff:g} µM",
        "n_one_digit_any_matched_strain": int(len(qualifying)),
        "percentage_one_digit_any_matched_strain": float(100.0 * len(qualifying) / len(per_candidate)),
        "n_one_digit_all_matched_strains": n_all,
        "percentage_one_digit_all_matched_strains": float(100.0 * n_all / len(per_candidate)),
    }
    with open(output_dir / f"{slug}_analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nSPECIES-SPECIFIC MIC ANALYSIS")
    print(json.dumps(summary, indent=2))
    print("\nPER-STRAIN COUNTS")
    print(strain_summary.to_string(index=False))
    print("\nOVERLAP COUNTS")
    print(overlap_summary.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

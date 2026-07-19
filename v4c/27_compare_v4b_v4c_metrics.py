#!/usr/bin/env python3
"""Compare native V4B and V4C AMP-JEPA portfolio metrics.

This script compares the final self-nonredundant portfolios and their
pre-self-similarity MIC<=32 pools using the same reporting code. It does not
claim that the native self-redundancy filters are methodologically identical:
V4B used its documented global-edit screen, whereas V4C used iterative CD-HIT
at 75% identity with bilateral 75% coverage.

All MIC values are APEX computational predictions, not measured MICs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
V4C_DIR = REPO_ROOT / "v4c"
if str(V4C_DIR) not in sys.path:
    sys.path.insert(0, str(V4C_DIR))

from apex_model_catalog import detect_apex_model_columns, infer_species_label  # noqa: E402


VERSION_ORDER = ["V4B", "V4C"]
MIC_THRESHOLDS = (10.0, 16.0, 32.0)
QUANTILES = (0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v4b-final",
        default=(
            "v4b/results/novelty_nonredundant_mic32/"
            "v4b_novel_MIC32_self_nonredundant_lt75.csv"
        ),
    )
    parser.add_argument(
        "--v4b-pre-self",
        default=(
            "v4b/results/novelty_nonredundant_mic32/"
            "v4b_novel_MIC32_ranked_before_self_similarity.csv"
        ),
    )
    parser.add_argument(
        "--v4c-final",
        default=(
            "v4c/results/final_funnel/04_self_nonredundant_cdhit75/"
            "v4c_novel_MIC32_self_nonredundant_cdhit75.csv"
        ),
    )
    parser.add_argument(
        "--v4c-pre-self",
        default=(
            "v4c/results/final_funnel/03_dual_novelty_mic32/"
            "v4c_dual_novel_MIC32_ranked.csv"
        ),
    )
    parser.add_argument("--v4b-generated", type=int, default=100000)
    parser.add_argument("--v4c-generated", type=int, default=1000000)
    parser.add_argument(
        "--output-dir",
        default="v4c/results/comparison_v4b_v4c",
    )
    parser.add_argument("--one-digit-cutoff", type=float, default=10.0)
    parser.add_argument("--minimum-numeric-fraction", type=float, default=0.95)
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500],
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def clean_sequence(value: object) -> str:
    return "".join(str(value).split()).upper()


def first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def prepare_frame(path: Path, version: str) -> pd.DataFrame:
    require_file(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"{version}: empty table: {path}")

    id_col = first_existing(
        frame.columns,
        ["candidate_id", "apex_candidate_id", "id", "peptide_id"],
    )
    seq_col = first_existing(
        frame.columns,
        ["sequence_clean", "sequence", "peptide_sequence", "peptide"],
    )
    if id_col is None:
        frame["candidate_id"] = [f"{version}_{i + 1:09d}" for i in range(len(frame))]
    elif id_col != "candidate_id":
        frame["candidate_id"] = frame[id_col].astype(str)
    else:
        frame["candidate_id"] = frame["candidate_id"].astype(str)

    if seq_col is None:
        raise ValueError(f"{version}: no sequence column detected in {path}")
    frame["sequence_clean"] = frame[seq_col].map(clean_sequence)

    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{version}: duplicate candidate IDs in {path}")
    if frame["sequence_clean"].duplicated().any():
        raise ValueError(f"{version}: duplicate peptide sequences in {path}")

    required_numeric = ["APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC"]
    missing = [column for column in required_numeric if column not in frame.columns]
    if missing:
        raise ValueError(f"{version}: missing APEX summary columns: {missing}")

    for column in required_numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required_numeric].isna().any().any():
        raise ValueError(f"{version}: missing/non-numeric APEX summary MIC values")

    aliases = {
        "criteria_length": ["criteria_length", "sequence_length", "length"],
        "criteria_charge": [
            "criteria_charge",
            "net_charge_KR_minus_DE",
            "net_charge",
            "charge",
        ],
        "criteria_hydrophobic_fraction": [
            "criteria_hydrophobic_fraction",
            "hydrophobic_fraction",
            "hydrophobicity",
        ],
        "generation_source": ["generation_source", "generation"],
        "organisms_MIC_le_64": [
            "organisms_MIC_le_64",
            "n_organisms_MIC_le_64",
            "n_pathogens_MIC_le_64",
        ],
    }
    for canonical, candidates in aliases.items():
        source = first_existing(frame.columns, candidates)
        if source is not None and canonical not in frame.columns:
            frame[canonical] = frame[source]

    for column in [
        "criteria_length",
        "criteria_charge",
        "criteria_hydrophobic_fraction",
        "organisms_MIC_le_64",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "criteria_length" not in frame.columns:
        frame["criteria_length"] = frame["sequence_clean"].str.len()

    frame["comparison_version"] = version
    return frame


def detect_models(
    frame: pd.DataFrame,
    minimum_numeric_fraction: float,
) -> list[str]:
    model_columns, _ = detect_apex_model_columns(
        frame,
        minimum_numeric_fraction=minimum_numeric_fraction,
    )
    for column in model_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return model_columns


def safe_numeric_summary(values: pd.Series, prefix: str) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_p05": float(numeric.quantile(0.05)),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_p95": float(numeric.quantile(0.95)),
        f"{prefix}_max": float(numeric.max()),
    }


def general_metrics(
    version: str,
    final: pd.DataFrame,
    pre_self: pd.DataFrame,
    generated: int,
    model_columns: list[str],
    one_digit_cutoff: float,
) -> dict[str, object]:
    mic = final[model_columns]
    tested = mic.notna().sum(axis=1)
    below = mic.lt(one_digit_cutoff)
    n_below = below.sum(axis=1)

    row: dict[str, object] = {
        "version": version,
        "generated_candidates": int(generated),
        "pre_self_MIC32_candidates": int(len(pre_self)),
        "final_self_nonredundant_candidates": int(len(final)),
        "unique_final_sequences": int(final["sequence_clean"].nunique()),
        "final_yield_percent_of_generated": float(100 * len(final) / generated),
        "final_candidates_per_100k_generated": float(100000 * len(final) / generated),
        "self_nonredundancy_retention_percent": float(
            100 * len(final) / max(len(pre_self), 1)
        ),
        "n_apex_models": int(len(model_columns)),
        "best_APEX_median_MIC_uM": float(final["APEX_median_MIC"].min()),
        "portfolio_median_APEX_median_MIC_uM": float(
            final["APEX_median_MIC"].median()
        ),
        "portfolio_median_APEX_worst_MIC_uM": float(
            final["APEX_worst_MIC"].median()
        ),
        "n_any_model_MIC_lt_10": int(n_below.ge(1).sum()),
        "percent_any_model_MIC_lt_10": float(100 * n_below.ge(1).mean()),
        "n_all_models_MIC_lt_10": int((tested.gt(0) & n_below.eq(tested)).sum()),
        "percent_all_models_MIC_lt_10": float(
            100 * (tested.gt(0) & n_below.eq(tested)).mean()
        ),
        "median_number_models_MIC_lt_10": float(n_below.median()),
        "maximum_number_models_MIC_lt_10": int(n_below.max()),
    }
    for threshold in MIC_THRESHOLDS:
        label = str(int(threshold))
        mask = final["APEX_median_MIC"].lt(threshold) if threshold == 10 else final[
            "APEX_median_MIC"
        ].le(threshold)
        row[f"n_APEX_median_MIC_{'lt' if threshold == 10 else 'le'}_{label}"] = int(
            mask.sum()
        )
        row[
            f"percent_APEX_median_MIC_{'lt' if threshold == 10 else 'le'}_{label}"
        ] = float(100 * mask.mean())

    row.update(safe_numeric_summary(final["criteria_length"], "length"))
    if "criteria_charge" in final.columns:
        row.update(safe_numeric_summary(final["criteria_charge"], "charge"))
    if "criteria_hydrophobic_fraction" in final.columns:
        row.update(
            safe_numeric_summary(
                final["criteria_hydrophobic_fraction"],
                "hydrophobic_fraction",
            )
        )
    return row


def potency_quantiles(version: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ["APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC"]:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        for quantile in QUANTILES:
            rows.append(
                {
                    "version": version,
                    "metric": metric,
                    "quantile": quantile,
                    "value_uM": float(values.quantile(quantile)),
                }
            )
    return pd.DataFrame(rows)


def species_metrics(
    version: str,
    frame: pd.DataFrame,
    model_columns: list[str],
    cutoff: float,
) -> pd.DataFrame:
    species_to_columns: dict[str, list[str]] = {}
    for column in model_columns:
        species_to_columns.setdefault(infer_species_label(column), []).append(column)

    rows: list[dict[str, object]] = []
    for species, columns in sorted(species_to_columns.items()):
        mic = frame[columns]
        tested = mic.notna().sum(axis=1)
        below = mic.lt(cutoff)
        count_below = below.sum(axis=1)
        any_mask = count_below.ge(1)
        all_mask = tested.gt(0) & count_below.eq(tested)
        per_candidate_median = mic.median(axis=1)
        best_per_candidate = mic.min(axis=1)
        best_index = best_per_candidate.idxmin()
        rows.append(
            {
                "version": version,
                "species": species,
                "n_models": len(columns),
                "model_columns": " | ".join(columns),
                "n_portfolio_candidates": len(frame),
                "n_candidates_any_model_MIC_lt_10": int(any_mask.sum()),
                "percent_candidates_any_model_MIC_lt_10": float(
                    100 * any_mask.mean()
                ),
                "n_candidates_all_models_MIC_lt_10": int(all_mask.sum()),
                "percent_candidates_all_models_MIC_lt_10": float(
                    100 * all_mask.mean()
                ),
                "best_predicted_species_MIC_uM": float(mic.min().min()),
                "median_candidate_species_median_MIC_uM": float(
                    per_candidate_median.median()
                ),
                "best_candidate_id": str(frame.loc[best_index, "candidate_id"]),
            }
        )
    return pd.DataFrame(rows)


def top_k_metrics(
    version: str,
    frame: pd.DataFrame,
    model_columns: list[str],
    top_k_values: list[int],
    cutoff: float,
) -> pd.DataFrame:
    ranked = frame.copy()
    mic = ranked[model_columns]
    ranked["_n_models_lt10"] = mic.lt(cutoff).sum(axis=1)
    ranked["_fraction_models_lt10"] = (
        ranked["_n_models_lt10"] / mic.notna().sum(axis=1).replace(0, np.nan)
    )
    ranked = ranked.sort_values(
        [
            "_n_models_lt10",
            "_fraction_models_lt10",
            "APEX_median_MIC",
            "APEX_worst_MIC",
            "APEX_mean_MIC",
        ],
        ascending=[False, False, True, True, True],
    )

    rows: list[dict[str, object]] = []
    for requested_k in sorted(set(top_k_values)):
        k = min(requested_k, len(ranked))
        subset = ranked.head(k)
        rows.append(
            {
                "version": version,
                "requested_top_k": requested_k,
                "actual_k": k,
                "median_APEX_median_MIC_uM": float(
                    subset["APEX_median_MIC"].median()
                ),
                "worst_APEX_median_MIC_uM": float(
                    subset["APEX_median_MIC"].max()
                ),
                "median_APEX_worst_MIC_uM": float(
                    subset["APEX_worst_MIC"].median()
                ),
                "median_models_MIC_lt_10": float(
                    subset["_n_models_lt10"].median()
                ),
                "minimum_models_MIC_lt_10": int(
                    subset["_n_models_lt10"].min()
                ),
                "unique_lengths": int(subset["criteria_length"].nunique()),
                "unique_sequences": int(subset["sequence_clean"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def generation_metrics(version: str, frame: pd.DataFrame) -> pd.DataFrame:
    if "generation_source" not in frame.columns:
        return pd.DataFrame()
    result = (
        frame.groupby("generation_source", dropna=False)
        .agg(
            final_candidates=("candidate_id", "count"),
            best_APEX_median_MIC_uM=("APEX_median_MIC", "min"),
            median_APEX_median_MIC_uM=("APEX_median_MIC", "median"),
            n_APEX_median_MIC_lt_10=("APEX_median_MIC", lambda x: x.lt(10).sum()),
            unique_sequences=("sequence_clean", "nunique"),
        )
        .reset_index()
    )
    result.insert(0, "version", version)
    return result


def exact_overlap(v4b: pd.DataFrame, v4c: pd.DataFrame) -> dict[str, object]:
    b = set(v4b["sequence_clean"])
    c = set(v4c["sequence_clean"])
    shared = b & c
    return {
        "v4b_final_sequences": len(b),
        "v4c_final_sequences": len(c),
        "shared_exact_sequences": len(shared),
        "v4b_fraction_shared_with_v4c": len(shared) / max(len(b), 1),
        "v4c_fraction_inherited_exactly_from_v4b_final": len(shared) / max(len(c), 1),
        "v4c_sequences_not_in_v4b_final": len(c - b),
    }


def plot_ecdf(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for version in VERSION_ORDER:
        values = np.sort(frames[version]["APEX_median_MIC"].to_numpy(dtype=float))
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, label=f"{version} (n={len(values):,})")
    ax.set_xlabel("Predicted APEX median MIC (µM)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("V4B vs V4C final portfolio potency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_v4b_v4c_APEX_median_MIC_ECDF.png", dpi=600)
    fig.savefig(output_dir / "01_v4b_v4c_APEX_median_MIC_ECDF.pdf")
    plt.close(fig)


def plot_yield(general: pd.DataFrame, output_dir: Path) -> None:
    values = general.set_index("version").loc[VERSION_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(values.index, values["final_candidates_per_100k_generated"])
    ax.set_ylabel("Final candidates per 100,000 generated")
    ax.set_title("Native final portfolio yield")
    for index, value in enumerate(values["final_candidates_per_100k_generated"]):
        ax.text(index, value, f"{value:,.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "02_v4b_v4c_final_yield_per_100k.png", dpi=600)
    fig.savefig(output_dir / "02_v4b_v4c_final_yield_per_100k.pdf")
    plt.close(fig)


def plot_species_heatmap(species: pd.DataFrame, output_dir: Path) -> None:
    matrix = species.pivot(
        index="species",
        columns="version",
        values="percent_candidates_any_model_MIC_lt_10",
    ).reindex(columns=VERSION_ORDER)
    matrix = matrix.dropna(how="all").sort_index()
    matrix.to_csv(output_dir / "v4b_v4c_species_one_digit_percentage_matrix.csv")
    if matrix.empty:
        return
    fig_height = max(6, 0.42 * len(matrix) + 2)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Version")
    ax.set_ylabel("Species")
    ax.set_title("Final portfolio with predicted MIC <10 µM\nagainst any species model (%)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Candidates (%)")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            if pd.notna(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    fig.tight_layout()
    fig.savefig(output_dir / "03_v4b_v4c_species_one_digit_heatmap.png", dpi=600)
    fig.savefig(output_dir / "03_v4b_v4c_species_one_digit_heatmap.pdf")
    plt.close(fig)


def plot_distribution(
    frames: dict[str, pd.DataFrame],
    column: str,
    xlabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    if not all(column in frames[version].columns for version in VERSION_ORDER):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for version in VERSION_ORDER:
        values = pd.to_numeric(frames[version][column], errors="coerce").dropna()
        ax.hist(values, bins=30, density=True, alpha=0.5, label=version)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"V4B vs V4C final portfolio: {xlabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=600)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "V4B": {
            "final": Path(args.v4b_final),
            "pre_self": Path(args.v4b_pre_self),
            "generated": args.v4b_generated,
        },
        "V4C": {
            "final": Path(args.v4c_final),
            "pre_self": Path(args.v4c_pre_self),
            "generated": args.v4c_generated,
        },
    }

    final_frames: dict[str, pd.DataFrame] = {}
    pre_self_frames: dict[str, pd.DataFrame] = {}
    model_columns_by_version: dict[str, list[str]] = {}

    for version in VERSION_ORDER:
        print(f"[COMPARE] Loading {version} final portfolio")
        final_frames[version] = prepare_frame(paths[version]["final"], version)
        print(f"[COMPARE] Loading {version} pre-self MIC32 pool")
        pre_self_frames[version] = prepare_frame(paths[version]["pre_self"], version)
        model_columns_by_version[version] = detect_models(
            final_frames[version],
            args.minimum_numeric_fraction,
        )

    common_models = [
        column
        for column in model_columns_by_version["V4B"]
        if column in set(model_columns_by_version["V4C"])
    ]
    if not common_models:
        raise RuntimeError("V4B and V4C have no common recognized APEX model columns.")

    for version in VERSION_ORDER:
        missing = sorted(set(common_models) - set(final_frames[version].columns))
        if missing:
            raise RuntimeError(f"{version} missing common model columns: {missing}")

    print(f"[COMPARE] Common APEX organism/strain models: {len(common_models)}")
    (output_dir / "common_apex_model_columns.txt").write_text(
        "\n".join(common_models) + "\n",
        encoding="utf-8",
    )

    general_rows = [
        general_metrics(
            version,
            final_frames[version],
            pre_self_frames[version],
            int(paths[version]["generated"]),
            common_models,
            args.one_digit_cutoff,
        )
        for version in VERSION_ORDER
    ]
    general = pd.DataFrame(general_rows)
    general.to_csv(output_dir / "v4b_v4c_general_metrics.csv", index=False)

    quantiles = pd.concat(
        [
            potency_quantiles(version, final_frames[version])
            for version in VERSION_ORDER
        ],
        ignore_index=True,
    )
    quantiles.to_csv(output_dir / "v4b_v4c_potency_quantiles.csv", index=False)

    species = pd.concat(
        [
            species_metrics(
                version,
                final_frames[version],
                common_models,
                args.one_digit_cutoff,
            )
            for version in VERSION_ORDER
        ],
        ignore_index=True,
    )
    species.to_csv(output_dir / "v4b_v4c_species_metrics.csv", index=False)

    top_k = pd.concat(
        [
            top_k_metrics(
                version,
                final_frames[version],
                common_models,
                args.top_k,
                args.one_digit_cutoff,
            )
            for version in VERSION_ORDER
        ],
        ignore_index=True,
    )
    top_k.to_csv(output_dir / "v4b_v4c_top_k_metrics.csv", index=False)

    generation_tables = [
        generation_metrics(version, final_frames[version])
        for version in VERSION_ORDER
    ]
    generation_tables = [table for table in generation_tables if not table.empty]
    if generation_tables:
        pd.concat(generation_tables, ignore_index=True).to_csv(
            output_dir / "v4b_v4c_generation_metrics.csv",
            index=False,
        )

    overlap = exact_overlap(final_frames["V4B"], final_frames["V4C"])
    with (output_dir / "v4b_v4c_exact_sequence_overlap.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(overlap, handle, indent=2)

    plot_ecdf(final_frames, output_dir)
    plot_yield(general, output_dir)
    plot_species_heatmap(species, output_dir)
    plot_distribution(
        final_frames,
        "criteria_length",
        "Peptide length (aa)",
        "04_v4b_v4c_length_distribution",
        output_dir,
    )
    plot_distribution(
        final_frames,
        "criteria_charge",
        "Net charge (K+R−D−E)",
        "05_v4b_v4c_charge_distribution",
        output_dir,
    )
    plot_distribution(
        final_frames,
        "criteria_hydrophobic_fraction",
        "Hydrophobic fraction",
        "06_v4b_v4c_hydrophobic_fraction_distribution",
        output_dir,
    )

    comparison_summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "AMP-JEPA-Hybrid V4B versus V4C native final portfolios",
        "one_digit_definition": (
            f"predicted MIC < {args.one_digit_cutoff:g} µM"
        ),
        "common_apex_models": len(common_models),
        "general_metrics": general.to_dict(orient="records"),
        "exact_sequence_overlap": overlap,
        "interpretation_guardrails": [
            "All MIC values are APEX computational predictions, not measured MICs.",
            (
                "This is a native-workflow comparison. V4B and V4C self-redundancy "
                "methods are not exactly equivalent, so final portfolio counts are "
                "not a fully harmonized method comparison."
            ),
            (
                "Raw counts must be interpreted together with candidates per 100,000 "
                "generated because V4C used a tenfold larger generation budget."
            ),
        ],
        "outputs": {
            "general_metrics": str(
                output_dir / "v4b_v4c_general_metrics.csv"
            ),
            "potency_quantiles": str(
                output_dir / "v4b_v4c_potency_quantiles.csv"
            ),
            "species_metrics": str(
                output_dir / "v4b_v4c_species_metrics.csv"
            ),
            "top_k_metrics": str(
                output_dir / "v4b_v4c_top_k_metrics.csv"
            ),
            "exact_sequence_overlap": str(
                output_dir / "v4b_v4c_exact_sequence_overlap.json"
            ),
        },
    }
    with (output_dir / "v4b_v4c_comparison_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(comparison_summary, handle, indent=2)

    display_columns = [
        "version",
        "generated_candidates",
        "pre_self_MIC32_candidates",
        "final_self_nonredundant_candidates",
        "final_candidates_per_100k_generated",
        "self_nonredundancy_retention_percent",
        "best_APEX_median_MIC_uM",
        "portfolio_median_APEX_median_MIC_uM",
        "n_APEX_median_MIC_lt_10",
        "n_any_model_MIC_lt_10",
    ]
    print("\nV4B VERSUS V4C GENERAL METRICS")
    print(general[display_columns].round(4).to_string(index=False))
    print("\nEXACT FINAL-SEQUENCE OVERLAP")
    print(json.dumps(overlap, indent=2))
    print(f"\n[COMPARE] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

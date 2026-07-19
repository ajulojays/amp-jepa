#!/usr/bin/env python3
"""Prioritize the certified V4C portfolio without introducing an opaque composite score.

This stage starts from the frozen, CD-HIT-certified V4C portfolio and produces
transparent lexicographic rankings for:

1. global predicted potency;
2. broad one-digit predicted-MIC coverage;
3. worst-case robustness;
4. Salmonella enterica activity, when Salmonella models are present.

The organism-specific APEX columns are read from an APEX scoring-summary JSON,
which avoids guessing columns from the wide candidate table. All MIC values remain
computational predictions in micromolar units; this script does not treat them as
experimental measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "v4c/results/final_funnel/04_self_nonredundant_cdhit75/"
            "v4c_novel_MIC32_self_nonredundant_cdhit75.csv"
        ),
    )
    parser.add_argument(
        "--apex-summary",
        default=(
            "v4c/results/generation_01/"
            "generation_01_apex_scores_apex_summary.json"
        ),
        help="Any V4C generation APEX summary containing the organism_columns list.",
    )
    parser.add_argument(
        "--output-dir",
        default="v4c/results/final_funnel/05_portfolio_prioritization",
    )
    parser.add_argument("--expected-input", type=int, default=7507)
    parser.add_argument("--one-digit-cutoff", type=float, default=10.0)
    parser.add_argument("--intermediate-cutoff", type=float, default=16.0)
    parser.add_argument("--broad-cutoff", type=float, default=32.0)
    parser.add_argument("--coverage-cutoff", type=float, default=64.0)
    parser.add_argument("--top-n", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--per-organism-top-n", type=int, default=25)
    parser.add_argument("--salmonella-regex", default=r"Salmonella enterica")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Required numeric column is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"Column {column!r} contains {int(values.isna().sum()):,} missing/non-numeric values."
        )
    return values.astype(float)


def safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return value[:180] or "organism"


def validate_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Portfolio-prioritization outputs already exist. Review them or use "
            "--overwrite:\n" + rendered
        )


def rank_lexicographically(
    frame: pd.DataFrame,
    sort_columns: list[str],
    ascending: list[bool],
    rank_column: str,
) -> pd.DataFrame:
    ranked = frame.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked[rank_column] = np.arange(1, len(ranked) + 1)
    return ranked


def export_top_panels(
    ranked: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    top_values: list[int],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for requested in sorted(set(int(value) for value in top_values if int(value) > 0)):
        actual = min(requested, len(ranked))
        path = output_dir / f"{prefix}_top{actual}.csv"
        atomic_csv(ranked.head(actual), path)
        outputs[f"top_{actual}"] = str(path)
    return outputs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    apex_summary_path = Path(args.apex_summary)
    output_dir = Path(args.output_dir)
    per_organism_dir = output_dir / "per_organism_top_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_organism_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(input_path)
    if not apex_summary_path.exists() or apex_summary_path.stat().st_size == 0:
        raise FileNotFoundError(apex_summary_path)

    ranked_all_path = output_dir / "v4c_certified_portfolio_all_ranked.csv"
    one_digit_median_path = output_dir / "v4c_median_predicted_MIC_lt10.csv"
    organism_summary_path = output_dir / "v4c_APEX_model_summary.csv"
    organism_top_path = output_dir / "v4c_top_candidates_per_APEX_model.csv"
    threshold_summary_path = output_dir / "v4c_portfolio_threshold_summary.csv"
    generation_summary_path = output_dir / "v4c_portfolio_summary_by_generation.csv"
    salmonella_path = output_dir / "v4c_salmonella_prioritized.csv"
    salmonella_summary_path = output_dir / "v4c_salmonella_summary.json"
    summary_path = output_dir / "v4c_portfolio_prioritization_summary.json"

    validate_outputs(
        [
            ranked_all_path,
            one_digit_median_path,
            organism_summary_path,
            organism_top_path,
            threshold_summary_path,
            generation_summary_path,
            summary_path,
        ],
        overwrite=args.overwrite,
    )

    with apex_summary_path.open("r", encoding="utf-8") as handle:
        apex_summary = json.load(handle)
    organism_columns = apex_summary.get("organism_columns")
    if not isinstance(organism_columns, list) or not organism_columns:
        raise ValueError(
            f"APEX summary does not contain a non-empty organism_columns list: {apex_summary_path}"
        )
    organism_columns = [str(column) for column in organism_columns]
    if len(organism_columns) != len(set(organism_columns)):
        raise ValueError("APEX summary contains duplicate organism columns.")

    print("[V4C-PRIORITY] Loading certified portfolio")
    frame = pd.read_csv(input_path, low_memory=False)
    required = {
        "candidate_id",
        "APEX_median_MIC",
        "APEX_mean_MIC",
        "APEX_worst_MIC",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Certified portfolio is missing required columns: {missing}")
    missing_organisms = [column for column in organism_columns if column not in frame.columns]
    if missing_organisms:
        raise ValueError(
            "Certified portfolio is missing APEX organism columns from the summary: "
            + ", ".join(missing_organisms[:10])
        )

    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Certified portfolio contains duplicate candidate IDs.")
    sequence_column = "sequence_clean" if "sequence_clean" in frame.columns else "sequence"
    if sequence_column not in frame.columns:
        raise ValueError("Certified portfolio must contain sequence_clean or sequence.")
    frame["sequence_clean"] = (
        frame[sequence_column].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    )
    if frame["sequence_clean"].duplicated().any():
        raise ValueError("Certified portfolio contains duplicate peptide sequences.")
    if args.expected_input > 0 and len(frame) != args.expected_input:
        raise ValueError(
            f"Certified portfolio count {len(frame):,} != expected {args.expected_input:,}."
        )

    for column in ["APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC"] + organism_columns:
        frame[column] = numeric(frame, column)

    mic = frame[organism_columns]
    if mic.isna().any().any():
        raise ValueError("At least one APEX organism-specific MIC value is missing.")
    if (mic <= 0).any().any():
        raise ValueError("APEX organism-specific MIC values must be positive.")

    n_models = len(organism_columns)
    frame["n_APEX_models"] = n_models
    thresholds = [
        ("lt10", args.one_digit_cutoff, False),
        ("le16", args.intermediate_cutoff, True),
        ("le32", args.broad_cutoff, True),
        ("le64", args.coverage_cutoff, True),
    ]
    for label, cutoff, inclusive in thresholds:
        hits = mic.le(cutoff) if inclusive else mic.lt(cutoff)
        frame[f"n_models_MIC_{label}"] = hits.sum(axis=1).astype(int)
        frame[f"fraction_models_MIC_{label}"] = (
            frame[f"n_models_MIC_{label}"] / float(n_models)
        )

    frame["APEX_best_model_MIC"] = mic.min(axis=1)
    frame["APEX_recomputed_mean_MIC"] = mic.mean(axis=1)
    frame["APEX_recomputed_median_MIC"] = mic.median(axis=1)
    frame["APEX_recomputed_worst_MIC"] = mic.max(axis=1)

    tolerance = 1e-5
    for stored, recomputed in [
        ("APEX_mean_MIC", "APEX_recomputed_mean_MIC"),
        ("APEX_median_MIC", "APEX_recomputed_median_MIC"),
        ("APEX_worst_MIC", "APEX_recomputed_worst_MIC"),
    ]:
        delta = (frame[stored] - frame[recomputed]).abs()
        frame[f"audit_{stored}_matches_models"] = delta.le(tolerance)
        if not frame[f"audit_{stored}_matches_models"].all():
            raise ValueError(
                f"Stored {stored} disagrees with organism-model values for "
                f"{int((delta > tolerance).sum()):,} candidates."
            )

    global_ranked = rank_lexicographically(
        frame,
        [
            "APEX_median_MIC",
            "APEX_worst_MIC",
            "APEX_mean_MIC",
            "n_models_MIC_lt10",
            "n_models_MIC_le16",
        ],
        [True, True, True, False, False],
        "v4c_global_potency_rank",
    )
    global_rank_map = dict(
        zip(global_ranked["candidate_id"], global_ranked["v4c_global_potency_rank"])
    )
    frame["v4c_global_potency_rank"] = frame["candidate_id"].map(global_rank_map)

    breadth_ranked = rank_lexicographically(
        frame,
        [
            "n_models_MIC_lt10",
            "n_models_MIC_le16",
            "n_models_MIC_le32",
            "APEX_median_MIC",
            "APEX_worst_MIC",
        ],
        [False, False, False, True, True],
        "v4c_breadth_rank",
    )
    breadth_rank_map = dict(
        zip(breadth_ranked["candidate_id"], breadth_ranked["v4c_breadth_rank"])
    )
    frame["v4c_breadth_rank"] = frame["candidate_id"].map(breadth_rank_map)

    robustness_ranked = rank_lexicographically(
        frame,
        [
            "APEX_worst_MIC",
            "APEX_median_MIC",
            "APEX_mean_MIC",
            "n_models_MIC_le64",
        ],
        [True, True, True, False],
        "v4c_robustness_rank",
    )
    robustness_rank_map = dict(
        zip(robustness_ranked["candidate_id"], robustness_ranked["v4c_robustness_rank"])
    )
    frame["v4c_robustness_rank"] = frame["candidate_id"].map(robustness_rank_map)

    frame = frame.sort_values("v4c_global_potency_rank").reset_index(drop=True)
    atomic_csv(frame, ranked_all_path)

    one_digit_median = frame[frame["APEX_median_MIC"].lt(args.one_digit_cutoff)].copy()
    atomic_csv(one_digit_median, one_digit_median_path)

    panel_outputs = {
        "global_potency": export_top_panels(
            global_ranked,
            output_dir,
            "v4c_global_potency",
            args.top_n,
        ),
        "breadth": export_top_panels(
            breadth_ranked,
            output_dir,
            "v4c_broad_spectrum",
            args.top_n,
        ),
        "robustness": export_top_panels(
            robustness_ranked,
            output_dir,
            "v4c_worst_case_robustness",
            args.top_n,
        ),
    }

    organism_rows: list[dict[str, object]] = []
    organism_top_frames: list[pd.DataFrame] = []
    top_per_organism = max(1, int(args.per_organism_top_n))
    for column in organism_columns:
        values = frame[column]
        ranked_model = frame.sort_values(
            [column, "APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).head(top_per_organism).copy()
        ranked_model["APEX_model"] = column
        ranked_model["APEX_model_predicted_MIC"] = ranked_model[column]
        ranked_model["APEX_model_rank"] = np.arange(1, len(ranked_model) + 1)
        compact_columns = [
            candidate
            for candidate in [
                "APEX_model",
                "APEX_model_rank",
                "candidate_id",
                "sequence_clean",
                "generation_source",
                "APEX_model_predicted_MIC",
                "APEX_median_MIC",
                "APEX_worst_MIC",
                "n_models_MIC_lt10",
                "n_models_MIC_le16",
                "n_models_MIC_le32",
                "v4c_global_potency_rank",
                "v4c_breadth_rank",
                "v4c_robustness_rank",
                "criteria_length",
                "criteria_charge",
                "criteria_hydrophobic_fraction",
            ]
            if candidate in ranked_model.columns
        ]
        ranked_compact = ranked_model[compact_columns]
        organism_top_frames.append(ranked_compact)
        atomic_csv(
            ranked_compact,
            per_organism_dir / f"{safe_slug(column)}_top{len(ranked_compact)}.csv",
        )

        organism_rows.append(
            {
                "APEX_model": column,
                "n_candidates": int(values.notna().sum()),
                "n_predicted_MIC_lt10": int(values.lt(args.one_digit_cutoff).sum()),
                "fraction_predicted_MIC_lt10": float(values.lt(args.one_digit_cutoff).mean()),
                "n_predicted_MIC_le16": int(values.le(args.intermediate_cutoff).sum()),
                "n_predicted_MIC_le32": int(values.le(args.broad_cutoff).sum()),
                "n_predicted_MIC_le64": int(values.le(args.coverage_cutoff).sum()),
                "best_predicted_MIC_uM": float(values.min()),
                "median_predicted_MIC_uM": float(values.median()),
                "worst_predicted_MIC_uM": float(values.max()),
                "best_candidate_id": str(frame.loc[values.idxmin(), "candidate_id"]),
            }
        )

    organism_summary = pd.DataFrame(organism_rows).sort_values(
        ["median_predicted_MIC_uM", "best_predicted_MIC_uM"],
        ascending=[True, True],
    )
    atomic_csv(organism_summary, organism_summary_path)
    organism_top = pd.concat(organism_top_frames, ignore_index=True)
    atomic_csv(organism_top, organism_top_path)

    threshold_rows = []
    for label, cutoff, inclusive in thresholds:
        median_mask = (
            frame["APEX_median_MIC"].le(cutoff)
            if inclusive
            else frame["APEX_median_MIC"].lt(cutoff)
        )
        worst_mask = (
            frame["APEX_worst_MIC"].le(cutoff)
            if inclusive
            else frame["APEX_worst_MIC"].lt(cutoff)
        )
        operator = "<=" if inclusive else "<"
        threshold_rows.append(
            {
                "criterion": f"predicted median MIC {operator} {cutoff:g} uM",
                "n_candidates": int(median_mask.sum()),
                "percentage_portfolio": float(100.0 * median_mask.mean()),
            }
        )
        threshold_rows.append(
            {
                "criterion": f"predicted worst MIC {operator} {cutoff:g} uM",
                "n_candidates": int(worst_mask.sum()),
                "percentage_portfolio": float(100.0 * worst_mask.mean()),
            }
        )
    threshold_rows.extend(
        [
            {
                "criterion": f"predicted MIC < {args.one_digit_cutoff:g} uM for at least one APEX model",
                "n_candidates": int(frame["n_models_MIC_lt10"].ge(1).sum()),
                "percentage_portfolio": float(100.0 * frame["n_models_MIC_lt10"].ge(1).mean()),
            },
            {
                "criterion": f"predicted MIC < {args.one_digit_cutoff:g} uM for all APEX models",
                "n_candidates": int(frame["n_models_MIC_lt10"].eq(n_models).sum()),
                "percentage_portfolio": float(100.0 * frame["n_models_MIC_lt10"].eq(n_models).mean()),
            },
        ]
    )
    threshold_summary = pd.DataFrame(threshold_rows)
    atomic_csv(threshold_summary, threshold_summary_path)

    if "generation_source" in frame.columns:
        generation_summary = (
            frame.groupby("generation_source")
            .agg(
                n_candidates=("candidate_id", "count"),
                n_median_MIC_lt10=("APEX_median_MIC", lambda values: int(values.lt(args.one_digit_cutoff).sum())),
                best_median_MIC=("APEX_median_MIC", "min"),
                median_of_median_MIC=("APEX_median_MIC", "median"),
                median_one_digit_model_coverage=("n_models_MIC_lt10", "median"),
            )
            .reset_index()
        )
        atomic_csv(generation_summary, generation_summary_path)
    else:
        generation_summary = pd.DataFrame()

    salmonella_pattern = re.compile(args.salmonella_regex, flags=re.IGNORECASE)
    salmonella_columns = [
        column for column in organism_columns if salmonella_pattern.search(column)
    ]
    salmonella_summary: dict[str, object] | None = None
    if salmonella_columns:
        salmonella_mic = frame[salmonella_columns]
        frame["n_salmonella_models_MIC_lt10"] = salmonella_mic.lt(
            args.one_digit_cutoff
        ).sum(axis=1)
        frame["salmonella_best_predicted_MIC"] = salmonella_mic.min(axis=1)
        frame["salmonella_median_predicted_MIC"] = salmonella_mic.median(axis=1)
        frame["salmonella_worst_predicted_MIC"] = salmonella_mic.max(axis=1)
        salmonella_ranked = rank_lexicographically(
            frame,
            [
                "n_salmonella_models_MIC_lt10",
                "salmonella_median_predicted_MIC",
                "salmonella_worst_predicted_MIC",
                "APEX_median_MIC",
            ],
            [False, True, True, True],
            "v4c_salmonella_rank",
        )
        atomic_csv(salmonella_ranked, salmonella_path)
        panel_outputs["salmonella"] = export_top_panels(
            salmonella_ranked,
            output_dir,
            "v4c_salmonella",
            args.top_n,
        )
        salmonella_summary = {
            "matched_models": salmonella_columns,
            "n_matched_models": len(salmonella_columns),
            "n_candidates_MIC_lt10_any_salmonella": int(
                salmonella_ranked["n_salmonella_models_MIC_lt10"].ge(1).sum()
            ),
            "n_candidates_MIC_lt10_all_salmonella": int(
                salmonella_ranked["n_salmonella_models_MIC_lt10"].eq(len(salmonella_columns)).sum()
            ),
            "best_predicted_salmonella_MIC_uM": float(
                salmonella_ranked["salmonella_best_predicted_MIC"].min()
            ),
            "best_salmonella_candidate_id": str(salmonella_ranked.iloc[0]["candidate_id"]),
            "ranked_output": str(salmonella_path),
        }
        salmonella_summary_path.write_text(
            json.dumps(salmonella_summary, indent=2), encoding="utf-8"
        )

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "certified_portfolio_prioritization",
        "input_file": str(input_path),
        "apex_summary_file": str(apex_summary_path),
        "input_candidates": int(len(frame)),
        "n_APEX_models": n_models,
        "one_digit_definition": f"predicted MIC < {args.one_digit_cutoff:g} uM",
        "n_candidates_predicted_median_MIC_lt10": int(len(one_digit_median)),
        "n_candidates_any_model_MIC_lt10": int(frame["n_models_MIC_lt10"].ge(1).sum()),
        "n_candidates_all_models_MIC_lt10": int(frame["n_models_MIC_lt10"].eq(n_models).sum()),
        "best_predicted_median_MIC_uM": float(frame["APEX_median_MIC"].min()),
        "portfolio_median_predicted_median_MIC_uM": float(frame["APEX_median_MIC"].median()),
        "ranking_methods": {
            "global_potency": [
                "APEX_median_MIC ascending",
                "APEX_worst_MIC ascending",
                "APEX_mean_MIC ascending",
                "n_models_MIC_lt10 descending",
                "n_models_MIC_le16 descending",
            ],
            "breadth": [
                "n_models_MIC_lt10 descending",
                "n_models_MIC_le16 descending",
                "n_models_MIC_le32 descending",
                "APEX_median_MIC ascending",
                "APEX_worst_MIC ascending",
            ],
            "robustness": [
                "APEX_worst_MIC ascending",
                "APEX_median_MIC ascending",
                "APEX_mean_MIC ascending",
                "n_models_MIC_le64 descending",
            ],
        },
        "important_note": (
            "All MIC values are APEX computational predictions, not measured MICs. "
            "Rankings are transparent lexicographic orderings rather than an opaque weighted score."
        ),
        "salmonella": salmonella_summary,
        "outputs": {
            "all_ranked": str(ranked_all_path),
            "median_predicted_MIC_lt10": str(one_digit_median_path),
            "APEX_model_summary": str(organism_summary_path),
            "top_candidates_per_APEX_model": str(organism_top_path),
            "threshold_summary": str(threshold_summary_path),
            "generation_summary": str(generation_summary_path),
            "per_organism_directory": str(per_organism_dir),
            "panels": panel_outputs,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nV4C CERTIFIED PORTFOLIO PRIORITIZATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nTHRESHOLD COUNTS")
    print(threshold_summary.round(3).to_string(index=False))
    print("\nTOP 20 GLOBAL POTENCY CANDIDATES")
    show = [
        column
        for column in [
            "v4c_global_potency_rank",
            "candidate_id",
            "generation_source",
            "sequence_clean",
            "APEX_median_MIC",
            "APEX_worst_MIC",
            "APEX_mean_MIC",
            "n_models_MIC_lt10",
            "n_models_MIC_le16",
            "n_models_MIC_le32",
            "criteria_length",
            "criteria_charge",
            "criteria_hydrophobic_fraction",
        ]
        if column in global_ranked.columns
    ]
    print(global_ranked[show].head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

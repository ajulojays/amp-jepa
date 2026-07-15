#!/usr/bin/env python3
"""QC audit for the AMP-JEPA-Hybrid v3 upscaled corpus.

This script audits the merged corpus created by v3/37_build_upscaled_corpus.py.
It does not overwrite the original corpus. Instead, it writes:

- a row-level QC table with flags and confidence tiers
- source-level and length-bin summaries
- amino-acid composition summaries
- duplicate/provenance summaries
- trainable-core CSV/FASTA files for cleaner retraining
- a JSON report with key warnings

The goal is to catch problems introduced by permissive archive extraction, such as
non-AMP controls, feature matrices, requirement files, validation-result tables, or
columns that were automatically inferred but may not truly contain peptide sequences.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWYC")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

HIGH_CONFIDENCE_SOURCE_NAMES = {
    "APD",
    "DRAMP",
    "DBAASP",
    "DBAMP",
    "DBAMP",
    "CAMPR",
    "CAMP",
    "UNIPROT",
    "STARPEP",
}

MEDIUM_CONFIDENCE_SOURCE_NAMES = {
    "AMPLIFY",
    "AMPCLIFF",
    "ADAM",
}

SUSPICIOUS_PATH_MARKERS = [
    "noamp",
    "no_amp",
    "nonamp",
    "non-amp",
    "negative",
    "negatives",
    "control",
    "decoy",
    "random",
    "requirements",
    "documentation",
    "readme",
    "license",
    "models' evaluation",
    "model_evaluation",
    "evaluation",
    "selected_features",
    "features.csv",
    "blosum",
    "tanimoto",
    "normalized.csv",
    "final_data.csv",
    "combined_prop.tsv",
    "filename",
]

SUSPICIOUS_SOURCE_NAME_MARKERS = [
    "noamp",
    "requirements",
    "documentation",
    "comb_",
    "final_data",
]

PREFERRED_TRAINING_SOURCE_NAMES = {
    "APD",
    "DRAMP",
    "DBAASP",
    "DBAMP",
    "CAMPR",
    "CAMP",
    "UNIPROT",
}


# -----------------------------
# Basic sequence feature helpers
# -----------------------------

def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def clean_sequence(value: object) -> str:
    sequence = str(value).strip().upper()
    return "".join(residue for residue in sequence if residue in AMINO_ACIDS)


def sequence_entropy(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    length = len(sequence)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def longest_homopolymer(sequence: str) -> int:
    if not sequence:
        return 0
    best = 1
    current = 1
    for index in range(1, len(sequence)):
        if sequence[index] == sequence[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def residue_fraction(sequence: str, residues: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(residue in residues for residue in sequence) / len(sequence)


def approximate_net_charge(sequence: str) -> float:
    # Same approximate convention used elsewhere in v3: H contributes weakly.
    return sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H") - sequence.count("D") - sequence.count("E")


def split_semicolon(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def source_set(value: object) -> set[str]:
    return {item.upper() for item in split_semicolon(value)}


def contains_marker(value: object, markers: Iterable[str]) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in markers)


def source_tier(row: pd.Series) -> str:
    names = source_set(row.get("source_names", ""))
    files = str(row.get("source_files", "")).lower()

    if contains_marker(row.get("source_names", ""), SUSPICIOUS_SOURCE_NAME_MARKERS) or contains_marker(files, SUSPICIOUS_PATH_MARKERS):
        return "exclude_review"

    if names & HIGH_CONFIDENCE_SOURCE_NAMES:
        return "tier1_curated_or_uniprot"
    if names & MEDIUM_CONFIDENCE_SOURCE_NAMES:
        return "tier2_benchmark_or_predicted"
    return "tier3_other_or_unknown"


def length_bin(length: int) -> str:
    if length < 8:
        return "lt8"
    if length <= 12:
        return "8_12"
    if length <= 20:
        return "13_20"
    if length <= 32:
        return "21_32"
    if length <= 50:
        return "33_50"
    if length <= 64:
        return "51_64"
    return "gt64"


def add_qc_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["sequence"] = out["sequence"].map(clean_sequence)
    out["length"] = out["sequence"].str.len()
    out["entropy"] = out["sequence"].map(sequence_entropy)
    out["longest_homopolymer"] = out["sequence"].map(longest_homopolymer)
    out["net_charge_KR_minus_DE"] = out["sequence"].map(approximate_net_charge)
    out["hydrophobic_fraction"] = out["sequence"].map(lambda seq: residue_fraction(seq, HYDROPHOBIC))
    out["aromatic_fraction"] = out["sequence"].map(lambda seq: residue_fraction(seq, AROMATIC))
    out["positive_fraction"] = out["sequence"].map(lambda seq: residue_fraction(seq, POSITIVE))
    out["negative_fraction"] = out["sequence"].map(lambda seq: residue_fraction(seq, NEGATIVE))
    out["cysteine_count"] = out["sequence"].str.count("C")
    out["tryptophan_count"] = out["sequence"].str.count("W")
    out["length_bin"] = out["length"].map(length_bin)

    out["n_source_names"] = out["source_names"].map(lambda value: len(source_set(value))) if "source_names" in out.columns else 0
    out["has_high_conf_source"] = out.get("source_names", pd.Series("", index=out.index)).map(
        lambda value: bool(source_set(value) & HIGH_CONFIDENCE_SOURCE_NAMES)
    )
    out["has_preferred_training_source"] = out.get("source_names", pd.Series("", index=out.index)).map(
        lambda value: bool(source_set(value) & PREFERRED_TRAINING_SOURCE_NAMES)
    )
    out["suspicious_source_name"] = out.get("source_names", pd.Series("", index=out.index)).map(
        lambda value: contains_marker(value, SUSPICIOUS_SOURCE_NAME_MARKERS)
    )
    out["suspicious_source_file"] = out.get("source_files", pd.Series("", index=out.index)).map(
        lambda value: contains_marker(value, SUSPICIOUS_PATH_MARKERS)
    )
    out["qc_tier"] = out.apply(source_tier, axis=1)

    out["qc_flag_short_8_12"] = out["length"].between(8, 12)
    out["qc_flag_very_short"] = out["length"] < args.min_train_length
    out["qc_flag_low_entropy"] = out["entropy"] < args.min_entropy
    out["qc_flag_homopolymer"] = out["longest_homopolymer"] > args.max_homopolymer
    out["qc_flag_hydrophobic_extreme"] = out["hydrophobic_fraction"] > args.max_hydrophobic_fraction
    out["qc_flag_charge_extreme"] = out["net_charge_KR_minus_DE"].abs() > args.max_abs_charge
    out["qc_flag_cys_extreme"] = out["cysteine_count"] > args.max_cysteines
    out["qc_flag_trp_extreme"] = out["tryptophan_count"] > args.max_tryptophans
    out["qc_flag_suspicious_source"] = out["suspicious_source_name"] | out["suspicious_source_file"] | (out["qc_tier"] == "exclude_review")

    out["passes_trainable_core"] = (
        out["length"].between(args.min_train_length, args.max_train_length)
        & (out["entropy"] >= args.min_entropy)
        & (out["longest_homopolymer"] <= args.max_homopolymer)
        & (out["hydrophobic_fraction"] <= args.max_hydrophobic_fraction)
        & (out["net_charge_KR_minus_DE"].abs() <= args.max_abs_charge)
        & (out["cysteine_count"] <= args.max_cysteines)
        & (out["tryptophan_count"] <= args.max_tryptophans)
        & (~out["qc_flag_suspicious_source"])
        & (out["has_preferred_training_source"] | (out["qc_tier"] == "tier2_benchmark_or_predicted"))
    )

    return out


def explode_sources(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        names = split_semicolon(row.get("source_names", "")) or ["UNKNOWN"]
        for name in names:
            rows.append(
                {
                    "source_name": name,
                    "sequence": row["sequence"],
                    "length": row["length"],
                    "qc_tier": row["qc_tier"],
                    "passes_trainable_core": bool(row["passes_trainable_core"]),
                    "suspicious_source_file": bool(row["suspicious_source_file"]),
                    "suspicious_source_name": bool(row["suspicious_source_name"]),
                    "n_source_records": row.get("n_source_records", np.nan),
                }
            )
    return pd.DataFrame(rows)


def amino_acid_composition(df: pd.DataFrame) -> pd.DataFrame:
    total = Counter()
    for sequence in df["sequence"].dropna().astype(str):
        total.update(sequence)
    denom = sum(total.values()) or 1
    return pd.DataFrame(
        {
            "residue": sorted(AMINO_ACIDS),
            "count": [int(total.get(residue, 0)) for residue in sorted(AMINO_ACIDS)],
            "fraction": [float(total.get(residue, 0) / denom) for residue in sorted(AMINO_ACIDS)],
        }
    )


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            peptide_id = row.get("peptide_id", row.get("sequence", "sequence"))
            header = (
                f">{peptide_id}"
                f"|tier={row.get('qc_tier', 'NA')}"
                f"|len={int(row['length'])}"
                f"|charge={float(row['net_charge_KR_minus_DE']):.1f}"
                f"|hydro={float(row['hydrophobic_fraction']):.3f}"
            )
            handle.write(header + "\n")
            handle.write(str(row["sequence"]) + "\n")


def maybe_plot(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """Make simple PNG plots if matplotlib is installed. Return created paths."""
    created: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return created

    # Length histogram.
    fig = plt.figure(figsize=(8, 5))
    df["length"].hist(bins=range(8, 66, 2))
    plt.xlabel("Peptide length")
    plt.ylabel("Number of unique sequences")
    plt.title("AMP-JEPA v3 upscaled corpus length distribution")
    path = output_dir / "length_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    created.append(str(path))

    # Tier counts.
    fig = plt.figure(figsize=(9, 5))
    df["qc_tier"].value_counts().plot(kind="bar")
    plt.xlabel("QC tier")
    plt.ylabel("Number of unique sequences")
    plt.title("QC tier counts")
    path = output_dir / "qc_tier_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    created.append(str(path))

    # Hydrophobicity vs charge sample to avoid huge scatter.
    sample = df.sample(min(len(df), 50000), random_state=7) if len(df) > 50000 else df
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(sample["hydrophobic_fraction"], sample["net_charge_KR_minus_DE"], s=2, alpha=0.2)
    plt.xlabel("Hydrophobic fraction")
    plt.ylabel("Approximate net charge")
    plt.title("Hydrophobicity vs charge")
    path = output_dir / "hydrophobicity_vs_charge.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    created.append(str(path))

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="v3/data/processed/upscaled_peptide_corpus_v3.csv")
    parser.add_argument("--output-dir", default="v3/results/upscaled_corpus_qc")
    parser.add_argument("--min-train-length", type=int, default=10)
    parser.add_argument("--max-train-length", type=int, default=50)
    parser.add_argument("--min-entropy", type=float, default=1.5)
    parser.add_argument("--max-homopolymer", type=int, default=6)
    parser.add_argument("--max-hydrophobic-fraction", type=float, default=0.70)
    parser.add_argument("--max-abs-charge", type=float, default=15.0)
    parser.add_argument("--max-cysteines", type=int, default=8)
    parser.add_argument("--max-tryptophans", type=int, default=6)
    parser.add_argument("--top-examples", type=int, default=25)
    args = parser.parse_args()

    corpus_path = resolve_path(args.corpus)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise SystemExit(f"[ERROR] Corpus not found: {corpus_path}")

    df = pd.read_csv(corpus_path, low_memory=False)
    if "sequence" not in df.columns:
        raise SystemExit(f"[ERROR] Corpus lacks required 'sequence' column: {corpus_path}")

    qc = add_qc_features(df, args)
    core = qc.loc[qc["passes_trainable_core"]].copy()
    review = qc.loc[~qc["passes_trainable_core"]].copy()

    # Outputs.
    qc_table_path = output_dir / "upscaled_corpus_qc_table.csv"
    core_csv_path = output_dir / "upscaled_corpus_trainable_core.csv"
    core_fasta_path = output_dir / "upscaled_corpus_trainable_core.fasta"
    review_csv_path = output_dir / "upscaled_corpus_review_or_excluded.csv"
    source_summary_path = output_dir / "source_qc_summary.csv"
    length_summary_path = output_dir / "length_bin_summary.csv"
    aa_all_path = output_dir / "amino_acid_composition_all.csv"
    aa_core_path = output_dir / "amino_acid_composition_trainable_core.csv"
    duplicates_path = output_dir / "top_duplicate_sequences.csv"
    report_path = output_dir / "upscaled_corpus_qc_report.json"
    markdown_path = output_dir / "upscaled_corpus_qc_report.md"

    qc.to_csv(qc_table_path, index=False)
    core.to_csv(core_csv_path, index=False)
    review.to_csv(review_csv_path, index=False)
    write_fasta(core, core_fasta_path)

    source_long = explode_sources(qc)
    source_summary = (
        source_long.groupby("source_name", as_index=False)
        .agg(
            unique_sequences=("sequence", "nunique"),
            median_length=("length", "median"),
            trainable_core_sequences=("passes_trainable_core", "sum"),
            suspicious_file_hits=("suspicious_source_file", "sum"),
            suspicious_name_hits=("suspicious_source_name", "sum"),
        )
        .sort_values("unique_sequences", ascending=False)
    )
    source_summary["trainable_core_fraction"] = source_summary["trainable_core_sequences"] / source_summary["unique_sequences"].replace(0, np.nan)
    source_summary.to_csv(source_summary_path, index=False)

    length_summary = (
        qc.groupby("length_bin", as_index=False)
        .agg(
            unique_sequences=("sequence", "count"),
            trainable_core_sequences=("passes_trainable_core", "sum"),
            median_charge=("net_charge_KR_minus_DE", "median"),
            median_hydrophobic_fraction=("hydrophobic_fraction", "median"),
        )
    )
    length_summary.to_csv(length_summary_path, index=False)

    amino_acid_composition(qc).to_csv(aa_all_path, index=False)
    amino_acid_composition(core).to_csv(aa_core_path, index=False)

    top_dupes = qc.sort_values("n_source_records", ascending=False).head(args.top_examples) if "n_source_records" in qc.columns else qc.head(0)
    top_dupes.to_csv(duplicates_path, index=False)

    plot_paths = maybe_plot(qc, output_dir)

    flag_columns = [column for column in qc.columns if column.startswith("qc_flag_")]
    flag_counts = {column: int(qc[column].sum()) for column in flag_columns}
    tier_counts = {str(key): int(value) for key, value in qc["qc_tier"].value_counts(dropna=False).items()}
    source_top = source_summary.head(20).to_dict(orient="records")

    report = {
        "input_corpus": str(corpus_path),
        "n_unique_sequences": int(len(qc)),
        "n_trainable_core": int(len(core)),
        "trainable_core_fraction": float(len(core) / max(len(qc), 1)),
        "n_review_or_excluded": int(len(review)),
        "length": {
            "min": int(qc["length"].min()),
            "median": float(qc["length"].median()),
            "mean": float(qc["length"].mean()),
            "max": int(qc["length"].max()),
        },
        "trainable_core_length": {
            "min": int(core["length"].min()) if len(core) else None,
            "median": float(core["length"].median()) if len(core) else None,
            "mean": float(core["length"].mean()) if len(core) else None,
            "max": int(core["length"].max()) if len(core) else None,
        },
        "tier_counts": tier_counts,
        "flag_counts": flag_counts,
        "top_sources": source_top,
        "outputs": {
            "qc_table": str(qc_table_path),
            "trainable_core_csv": str(core_csv_path),
            "trainable_core_fasta": str(core_fasta_path),
            "review_or_excluded_csv": str(review_csv_path),
            "source_summary": str(source_summary_path),
            "length_summary": str(length_summary_path),
            "amino_acid_composition_all": str(aa_all_path),
            "amino_acid_composition_trainable_core": str(aa_core_path),
            "top_duplicate_sequences": str(duplicates_path),
            "plots": plot_paths,
        },
        "recommendation": "Use trainable_core_fasta for the next cleaner expanded-corpus v3 run; keep the full corpus for audit and ablation only.",
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = [
        "# AMP-JEPA v3 upscaled corpus QC report",
        "",
        f"Input corpus: `{corpus_path}`",
        "",
        "## Summary",
        "",
        f"- Unique sequences: **{len(qc):,}**",
        f"- Trainable core sequences: **{len(core):,}** ({100 * len(core) / max(len(qc), 1):.1f}%)",
        f"- Review/excluded sequences: **{len(review):,}**",
        f"- Length median / mean / max: **{qc['length'].median():.1f} / {qc['length'].mean():.1f} / {qc['length'].max()}**",
        "",
        "## QC tiers",
        "",
    ]
    for tier, count in tier_counts.items():
        markdown.append(f"- `{tier}`: {count:,}")
    markdown.extend(["", "## Flag counts", ""])
    for flag, count in flag_counts.items():
        markdown.append(f"- `{flag}`: {count:,}")
    markdown.extend(["", "## Main outputs", ""])
    for key, value in report["outputs"].items():
        markdown.append(f"- `{key}`: `{value}`")
    markdown.append("")
    markdown.append("## Recommendation")
    markdown.append("")
    markdown.append(report["recommendation"])
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print("\n=== UPSCALED CORPUS QC SUMMARY ===")
    print(f"Input corpus: {corpus_path}")
    print(f"Unique sequences: {len(qc):,}")
    print(f"Trainable core: {len(core):,} ({100 * len(core) / max(len(qc), 1):.1f}%)")
    print(f"Review/excluded: {len(review):,}")
    print(f"Length median/mean/max: {qc['length'].median():.1f}/{qc['length'].mean():.1f}/{qc['length'].max()}")
    print("\nQC tiers:")
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,}")
    print("\nTop flags:")
    for flag, count in sorted(flag_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {flag}: {count:,}")
    print("\nOutputs:")
    print(f"  {qc_table_path}")
    print(f"  {core_csv_path}")
    print(f"  {core_fasta_path}")
    print(f"  {review_csv_path}")
    print(f"  {source_summary_path}")
    print(f"  {report_path}")
    print(f"  {markdown_path}")


if __name__ == "__main__":
    main()

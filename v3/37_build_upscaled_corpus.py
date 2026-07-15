#!/usr/bin/env python3
"""Build an upscaled, deduplicated AMP corpus for AMP-JEPA-Hybrid v3.

This script is a stronger corpus-preparation layer than 00_prepare_corpus.py.
It is designed for combining APD, dbAMP, DRAMP, CAMPR, DBAASP exports, and
other local AMP FASTA/CSV/TSV files without losing source provenance.

Inputs can be FASTA, CSV, TSV, or TAB files. CSV/TSV files must contain one of:
sequence, Sequence, PeptideSequence, peptide_sequence, aa_sequence, seq.

Outputs
-------
upscaled_peptide_corpus_v3.csv
upscaled_peptide_corpus_v3.fasta
upscaled_corpus_source_summary.csv
upscaled_corpus_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWYC")
SEQ_COLUMNS = [
    "sequence",
    "Sequence",
    "PeptideSequence",
    "peptide_sequence",
    "aa_sequence",
    "seq",
    "Seq",
]
ID_COLUMNS = [
    "id",
    "ID",
    "name",
    "Name",
    "accession",
    "Accession",
    "peptide_id",
    "Peptide_ID",
    "DRAMP_ID",
    "APD_ID",
    "DBAASP_ID",
    "Unnamed: 0",
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def clean_sequence(value: object) -> str:
    sequence = str(value).strip().upper()
    return "".join(residue for residue in sequence if residue in AMINO_ACIDS)


def source_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "apd" in stem:
        return "APD"
    if "dbamp" in stem:
        return "dbAMP"
    if "dramp" in stem:
        return "DRAMP"
    if "campr" in stem or "camp" in stem:
        return "CAMPR"
    if "dbaasp" in stem:
        return "DBAASP"
    if "starpep" in stem:
        return "StarPep"
    if "uniprot" in stem:
        return "UniProt"
    return path.stem


def read_fasta(path: Path, source_label: str) -> pd.DataFrame:
    rows = []
    header = None
    chunks: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    rows.append(
                        {
                            "source_id": header.split()[0] or f"{source_label}_{len(rows) + 1}",
                            "source_header": header,
                            "sequence": clean_sequence("".join(chunks)),
                            "source_name": source_label,
                            "source_file": str(path),
                        }
                    )
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

        if header is not None:
            rows.append(
                {
                    "source_id": header.split()[0] or f"{source_label}_{len(rows) + 1}",
                    "source_header": header,
                    "sequence": clean_sequence("".join(chunks)),
                    "source_name": source_label,
                    "source_file": str(path),
                }
            )

    return pd.DataFrame(rows)


def read_table(path: Path, source_label: str) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    seq_col = next((column for column in SEQ_COLUMNS if column in df.columns), None)
    if seq_col is None:
        raise ValueError(f"No sequence column found in {path}; tried {SEQ_COLUMNS}")

    id_col = next((column for column in ID_COLUMNS if column in df.columns), None)

    out = pd.DataFrame()
    out["source_id"] = df[id_col].astype(str) if id_col else [f"{source_label}_{i + 1}" for i in range(len(df))]
    out["source_header"] = out["source_id"]
    out["sequence"] = df[seq_col].map(clean_sequence)
    out["source_name"] = source_label
    out["source_file"] = str(path)

    return out


def read_input(path: Path, source_label: str | None = None) -> pd.DataFrame:
    source_label = source_label or source_name_from_path(path)
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fna", ".txt"}:
        return read_fasta(path, source_label)
    if suffix in {".csv", ".tsv", ".tab"}:
        return read_table(path, source_label)
    raise ValueError(f"Unsupported input file type: {path}")


def approximate_net_charge(sequence: str) -> float:
    return sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H") - sequence.count("D") - sequence.count("E")


def residue_fraction(sequence: str, residues: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(residue in residues for residue in sequence) / len(sequence)


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


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["length"] = out["sequence"].str.len()
    out["net_charge_KR_minus_DE"] = out["sequence"].map(approximate_net_charge)
    out["hydrophobic_fraction"] = out["sequence"].map(lambda seq: residue_fraction(seq, HYDROPHOBIC))
    out["cysteine_count"] = out["sequence"].str.count("C")
    out["tryptophan_count"] = out["sequence"].str.count("W")
    out["entropy"] = out["sequence"].map(sequence_entropy)
    out["longest_homopolymer"] = out["sequence"].map(longest_homopolymer)
    return out


def expand_inputs(paths: Iterable[str]) -> List[Path]:
    expanded: List[Path] = []
    for raw_path in paths:
        path = resolve_path(raw_path)
        if path.is_dir():
            for suffix in ["*.fasta", "*.fa", "*.faa", "*.csv", "*.tsv", "*.tab", "*.txt"]:
                expanded.extend(sorted(path.glob(suffix)))
        else:
            expanded.extend(sorted(path.parent.glob(path.name)))
    # Preserve order but remove duplicates.
    seen = set()
    unique = []
    for path in expanded:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            header = (
                f">{row['peptide_id']}"
                f"|sources={row['source_names']}"
                f"|n_records={int(row['n_source_records'])}"
                f"|len={int(row['length'])}"
            )
            handle.write(header + "\n")
            handle.write(str(row["sequence"]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Files, globs, or directories containing FASTA/CSV/TSV corpus sources.")
    parser.add_argument("--output-prefix", default="v3/data/processed/upscaled_peptide_corpus_v3")
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--max-homopolymer", type=int, default=6)
    parser.add_argument("--min-entropy", type=float, default=1.5)
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("[ERROR] No input files matched.")

    frames = []
    read_errors = []

    for path in input_paths:
        if not path.exists():
            read_errors.append({"path": str(path), "error": "missing"})
            continue
        try:
            df = read_input(path)
        except Exception as exc:
            read_errors.append({"path": str(path), "error": str(exc)})
            print(f"[WARN] Skipping {path}: {exc}")
            continue
        frames.append(df)
        print(f"[INFO] Read {len(df):,} rows from {path} as {df['source_name'].iloc[0] if len(df) else source_name_from_path(path)}")

    if not frames:
        message = "[ERROR] No readable corpus sources were loaded."
        if args.fail_on_empty:
            raise SystemExit(message)
        print(message)
        return

    raw = pd.concat(frames, ignore_index=True)
    raw["sequence"] = raw["sequence"].map(clean_sequence)
    raw["raw_length"] = raw["sequence"].str.len()
    raw["valid_canonical"] = raw["sequence"].map(lambda seq: bool(seq) and all(residue in AMINO_ACIDS for residue in seq))
    raw["valid_length"] = raw["raw_length"].between(args.min_len, args.max_len)
    raw["valid"] = raw["valid_canonical"] & raw["valid_length"]

    valid = raw.loc[raw["valid"]].copy()
    valid = add_sequence_features(valid)
    valid["valid_complexity"] = (valid["entropy"] >= args.min_entropy) & (valid["longest_homopolymer"] <= args.max_homopolymer)
    valid = valid.loc[valid["valid_complexity"]].copy()

    if valid.empty:
        raise SystemExit("[ERROR] No valid sequences survived filtering.")

    corpus = (
        valid.groupby("sequence", as_index=False)
        .agg(
            n_source_records=("source_id", "count"),
            source_ids=("source_id", lambda values: ";".join(sorted(set(map(str, values)))[:250])),
            source_names=("source_name", lambda values: ";".join(sorted(set(map(str, values))))),
            source_files=("source_file", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .sort_values(["n_source_records", "sequence"], ascending=[False, True])
        .reset_index(drop=True)
    )

    corpus.insert(0, "peptide_id", [f"v3_upscaled_{index:08d}" for index in range(1, len(corpus) + 1)])
    corpus = add_sequence_features(corpus)

    source_summary = (
        raw.groupby("source_name", as_index=False)
        .agg(
            raw_rows=("sequence", "count"),
            valid_length_rows=("valid_length", "sum"),
            valid_rows=("valid", "sum"),
            unique_valid_sequences=("sequence", lambda values: len(set(seq for seq in values if seq))),
        )
        .sort_values("raw_rows", ascending=False)
    )

    output_prefix = resolve_path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    fasta_path = output_prefix.with_suffix(".fasta")
    source_summary_path = output_prefix.parent / f"{output_prefix.name}_source_summary.csv"
    report_path = output_prefix.parent / f"{output_prefix.name}_report.json"

    corpus.to_csv(csv_path, index=False)
    write_fasta(corpus, fasta_path)
    source_summary.to_csv(source_summary_path, index=False)

    report = {
        "input_files": [str(path) for path in input_paths],
        "read_errors": read_errors,
        "raw_rows": int(len(raw)),
        "valid_rows_after_length_and_canonical_filters": int(raw["valid"].sum()),
        "valid_rows_after_complexity_filters": int(len(valid)),
        "unique_sequences": int(len(corpus)),
        "min_length": int(corpus["length"].min()),
        "median_length": float(corpus["length"].median()),
        "max_length": int(corpus["length"].max()),
        "mean_charge": float(corpus["net_charge_KR_minus_DE"].mean()),
        "mean_hydrophobic_fraction": float(corpus["hydrophobic_fraction"].mean()),
        "outputs": {
            "csv": str(csv_path),
            "fasta": str(fasta_path),
            "source_summary": str(source_summary_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== UPSCALED CORPUS SUMMARY ===")
    print(f"Raw rows: {len(raw):,}")
    print(f"Valid rows after filters: {len(valid):,}")
    print(f"Unique sequences: {len(corpus):,}")
    print(f"Length: min={report['min_length']} median={report['median_length']:.1f} max={report['max_length']}")
    print("\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {fasta_path}")
    print(f"  {source_summary_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()

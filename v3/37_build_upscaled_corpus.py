#!/usr/bin/env python3
"""Build an upscaled, deduplicated AMP corpus for AMP-JEPA-Hybrid v3.

This script is a stronger corpus-preparation layer than 00_prepare_corpus.py.
It is designed for combining APD, dbAMP, DRAMP, CAMPR, DBAASP exports,
UniProt FASTA exports, public benchmark repositories, and other local AMP
FASTA/CSV/TSV files without losing source provenance.

Inputs can be FASTA, CSV, TSV, TAB, or TXT files. For tables, the script first
looks for known sequence-column names and then falls back to automatic peptide
column detection. This matters because many public AMP benchmark repos use
idiosyncratic column names.

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
    "sequences",
    "Sequences",
    "PeptideSequence",
    "peptide_sequence",
    "peptide_seq",
    "peptide",
    "Peptide",
    "peptides",
    "aa_sequence",
    "amino_acid_sequence",
    "aminoacid_sequence",
    "protein_sequence",
    "protein_seq",
    "seq",
    "Seq",
    "SEQ",
]
ID_COLUMNS = [
    "id",
    "ID",
    "name",
    "Name",
    "accession",
    "Accession",
    "entry",
    "Entry",
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
    text = str(path).lower()
    if "apd" in text:
        return "APD"
    if "dbamp" in text:
        return "dbAMP"
    if "dramp" in text:
        return "DRAMP"
    if "campr" in text or "camp" in text:
        return "CAMPR"
    if "dbaasp" in text:
        return "DBAASP"
    if "starpep" in text:
        return "StarPep"
    if "uniprot" in text:
        return "UniProt"
    if "amplify" in text:
        return "AMPlify"
    if "ampcliff" in text:
        return "AMPCliff"
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

    # Some public TXT files are one raw peptide per line rather than FASTA.
    if not rows and path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for i, raw_line in enumerate(handle, start=1):
                seq = clean_sequence(raw_line)
                if seq:
                    rows.append(
                        {
                            "source_id": f"{source_label}_{i}",
                            "source_header": f"{source_label}_{i}",
                            "sequence": seq,
                            "source_name": source_label,
                            "source_file": str(path),
                        }
                    )

    return pd.DataFrame(rows)


def peptide_like_fraction(values: pd.Series, min_len: int = 8, max_len: int = 100) -> float:
    if values.empty:
        return 0.0
    sample = values.dropna().astype(str).head(500)
    if sample.empty:
        return 0.0
    cleaned = sample.map(clean_sequence)
    ok = cleaned.map(lambda seq: min_len <= len(seq) <= max_len)
    # Penalize columns where cleaning removed a lot of text, e.g. descriptions.
    retention = cleaned.str.len() / sample.str.len().replace(0, 1)
    ok = ok & (retention >= 0.70)
    return float(ok.mean())


def infer_sequence_column(df: pd.DataFrame) -> str | None:
    for column in SEQ_COLUMNS:
        if column in df.columns:
            return column

    candidates = []
    for column in df.columns:
        if df[column].dtype == object or str(df[column].dtype).startswith("string"):
            frac = peptide_like_fraction(df[column])
            if frac >= 0.50:
                candidates.append((frac, str(column)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def sniff_table(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", low_memory=False), "\t"

    # Try comma, tab, and semicolon because public benchmark exports are messy.
    last_error = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1 or sep == ",":
                return df, sep
        except Exception as exc:  # pragma: no cover - file dependent
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path, low_memory=False), ","


def read_table(path: Path, source_label: str) -> pd.DataFrame:
    df, _sep = sniff_table(path)

    seq_col = infer_sequence_column(df)
    if seq_col is None:
        raise ValueError(f"No peptide-like sequence column found in {path}; tried known columns and automatic inference")

    id_col = next((column for column in ID_COLUMNS if column in df.columns), None)

    out = pd.DataFrame()
    out["source_id"] = df[id_col].astype(str) if id_col else [f"{source_label}_{i + 1}" for i in range(len(df))]
    out["source_header"] = out["source_id"]
    out["sequence"] = df[seq_col].map(clean_sequence)
    out["source_name"] = source_label
    out["source_file"] = str(path)
    out["source_sequence_column"] = str(seq_col)

    return out


def read_input(path: Path, source_label: str | None = None) -> pd.DataFrame:
    source_label = source_label or source_name_from_path(path)
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fna"}:
        return read_fasta(path, source_label)
    if suffix in {".csv", ".tsv", ".tab"}:
        return read_table(path, source_label)
    if suffix == ".txt":
        # Try FASTA/raw-line first, then table fallback.
        fasta_df = read_fasta(path, source_label)
        if not fasta_df.empty:
            return fasta_df
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
            for suffix in ["*.fasta", "*.fa", "*.faa", "*.fna", "*.csv", "*.tsv", "*.tab", "*.txt"]:
                expanded.extend(sorted(path.rglob(suffix)))
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
        label = df["source_name"].iloc[0] if len(df) else source_name_from_path(path)
        seq_col = df.get("source_sequence_column", pd.Series(["FASTA/raw"])).iloc[0] if len(df) else "NA"
        print(f"[INFO] Read {len(df):,} rows from {path} as {label} using {seq_col}")

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

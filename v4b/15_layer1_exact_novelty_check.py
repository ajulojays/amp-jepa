#!/usr/bin/env python3
"""Layer 1 exact-sequence novelty screen for V4B priority peptides.

This script checks whether AMP-JEPA V4B priority candidates have exact sequence
matches in one or more known/reference sequence collections.

It is deliberately conservative and only answers Layer 1:
    "Is this exact peptide sequence already present in a supplied reference set?"

It does not assess near-neighbor similarity, obvious variants, patent claim
scope, or legal patentability. Those belong to later layers.

Reference inputs can be FASTA, CSV/TSV/TXT, or plain one-sequence-per-line text.
Pass references as repeated name:path pairs, for example:

    --reference APD:data/reference/APD2024.fasta \
    --reference DRAMP:data/reference/dramp.csv \
    --reference training:v4b/results/generation_00/generation_00_candidates.csv

For CSV/TSV files, the script tries common sequence columns first and then scans
all object columns for canonical peptide-like strings.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
COMMON_SEQUENCE_COLUMNS = [
    "sequence",
    "Sequence",
    "SEQUENCE",
    "seq",
    "Seq",
    "peptide",
    "Peptide",
    "peptide_sequence",
    "aa_sequence",
    "amino_acid_sequence",
]


def clean_sequence(value: object) -> str:
    """Normalize a peptide sequence for exact matching."""
    if value is None:
        return ""
    seq = str(value).strip().upper()
    # Remove whitespace and common separators; keep only letters for exact AA matching.
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("-", "").replace("_", "")
    return seq


def is_canonical_peptide(seq: str, min_len: int = 2, max_len: int = 200) -> bool:
    return bool(seq) and min_len <= len(seq) <= max_len and all(aa in CANONICAL for aa in seq)


def read_fasta(path: Path) -> list[dict]:
    records: list[dict] = []
    header = ""
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if not chunks:
            return
        seq = clean_sequence("".join(chunks))
        if is_canonical_peptide(seq):
            records.append({"reference_id": header, "reference_sequence": seq})
        chunks = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                chunks.append(line)
        flush()
    return records


def read_delimited(path: Path, sequence_columns: list[str]) -> list[dict]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else None
    try:
        df = pd.read_csv(path, sep=sep, engine="python", low_memory=False)
    except Exception:
        # Fall back to plain lines below.
        return []

    records: list[dict] = []
    candidate_cols = [c for c in sequence_columns if c in df.columns]
    if not candidate_cols:
        candidate_cols = [c for c in COMMON_SEQUENCE_COLUMNS if c in df.columns]

    # If there is no obvious column, scan object-like columns but only accept
    # clean canonical peptide-looking values.
    if not candidate_cols:
        candidate_cols = [c for c in df.columns if df[c].dtype == "object"]

    for col in candidate_cols:
        for idx, value in df[col].dropna().items():
            seq = clean_sequence(value)
            if is_canonical_peptide(seq):
                rid = ""
                for id_col in ["id", "ID", "name", "Name", "accession", "Accession", "header", "Header"]:
                    if id_col in df.columns and pd.notna(df.loc[idx, id_col]):
                        rid = str(df.loc[idx, id_col])
                        break
                if not rid:
                    rid = f"{path.name}:{col}:{idx}"
                records.append({"reference_id": rid, "reference_sequence": seq})
    return records


def read_plain_sequences(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">"): 
                continue
            # Accept either a bare sequence or the first token in a simple table-like line.
            tokens = re.split(r"[\s,;]+", line)
            for token in tokens[:3]:
                seq = clean_sequence(token)
                if is_canonical_peptide(seq):
                    records.append({"reference_id": f"{path.name}:line{idx}", "reference_sequence": seq})
                    break
    return records


def load_reference(path: Path, sequence_columns: list[str]) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fna", ".fas"}:
        return read_fasta(path)

    records = []
    if suffix in {".csv", ".tsv", ".tab"}:
        records = read_delimited(path, sequence_columns)
    if not records:
        # FASTA files with nonstandard extension still work here if they contain headers.
        text_head = path.read_text(encoding="utf-8", errors="ignore")[:1000]
        if ">" in text_head and "\n" in text_head:
            records = read_fasta(path)
        if not records:
            records = read_plain_sequences(path)
    return records


def parse_reference_arg(item: str) -> tuple[str, Path]:
    if ":" not in item:
        path = Path(item)
        return path.stem, path
    name, path_text = item.split(":", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if not name:
        name = path.stem
    return name, path


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "sequence" not in df.columns:
        raise ValueError(f"Candidate file must contain a 'sequence' column: {path}")
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"candidate_{i+1}" for i in range(len(df))]
    df["sequence_clean"] = df["sequence"].map(clean_sequence)
    df["candidate_sequence_valid_for_exact_screen"] = df["sequence_clean"].map(lambda s: is_canonical_peptide(s, min_len=1, max_len=500))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="v4b/results/elite_pareto_selection/v4b_elite_pareto_union_tiered.csv",
        help="Candidate CSV, usually the frozen Tier 1-4 priority universe.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Reference as name:path. Repeat for APD, DRAMP, dbAMP, DBAASP, training, patents, etc.",
    )
    parser.add_argument(
        "--reference-dir",
        action="append",
        default=[],
        help="Optional directory of reference files to scan. File stem is used as reference name.",
    )
    parser.add_argument(
        "--reference-glob",
        default="*.fa,*.fasta,*.faa,*.csv,*.tsv,*.txt",
        help="Comma-separated glob patterns used inside --reference-dir.",
    )
    parser.add_argument(
        "--sequence-column",
        action="append",
        default=[],
        help="Additional sequence column name to prioritize for CSV/TSV references.",
    )
    parser.add_argument("--output-dir", default="v4b/results/novelty_layer1_exact")
    parser.add_argument("--fail-on-missing-reference", action="store_true")
    args = parser.parse_args()

    candidate_path = Path(args.candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(candidate_path)

    reference_specs: list[tuple[str, Path]] = []
    for item in args.reference:
        reference_specs.append(parse_reference_arg(item))

    patterns = [p.strip() for p in args.reference_glob.split(",") if p.strip()]
    for d in args.reference_dir:
        ref_dir = Path(d)
        if not ref_dir.exists():
            msg = f"Reference directory does not exist: {ref_dir}"
            if args.fail_on_missing_reference:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}")
            continue
        seen_paths: set[Path] = set()
        for pattern in patterns:
            for path in sorted(ref_dir.glob(pattern)):
                if path.is_file() and path not in seen_paths:
                    seen_paths.add(path)
                    reference_specs.append((path.stem, path))

    if not reference_specs:
        raise SystemExit(
            "No references were provided. Add --reference name:path or --reference-dir path. "
            "Layer 1 requires known/training/reference sequences to compare against."
        )

    sequence_columns = args.sequence_column + COMMON_SEQUENCE_COLUMNS

    reference_rows: list[dict] = []
    missing_references: list[str] = []
    for ref_name, ref_path in reference_specs:
        if not ref_path.exists():
            msg = f"{ref_name}:{ref_path}"
            missing_references.append(msg)
            if args.fail_on_missing_reference:
                raise FileNotFoundError(msg)
            print(f"[WARN] Missing reference skipped: {msg}")
            continue

        records = load_reference(ref_path, sequence_columns=sequence_columns)
        if not records:
            print(f"[WARN] No canonical peptide sequences parsed from {ref_name}: {ref_path}")
            continue
        for rec in records:
            reference_rows.append(
                {
                    "reference_name": ref_name,
                    "reference_path": str(ref_path),
                    "reference_id": rec.get("reference_id", ""),
                    "reference_sequence": rec["reference_sequence"],
                }
            )
        print(f"[OK] {ref_name}: parsed {len(records):,} sequence records from {ref_path}")

    if not reference_rows:
        raise SystemExit("No usable reference sequences were loaded.")

    refs = pd.DataFrame(reference_rows)
    refs["reference_sequence_clean"] = refs["reference_sequence"].map(clean_sequence)
    refs = refs[refs["reference_sequence_clean"].map(lambda s: is_canonical_peptide(s, min_len=1, max_len=500))].copy()

    ref_inventory = (
        refs.groupby("reference_name")
        .agg(
            reference_records=("reference_sequence_clean", "count"),
            unique_reference_sequences=("reference_sequence_clean", "nunique"),
            min_length=("reference_sequence_clean", lambda s: min(map(len, s)) if len(s) else None),
            median_length=("reference_sequence_clean", lambda s: float(pd.Series([len(x) for x in s]).median()) if len(s) else None),
            max_length=("reference_sequence_clean", lambda s: max(map(len, s)) if len(s) else None),
        )
        .reset_index()
    )

    # Build exact sequence -> reference hits.
    ref_map: dict[str, list[dict]] = defaultdict(list)
    for row in refs.itertuples(index=False):
        ref_map[row.reference_sequence_clean].append(
            {
                "reference_name": row.reference_name,
                "reference_id": row.reference_id,
                "reference_path": row.reference_path,
            }
        )

    long_hits: list[dict] = []
    wide_rows: list[dict] = []
    for idx, row in candidates.iterrows():
        seq = row["sequence_clean"]
        hits = ref_map.get(seq, [])
        if not hits:
            continue
        names = sorted({h["reference_name"] for h in hits})
        ids_by_ref = defaultdict(list)
        paths_by_ref = defaultdict(set)
        for h in hits:
            ids_by_ref[h["reference_name"]].append(str(h["reference_id"]))
            paths_by_ref[h["reference_name"]].add(str(h["reference_path"]))
            long_hits.append(
                {
                    "candidate_id": row.get("candidate_id", f"candidate_{idx+1}"),
                    "sequence": row.get("sequence", seq),
                    "sequence_clean": seq,
                    "reference_name": h["reference_name"],
                    "reference_id": h["reference_id"],
                    "reference_path": h["reference_path"],
                }
            )
        wide_rows.append(
            {
                "candidate_id": row.get("candidate_id", f"candidate_{idx+1}"),
                "sequence": row.get("sequence", seq),
                "sequence_clean": seq,
                "exact_match_reference_count": len(names),
                "exact_match_references": ";".join(names),
                "exact_match_reference_ids": json.dumps({k: sorted(set(v))[:25] for k, v in ids_by_ref.items()}),
                "exact_match_reference_paths": json.dumps({k: sorted(v) for k, v in paths_by_ref.items()}),
            }
        )

    wide_hits = pd.DataFrame(wide_rows)
    long_hits_df = pd.DataFrame(long_hits)

    candidates_out = candidates.copy()
    if wide_hits.empty:
        candidates_out["has_exact_reference_match"] = False
        candidates_out["exact_match_reference_count"] = 0
        candidates_out["exact_match_references"] = ""
        candidates_out["exact_match_reference_ids"] = "{}"
        candidates_out["exact_match_reference_paths"] = "{}"
    else:
        candidates_out = candidates_out.merge(
            wide_hits.drop(columns=["sequence", "sequence_clean"], errors="ignore"),
            on="candidate_id",
            how="left",
        )
        candidates_out["has_exact_reference_match"] = candidates_out["exact_match_reference_count"].notna()
        candidates_out["exact_match_reference_count"] = candidates_out["exact_match_reference_count"].fillna(0).astype(int)
        for col in ["exact_match_references", "exact_match_reference_ids", "exact_match_reference_paths"]:
            candidates_out[col] = candidates_out[col].fillna("" if col == "exact_match_references" else "{}")

    no_exact = candidates_out.loc[~candidates_out["has_exact_reference_match"]].copy()
    exact = candidates_out.loc[candidates_out["has_exact_reference_match"]].copy()

    # Preserve existing tier/rank ordering if present; otherwise put exact-novel first by APEX/composite.
    sort_cols = [c for c in ["union_rank", "tier_rank", "v4b_elite_composite_score", "APEX_median_MIC", "APEX_worst_MIC"] if c in candidates_out.columns]
    if sort_cols:
        ascending = []
        for c in sort_cols:
            ascending.append(False if c == "v4b_elite_composite_score" else True)
        no_exact = no_exact.sort_values(sort_cols, ascending=ascending, na_position="last")
        exact = exact.sort_values(sort_cols, ascending=ascending, na_position="last")

    # Summaries.
    tier_col = "lead_tier" if "lead_tier" in candidates_out.columns else None
    if tier_col:
        tier_summary = (
            candidates_out.groupby(tier_col)
            .agg(
                total_candidates=("candidate_id", "count"),
                exact_matches=("has_exact_reference_match", "sum"),
                no_exact_match=("has_exact_reference_match", lambda s: int((~s).sum())),
            )
            .reset_index()
        )
        tier_summary["fraction_exact_match"] = tier_summary["exact_matches"] / tier_summary["total_candidates"]
    else:
        tier_summary = pd.DataFrame()

    ref_match_summary = pd.DataFrame()
    if not long_hits_df.empty:
        ref_match_summary = (
            long_hits_df.groupby("reference_name")
            .agg(
                matched_candidate_rows=("candidate_id", "count"),
                unique_matched_candidates=("candidate_id", "nunique"),
                unique_matched_sequences=("sequence_clean", "nunique"),
            )
            .reset_index()
            .sort_values("unique_matched_sequences", ascending=False)
        )

    outputs = {
        "candidates_with_exact_flags": output_dir / "v4b_layer1_candidates_with_exact_match_flags.csv",
        "no_exact_match_candidates": output_dir / "v4b_layer1_no_exact_reference_match_candidates.csv",
        "exact_match_candidates": output_dir / "v4b_layer1_exact_reference_match_candidates.csv",
        "exact_matches_long": output_dir / "v4b_layer1_exact_matches_long.csv",
        "reference_inventory": output_dir / "v4b_layer1_reference_inventory.csv",
        "tier_summary": output_dir / "v4b_layer1_exact_match_summary_by_tier.csv",
        "reference_match_summary": output_dir / "v4b_layer1_exact_match_summary_by_reference.csv",
        "summary_json": output_dir / "v4b_layer1_exact_novelty_summary.json",
    }

    candidates_out.to_csv(outputs["candidates_with_exact_flags"], index=False)
    no_exact.to_csv(outputs["no_exact_match_candidates"], index=False)
    exact.to_csv(outputs["exact_match_candidates"], index=False)
    long_hits_df.to_csv(outputs["exact_matches_long"], index=False)
    ref_inventory.to_csv(outputs["reference_inventory"], index=False)
    tier_summary.to_csv(outputs["tier_summary"], index=False)
    ref_match_summary.to_csv(outputs["reference_match_summary"], index=False)

    payload = {
        "layer": "Layer 1 exact-sequence novelty",
        "candidate_file": str(candidate_path),
        "candidate_count": int(len(candidates_out)),
        "unique_candidate_sequences": int(candidates_out["sequence_clean"].nunique()),
        "reference_files_requested": [f"{name}:{path}" for name, path in reference_specs],
        "missing_references": missing_references,
        "loaded_reference_records": int(len(refs)),
        "loaded_unique_reference_sequences": int(refs["reference_sequence_clean"].nunique()),
        "candidates_with_exact_reference_match": int(candidates_out["has_exact_reference_match"].sum()),
        "candidates_with_no_exact_reference_match": int((~candidates_out["has_exact_reference_match"]).sum()),
        "fraction_with_exact_reference_match": float(candidates_out["has_exact_reference_match"].mean()),
        "important_note": "No exact match is not proof of patentability; it only passes exact-sequence Layer 1 novelty against the supplied references.",
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\nREFERENCE INVENTORY")
    print(ref_inventory.to_string(index=False))

    print("\nLAYER 1 EXACT NOVELTY SUMMARY")
    print(json.dumps(payload, indent=2, default=str))

    if not tier_summary.empty:
        print("\nEXACT MATCH SUMMARY BY TIER")
        print(tier_summary.to_string(index=False))

    if not ref_match_summary.empty:
        print("\nEXACT MATCH SUMMARY BY REFERENCE")
        print(ref_match_summary.to_string(index=False))

    print("\nTop candidates with NO exact reference match")
    show_cols = [
        "union_rank",
        "lead_tier",
        "tier_rank",
        "candidate_id",
        "sequence",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "criteria_hydrophobic_fraction",
        "criteria_charge",
        "criteria_length",
    ]
    show_cols = [c for c in show_cols if c in no_exact.columns]
    print(no_exact[show_cols].head(30).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

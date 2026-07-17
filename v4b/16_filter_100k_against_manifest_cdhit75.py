#!/usr/bin/env python3
"""Filter V4B 100k candidates against a peptide manifest at 75% identity.

This is the first near-novelty screen before deeper AMP database/patent searching.
It compares all generated/scored V4B candidates from generations 1..10 against a
reference peptide manifest and removes candidates that are >=75% identical to any
manifest sequence.

Recommended method uses cd-hit-2d, which is designed for comparing a second
sequence set against a first reference set:

    cd-hit-2d -i reference.fa -i2 candidates.fa -o kept.fa -c 0.75 -n 2

The output FASTA contains candidates that are not similar to the manifest at the
chosen identity threshold. This script prepares FASTA files, runs cd-hit-2d,
parses kept/removed candidates, and writes CSV/FASTA outputs.

Input candidates default to all scored V4B generations, giving 100,000 rows for
G01..G10 in the current workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).upper().split())


def safe_id(raw: object, prefix: str, i: int) -> str:
    text = str(raw) if raw is not None and str(raw).strip() else f"{prefix}_{i}"
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text.strip())
    return text[:180] or f"{prefix}_{i}"


def read_fasta(path: Path) -> pd.DataFrame:
    records = []
    header = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append({"record_id": header, "sequence": "".join(chunks)})
                header = line[1:].split()[0] or f"record_{len(records)+1}"
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            records.append({"record_id": header, "sequence": "".join(chunks)})
    return pd.DataFrame(records)


def infer_sep(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    return ","


def choose_sequence_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = [
        "sequence",
        "Sequence",
        "seq",
        "peptide_sequence",
        "Peptide sequence",
        "aa_sequence",
        "peptide",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a sequence column. Columns: {list(df.columns)}")


def read_sequence_table(path: Path, sequence_col: str | None = None, id_col: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fna"}:
        df = read_fasta(path)
        df["source_file"] = str(path)
        return df

    df = pd.read_csv(path, sep=infer_sep(path), low_memory=False)
    seq_col = choose_sequence_column(df, sequence_col)
    if id_col and id_col in df.columns:
        record_id = df[id_col].astype(str)
    elif "candidate_id" in df.columns:
        record_id = df["candidate_id"].astype(str)
    elif "record_id" in df.columns:
        record_id = df["record_id"].astype(str)
    elif "id" in df.columns:
        record_id = df["id"].astype(str)
    else:
        record_id = pd.Series([f"{path.stem}_{i+1}" for i in range(len(df))])

    out = df.copy()
    out["record_id"] = record_id
    out["sequence"] = out[seq_col].astype(str)
    out["source_file"] = str(path)
    return out


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if args.candidates:
        path = Path(args.candidates)
        df = read_sequence_table(path, sequence_col=args.candidate_sequence_col, id_col=args.candidate_id_col)
        if "candidate_id" not in df.columns:
            df["candidate_id"] = df["record_id"].astype(str)
        if "generation_source" not in df.columns and "generation" in df.columns:
            df["generation_source"] = df["generation"]
        pieces.append(df)
    else:
        results_dir = Path(args.results_dir)
        for g in range(args.start_generation, args.end_generation + 1):
            path = results_dir / f"generation_{g:02d}" / f"generation_{g:02d}_candidates_scored.csv"
            if not path.exists():
                print(f"[WARN] Missing generation file: {path}")
                continue
            df = pd.read_csv(path, low_memory=False)
            if "candidate_id" not in df.columns:
                df["candidate_id"] = [f"V4B_G{g:02d}_{i+1}" for i in range(len(df))]
            df["record_id"] = df["candidate_id"].astype(str)
            df["generation_source"] = g
            pieces.append(df)

    if not pieces:
        raise SystemExit("No candidate sequences loaded.")

    df = pd.concat(pieces, ignore_index=True, sort=False)
    seq_col = choose_sequence_column(df, args.candidate_sequence_col)
    df["sequence_clean"] = df[seq_col].map(clean_sequence)
    df = df[df["sequence_clean"].map(lambda s: bool(s) and all(a in CANONICAL_AA for a in s))].copy()

    # Keep one row per sequence for the 100k V4B set, preserving the best scoring row if scores exist.
    sort_cols = []
    ascending = []
    for col, asc in [
        ("v4b_elite_composite_score", False),
        ("ampjepa_master_score", False),
        ("APEX_median_MIC", True),
        ("APEX_worst_MIC", True),
        ("organisms_MIC_le_64", False),
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    df = df.drop_duplicates("sequence_clean", keep="first").reset_index(drop=True)
    return df


def load_reference(path: Path, sequence_col: str | None = None, id_col: str | None = None) -> pd.DataFrame:
    df = read_sequence_table(path, sequence_col=sequence_col, id_col=id_col)
    df["reference_id"] = [safe_id(x, "ref", i + 1) for i, x in enumerate(df["record_id"])]
    df["reference_sequence_clean"] = df["sequence"].map(clean_sequence)
    df = df[df["reference_sequence_clean"].map(lambda s: bool(s) and all(a in CANONICAL_AA for a in s))].copy()
    df = df.drop_duplicates("reference_sequence_clean", keep="first").reset_index(drop=True)
    return df


def write_fasta(df: pd.DataFrame, path: Path, id_col: str, seq_col: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, row in df.iterrows():
            rid = safe_id(row.get(id_col), "seq", int(i) + 1)
            seq = clean_sequence(row.get(seq_col, ""))
            if not seq:
                continue
            handle.write(f">{rid}\n{seq}\n")


def parse_fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids


def parse_cdhit_removed_hits(clstr_path: Path, candidate_ids: set[str]) -> pd.DataFrame:
    """Parse best available cluster-level match information for removed candidates.

    CD-HIT .clstr stores cluster members with a representative marked by '*'. For
    cd-hit-2d, manifest/reference sequences are database 1 and candidates are
    database 2. We use the representative or first non-candidate member as the
    reference-like sequence for each removed candidate when possible.
    """
    rows = []
    if not clstr_path.exists():
        return pd.DataFrame(rows)

    cluster = None
    cluster_lines: list[str] = []

    def flush(lines: list[str], cluster_name: str | None) -> None:
        if not lines:
            return
        ref_id = None
        members = []
        for line in lines:
            m = re.search(r">([^\.\s]+)\.\.\.", line)
            if not m:
                continue
            sid = m.group(1)
            members.append((sid, line))
            if sid not in candidate_ids and ("*" in line or ref_id is None):
                ref_id = sid
        if ref_id is None:
            for sid, _ in members:
                if sid not in candidate_ids:
                    ref_id = sid
                    break
        for sid, line in members:
            if sid in candidate_ids:
                ident = None
                m_pct = re.search(r"at\s+([0-9.]+)%", line)
                if m_pct:
                    ident = float(m_pct.group(1)) / 100.0
                rows.append({
                    "candidate_id": sid,
                    "cdhit_cluster": cluster_name,
                    "manifest_match_reference_id": ref_id,
                    "cdhit_reported_identity": ident,
                    "cdhit_cluster_line": line.strip(),
                })

    with clstr_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">Cluster"):
                flush(cluster_lines, cluster)
                cluster = line[1:]
                cluster_lines = []
            else:
                cluster_lines.append(line)
    flush(cluster_lines, cluster)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Peptide manifest FASTA/CSV/TSV used as novelty reference.")
    parser.add_argument("--candidates", default=None, help="Optional candidate CSV/FASTA. If omitted, uses all G01..G10 scored files.")
    parser.add_argument("--results-dir", default="v4b/results")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--end-generation", type=int, default=10)
    parser.add_argument("--output-dir", default="v4b/results/novelty_manifest75_filter")
    parser.add_argument("--identity", type=float, default=0.75)
    parser.add_argument("--word-length", type=int, default=2, help="CD-HIT word size. For 0.75 identity on short peptides, 2 is conservative.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=0, help="CD-HIT -M memory in MB. 0 means unlimited.")
    parser.add_argument("--candidate-sequence-col", default=None)
    parser.add_argument("--candidate-id-col", default=None)
    parser.add_argument("--reference-sequence-col", default=None)
    parser.add_argument("--reference-id-col", default=None)
    parser.add_argument("--cdhit-bin", default="cd-hit-2d")
    parser.add_argument("--dry-run", action="store_true", help="Prepare FASTA and command but do not run cd-hit-2d.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args)
    reference = load_reference(Path(args.reference), args.reference_sequence_col, args.reference_id_col)

    candidate_fasta = output_dir / "v4b_100k_candidates_for_manifest75_filter.fasta"
    reference_fasta = output_dir / "manifest_reference_for_manifest75_filter.fasta"
    kept_fasta = output_dir / f"v4b_candidates_not_{int(args.identity*100)}pct_identical_to_manifest.fasta"

    # CD-HIT identifiers must be stable and parseable.
    candidates["cdhit_candidate_id"] = [safe_id(x, "cand", i + 1) for i, x in enumerate(candidates["candidate_id"])]
    reference["cdhit_reference_id"] = [safe_id(x, "ref", i + 1) for i, x in enumerate(reference["reference_id"])]

    write_fasta(candidates, candidate_fasta, "cdhit_candidate_id", "sequence_clean")
    write_fasta(reference, reference_fasta, "cdhit_reference_id", "reference_sequence_clean")

    cmd = [
        args.cdhit_bin,
        "-i", str(reference_fasta),
        "-i2", str(candidate_fasta),
        "-o", str(kept_fasta),
        "-c", str(args.identity),
        "-n", str(args.word_length),
        "-d", "0",
        "-T", str(args.threads),
        "-M", str(args.memory_mb),
    ]

    command_path = output_dir / "cdhit_manifest75_command.txt"
    command_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    if args.dry_run:
        print("[DRY RUN] FASTA files prepared. Command:")
        print(" ".join(cmd))
        return

    if shutil.which(args.cdhit_bin) is None:
        raise SystemExit(
            f"Could not find {args.cdhit_bin!r} on PATH. Install CD-HIT first, for example:\n"
            f"  conda install -c bioconda cd-hit -y\n\n"
            f"Prepared FASTA files and command at: {output_dir}"
        )

    print("[V4B] Running CD-HIT manifest 75% identity filter...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    kept_ids = parse_fasta_ids(kept_fasta)
    candidate_ids = set(candidates["cdhit_candidate_id"].astype(str))
    removed_ids = candidate_ids - kept_ids

    candidates["manifest75_status"] = candidates["cdhit_candidate_id"].map(lambda x: "kept_below_threshold" if x in kept_ids else "removed_manifest_ge_threshold")
    candidates["manifest_identity_threshold"] = float(args.identity)
    candidates["manifest_reference_file"] = str(args.reference)

    clstr_path = Path(str(kept_fasta) + ".clstr")
    hits = parse_cdhit_removed_hits(clstr_path, candidate_ids)
    if not hits.empty:
        candidates = candidates.merge(hits, on="cdhit_candidate_id", how="left") if "cdhit_candidate_id" in hits.columns else candidates
        # parse_cdhit_removed_hits uses candidate_id column; align it if needed
        if "candidate_id_y" in candidates.columns:
            candidates = candidates.drop(columns=["candidate_id_y"]).rename(columns={"candidate_id_x": "candidate_id"})

    kept = candidates[candidates["manifest75_status"] == "kept_below_threshold"].copy()
    removed = candidates[candidates["manifest75_status"] == "removed_manifest_ge_threshold"].copy()

    all_out = output_dir / "v4b_100k_manifest75_all_with_flags.csv"
    kept_out = output_dir / "v4b_100k_manifest75_kept_below_75_identity.csv"
    removed_out = output_dir / "v4b_100k_manifest75_removed_ge_75_identity.csv"
    kept_final_fasta = output_dir / "v4b_100k_manifest75_kept_below_75_identity.fasta"
    removed_final_fasta = output_dir / "v4b_100k_manifest75_removed_ge_75_identity.fasta"

    candidates.to_csv(all_out, index=False)
    kept.to_csv(kept_out, index=False)
    removed.to_csv(removed_out, index=False)
    write_fasta(kept, kept_final_fasta, "candidate_id", "sequence_clean")
    write_fasta(removed, removed_final_fasta, "candidate_id", "sequence_clean")

    by_generation = None
    if "generation_source" in candidates.columns:
        by_generation = (
            candidates.groupby(["generation_source", "manifest75_status"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        by_generation.to_csv(output_dir / "v4b_manifest75_status_by_generation.csv", index=False)

    summary = {
        "identity_threshold": float(args.identity),
        "reference_file": str(args.reference),
        "reference_unique_sequences": int(len(reference)),
        "candidate_unique_sequences": int(len(candidates)),
        "kept_below_threshold": int(len(kept)),
        "removed_manifest_ge_threshold": int(len(removed)),
        "fraction_kept": float(len(kept) / max(len(candidates), 1)),
        "fraction_removed": float(len(removed) / max(len(candidates), 1)),
        "method": "cd-hit-2d",
        "command": " ".join(cmd),
        "outputs": {
            "all_with_flags": str(all_out),
            "kept_csv": str(kept_out),
            "removed_csv": str(removed_out),
            "kept_fasta": str(kept_final_fasta),
            "removed_fasta": str(removed_final_fasta),
            "cdhit_kept_fasta_raw": str(kept_fasta),
            "cdhit_cluster_file": str(clstr_path),
            "command_file": str(command_path),
        },
    }
    (output_dir / "v4b_manifest75_filter_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nMANIFEST 75% IDENTITY FILTER SUMMARY")
    print(json.dumps(summary, indent=2, default=str))

    if by_generation is not None:
        print("\nSTATUS BY GENERATION")
        print(by_generation.to_string(index=False))

    print("\nTop kept candidates by existing ranking columns:")
    show_cols = [
        "candidate_id", "generation_source", "sequence_clean", "manifest75_status",
        "APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64",
        "criteria_hydrophobic_fraction", "criteria_charge", "criteria_length",
        "v4b_elite_composite_score", "ampjepa_master_score",
    ]
    show_cols = [c for c in show_cols if c in kept.columns]
    sort_cols = [c for c in ["v4b_elite_composite_score", "ampjepa_master_score", "APEX_median_MIC"] if c in kept.columns]
    if sort_cols:
        ascending = [False if c in {"v4b_elite_composite_score", "ampjepa_master_score"} else True for c in sort_cols]
        kept_show = kept.sort_values(sort_cols, ascending=ascending, na_position="last")
    else:
        kept_show = kept
    print(kept_show[show_cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()

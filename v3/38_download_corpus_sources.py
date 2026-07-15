#!/usr/bin/env python3
"""Download public/local AMP corpus sources for AMP-JEPA-Hybrid v3.

This script writes raw third-party corpus files into the local working tree under
v3/data/raw/corpus_sources/. These files should generally remain uncommitted
unless their license explicitly allows redistribution.

Built-in direct sources
-----------------------
1. APD natural AMPs FASTA:
   https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta
2. UniProt reviewed, short antimicrobial entries as FASTA, using the UniProt
   REST API query endpoint.

Portal/manual sources
---------------------
dbAMP, DRAMP, CAMP/CAMPR, DBAASP, StarPep and similar resources often expose
bulk exports through web portals, session-bound links, or database-specific
terms. For those, download the export from the source website and either place
it directly in v3/data/raw/corpus_sources/ or add the direct export URL to a
manifest TSV/CSV with columns:

    name,url,filename,enabled

Optional public ML benchmark repositories
-----------------------------------------
Use --include-public-ml-repos to download selected public GitHub repository ZIP
archives that are commonly used around AMP prediction. These are treated as
secondary sources because they can contain negative/control sequences or mixed
benchmark data. The extractor keeps only likely corpus files and avoids obvious
negative/non-AMP paths, but the merged corpus report should still be inspected.

After downloading, this script can optionally build the merged upscaled corpus by
calling v3/37_build_upscaled_corpus.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIPROT_REVIEWED_SHORT_AMP_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    "compressed=false&format=fasta&"
    "query=%28keyword%3A%22Antimicrobial%22%29%20AND%20"
    "%28reviewed%3Atrue%29%20AND%20%28length%3A%5B8%20TO%2064%5D%29"
)

DEFAULT_SOURCES = [
    {
        "name": "APD2024a_naturalAMPs",
        "url": "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
        "filename": "apd2024a_natural_amps.fasta",
        "enabled": "true",
        "kind": "file",
    },
    {
        "name": "UniProt_reviewed_short_antimicrobial",
        "url": UNIPROT_REVIEWED_SHORT_AMP_URL,
        "filename": "uniprot_reviewed_short_antimicrobial.fasta",
        "enabled": "true",
        "kind": "file",
    },
]

PUBLIC_ML_REPO_SOURCES = [
    {
        "name": "BirolLab_AMPlify_repo",
        "url": "https://codeload.github.com/BirolLab/AMPlify/zip/refs/heads/master",
        "filename": "BirolLab_AMPlify.zip",
        "enabled": "true",
        "kind": "github_zip",
        "notes": "Public AMPlify repository; can be large; inspect extracted corpus files.",
    },
    {
        "name": "gabrielalonde_antimicrobial_peptide_dataset_repo",
        "url": "https://codeload.github.com/gabrielalonde/Antimicrobial-Peptide-Dataset-/zip/refs/heads/main",
        "filename": "gabrielalonde_Antimicrobial-Peptide-Dataset.zip",
        "enabled": "true",
        "kind": "github_zip",
        "notes": "Public AMP dataset repository discovered by GitHub search; inspect source summary.",
    },
    {
        "name": "clennartz_protein_lm_amp_benchmark_repo",
        "url": "https://codeload.github.com/clennartz-umass/protein-lm-amp-benchmark/zip/refs/heads/main",
        "filename": "clennartz_protein_lm_amp_benchmark.zip",
        "enabled": "true",
        "kind": "github_zip",
        "notes": "Public protein-LM AMP benchmark repository; may contain benchmark splits.",
    },
    {
        "name": "AMPCliff_generation_repo",
        "url": "https://codeload.github.com/Kewei2023/AMPCliff-generation/zip/refs/heads/main",
        "filename": "AMPCliff_generation.zip",
        "enabled": "true",
        "kind": "github_zip",
        "notes": "Public AMPCliff-generation repository from AMPCliff paper; may contain MIC benchmark files.",
    },
]

CORPUS_EXTENSIONS = {".fa", ".fasta", ".faa", ".fna", ".csv", ".tsv", ".tab", ".txt"}
OBVIOUS_NEGATIVE_MARKERS = [
    "negative",
    "negatives",
    "nonamp",
    "non_amp",
    "non-amp",
    "non_amp",
    "nonantimicrobial",
    "non_antimicrobial",
    "decoy",
    "random",
    "control",
]
LIKELY_CORPUS_MARKERS = [
    "amp",
    "antimicrobial",
    "positive",
    "peptide",
    "sequence",
    "dataset",
    "data",
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def truthy(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "n", "skip"}


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]

    required = {"name", "url", "filename"}
    if rows:
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    for row in rows:
        row.setdefault("enabled", "true")
        row.setdefault("kind", "file")

    return rows


def download_one(source: Dict[str, str], output_dir: Path, overwrite: bool, timeout: int, retries: int) -> Dict[str, object]:
    name = str(source.get("name", "source")).strip() or "source"
    url = str(source.get("url", "")).strip()
    filename = str(source.get("filename", "")).strip() or Path(url).name

    if not url:
        return {"name": name, "status": "skipped", "reason": "empty_url"}

    output_path = output_dir / filename
    if output_path.exists() and not overwrite:
        return {
            "name": name,
            "url": url,
            "path": str(output_path),
            "status": "exists",
            "bytes": output_path.stat().st_size,
        }

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "AMP-JEPA-v3-corpus-downloader/1.1 (+research; contact via local user)",
                    "Accept": "text/plain, text/csv, text/tab-separated-values, application/zip, */*",
                },
            )
            temporary_path = output_path.with_suffix(output_path.suffix + ".part")
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temporary_path.replace(output_path)
            return {
                "name": name,
                "url": url,
                "path": str(output_path),
                "status": "downloaded",
                "bytes": output_path.stat().st_size,
                "attempt": attempt,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(min(2 * attempt, 10))

    return {
        "name": name,
        "url": url,
        "path": str(output_path),
        "status": "failed",
        "error": last_error,
    }


def safe_extract_member_name(member_name: str) -> str:
    member_name = member_name.replace("\\", "/")
    parts = [part for part in member_name.split("/") if part and part not in {".", ".."}]
    return "/".join(parts)


def is_likely_positive_corpus_path(path_text: str) -> bool:
    lower = path_text.lower()
    suffix = Path(lower).suffix
    if suffix not in CORPUS_EXTENSIONS:
        return False
    if any(marker in lower for marker in OBVIOUS_NEGATIVE_MARKERS):
        return False
    return any(marker in lower for marker in LIKELY_CORPUS_MARKERS)


def extract_corpus_files_from_zip(zip_path: Path, output_dir: Path, source_name: str, overwrite: bool) -> Dict[str, object]:
    extract_dir = output_dir / f"{zip_path.stem}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[str] = []
    skipped_files = 0

    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                safe_name = safe_extract_member_name(member.filename)
                if not is_likely_positive_corpus_path(safe_name):
                    skipped_files += 1
                    continue
                relative = Path(safe_name)
                # Drop the archive top-level directory for cleaner extraction.
                if len(relative.parts) > 1:
                    relative = Path(*relative.parts[1:])
                target_path = extract_dir / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if target_path.exists() and not overwrite:
                    extracted_files.append(str(target_path))
                    continue
                with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
                extracted_files.append(str(target_path))
    except zipfile.BadZipFile as exc:
        return {
            "name": source_name,
            "archive": str(zip_path),
            "status": "extract_failed",
            "error": str(exc),
        }

    return {
        "name": source_name,
        "archive": str(zip_path),
        "status": "extracted",
        "extract_dir": str(extract_dir),
        "extracted_files": extracted_files,
        "n_extracted_files": len(extracted_files),
        "n_skipped_files": skipped_files,
    }


def write_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": "APD2024a_naturalAMPs",
            "url": "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
            "filename": "apd2024a_natural_amps.fasta",
            "enabled": "true",
            "kind": "file",
            "notes": "Direct public APD FASTA URL.",
        },
        {
            "name": "UniProt_reviewed_short_antimicrobial",
            "url": UNIPROT_REVIEWED_SHORT_AMP_URL,
            "filename": "uniprot_reviewed_short_antimicrobial.fasta",
            "enabled": "true",
            "kind": "file",
            "notes": "UniProt REST query for reviewed short antimicrobial entries.",
        },
        {
            "name": "DRAMP_export",
            "url": "",
            "filename": "dramp_export.fasta",
            "enabled": "false",
            "kind": "file",
            "notes": "Paste a stable DRAMP export URL or place the file manually in corpus_sources.",
        },
        {
            "name": "dbAMP_export",
            "url": "",
            "filename": "dbamp_export.csv",
            "enabled": "false",
            "kind": "file",
            "notes": "Paste a stable dbAMP export URL or place the file manually in corpus_sources.",
        },
        {
            "name": "CAMPR_export",
            "url": "",
            "filename": "campr_export.fasta",
            "enabled": "false",
            "kind": "file",
            "notes": "Paste a stable CAMP/CAMPR export URL or place the file manually in corpus_sources.",
        },
        {
            "name": "DBAASP_export",
            "url": "",
            "filename": "dbaasp_export.csv",
            "enabled": "false",
            "kind": "file",
            "notes": "Paste a stable DBAASP export/API URL or place the file manually in corpus_sources.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def discover_corpus_files(output_dir: Path) -> List[Path]:
    inputs = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in CORPUS_EXTENSIONS
        and not path.name.endswith(".part")
        and "__MACOSX" not in str(path)
    )
    return inputs


def build_upscaled_corpus(output_dir: Path, output_prefix: Path, min_len: int, max_len: int) -> None:
    builder = PROJECT_ROOT / "v3" / "37_build_upscaled_corpus.py"
    if not builder.exists():
        raise FileNotFoundError(f"Builder not found: {builder}")

    inputs = discover_corpus_files(output_dir)
    if not inputs:
        raise RuntimeError(f"No corpus files found in {output_dir}")

    command = [
        sys.executable,
        str(builder),
        "--inputs",
        *[str(path) for path in inputs],
        "--output-prefix",
        str(output_prefix),
        "--min-len",
        str(min_len),
        "--max-len",
        str(max_len),
    ]
    print("[INFO] Building upscaled corpus from discovered files:")
    for path in inputs:
        print(f"  {path}")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="v3/data/raw/corpus_sources")
    parser.add_argument("--manifest", default="", help="Optional TSV/CSV manifest with name,url,filename,enabled columns.")
    parser.add_argument("--no-defaults", action="store_true", help="Do not download built-in direct sources such as APD/UniProt.")
    parser.add_argument("--include-public-ml-repos", action="store_true", help="Also download selected public GitHub AMP benchmark repository ZIP archives and extract likely positive corpus files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--write-template", default="", help="Write a manifest template and exit.")
    parser.add_argument("--build-corpus", action="store_true", help="Run v3/37_build_upscaled_corpus.py after downloads.")
    parser.add_argument("--output-prefix", default="v3/data/processed/upscaled_peptide_corpus_v3")
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()

    if args.write_template:
        template_path = resolve_path(args.write_template)
        write_manifest_template(template_path)
        print(f"[DONE] Wrote manifest template: {template_path}")
        return

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources: List[Dict[str, str]] = []
    if not args.no_defaults:
        sources.extend(DEFAULT_SOURCES)
    if args.include_public_ml_repos:
        sources.extend(PUBLIC_ML_REPO_SOURCES)
    if args.manifest:
        sources.extend(read_manifest(resolve_path(args.manifest)))

    enabled_sources = [source for source in sources if truthy(source.get("enabled", "true"))]
    if not enabled_sources:
        raise SystemExit("[ERROR] No enabled download sources. Use defaults, --manifest, --include-public-ml-repos, or --write-template.")

    results = []
    extraction_results = []
    for source in enabled_sources:
        result = download_one(source, output_dir, args.overwrite, args.timeout, args.retries)
        results.append(result)
        status = result.get("status")
        print(f"[{status}] {result.get('name')} -> {result.get('path', result.get('reason', ''))}")

        kind = str(source.get("kind", "file")).strip().lower()
        downloaded_path = Path(str(result.get("path", "")))
        if kind in {"github_zip", "zip", "archive"} and downloaded_path.exists() and status in {"downloaded", "exists"}:
            extraction = extract_corpus_files_from_zip(downloaded_path, output_dir, str(source.get("name", downloaded_path.stem)), args.overwrite)
            extraction_results.append(extraction)
            print(f"[{extraction.get('status')}] {extraction.get('name')} -> {extraction.get('n_extracted_files', 0)} corpus-like files")

    report_path = output_dir / "corpus_download_report.json"
    report = {
        "downloads": results,
        "extractions": extraction_results,
        "discovered_corpus_files": [str(path) for path in discover_corpus_files(output_dir)],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote download report: {report_path}")

    if args.build_corpus:
        build_upscaled_corpus(output_dir, resolve_path(args.output_prefix), args.min_len, args.max_len)


if __name__ == "__main__":
    main()

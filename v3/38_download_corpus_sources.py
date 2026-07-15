#!/usr/bin/env python3
"""Download public/local AMP corpus sources for AMP-JEPA-Hybrid v3.

This script intentionally writes raw third-party corpus files into the local
working tree under v3/data/raw/corpus_sources/. Those files should generally
remain uncommitted unless their license explicitly allows redistribution.

Default direct-download source
------------------------------
- APD natural AMPs FASTA:
  https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta

Additional sources
------------------
For dbAMP, DRAMP, CAMPR, DBAASP, StarPep, or lab-curated exports, download their
FASTA/CSV/TSV exports from the respective sites and either:

1. place the files directly in v3/data/raw/corpus_sources/, or
2. create a manifest TSV/CSV with columns:
      name,url,filename,enabled
   and run this script with --manifest path/to/manifest.tsv

After downloading, this script can optionally build the merged upscaled corpus by
calling v3/37_build_upscaled_corpus.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [
    {
        "name": "APD2024a_naturalAMPs",
        "url": "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
        "filename": "apd2024a_natural_amps.fasta",
        "enabled": "true",
    },
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
                    "User-Agent": "AMP-JEPA-v3-corpus-downloader/1.0 (+research; contact via local user)",
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


def write_manifest_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": "APD2024a_naturalAMPs",
            "url": "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
            "filename": "apd2024a_natural_amps.fasta",
            "enabled": "true",
            "notes": "Direct public APD FASTA URL.",
        },
        {
            "name": "DRAMP_export",
            "url": "",
            "filename": "dramp_export.fasta",
            "enabled": "false",
            "notes": "Paste a stable DRAMP export URL or place the file manually in corpus_sources.",
        },
        {
            "name": "dbAMP_export",
            "url": "",
            "filename": "dbamp_export.csv",
            "enabled": "false",
            "notes": "Paste a stable dbAMP export URL or place the file manually in corpus_sources.",
        },
        {
            "name": "CAMPR_export",
            "url": "",
            "filename": "campr_export.fasta",
            "enabled": "false",
            "notes": "Paste a stable CAMP/CAMPR export URL or place the file manually in corpus_sources.",
        },
        {
            "name": "DBAASP_export",
            "url": "",
            "filename": "dbaasp_export.csv",
            "enabled": "false",
            "notes": "Paste a stable DBAASP export/API URL or place the file manually in corpus_sources.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_upscaled_corpus(output_dir: Path, output_prefix: Path, min_len: int, max_len: int) -> None:
    builder = PROJECT_ROOT / "v3" / "37_build_upscaled_corpus.py"
    if not builder.exists():
        raise FileNotFoundError(f"Builder not found: {builder}")

    inputs = sorted(
        path
        for path in output_dir.iterdir()
        if path.suffix.lower() in {".fa", ".fasta", ".faa", ".csv", ".tsv", ".tab"}
    )
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
    print("[INFO] Building upscaled corpus:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="v3/data/raw/corpus_sources")
    parser.add_argument("--manifest", default="", help="Optional TSV/CSV manifest with name,url,filename,enabled columns.")
    parser.add_argument("--no-defaults", action="store_true", help="Do not download built-in direct sources such as APD.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
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

    if args.manifest:
        sources.extend(read_manifest(resolve_path(args.manifest)))

    enabled_sources = [source for source in sources if truthy(source.get("enabled", "true"))]
    if not enabled_sources:
        raise SystemExit("[ERROR] No enabled download sources. Use defaults, --manifest, or --write-template.")

    results = []
    for source in enabled_sources:
        result = download_one(source, output_dir, args.overwrite, args.timeout, args.retries)
        results.append(result)
        status = result.get("status")
        print(f"[{status}] {result.get('name')} -> {result.get('path', result.get('reason', ''))}")

    report_path = output_dir / "corpus_download_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote download report: {report_path}")

    if args.build_corpus:
        build_upscaled_corpus(output_dir, resolve_path(args.output_prefix), args.min_len, args.max_len)


if __name__ == "__main__":
    main()

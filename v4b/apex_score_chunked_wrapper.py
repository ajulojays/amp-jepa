#!/usr/bin/env python3
"""Adapter between the V4B evolutionary loop and the existing chunked APEX scorer.

The existing V4 scorer writes fixed filenames inside an output directory:

    apex_scored_v3_candidates.csv
    apex_scoring_summary.json
    apex_batch_logs.json
    apex_top_v3_candidates.fasta

The V4B closed-loop runner expects an external command that creates exactly the
CSV path supplied as {output}. This wrapper calls the proven chunked scorer and
then copies/renames its scored CSV to the requested output path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input candidate CSV from V4B generation loop.")
    parser.add_argument("--output", required=True, help="Exact scored CSV path expected by V4B loop.")
    parser.add_argument("--apex-root", default=None, help="APEX repository/root directory. Defaults to scorer/env default.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-fasta-count", type=int, default=50)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scorer = repo_root / "v4" / "01_score_candidates_chunked_with_apex.py"
    if not scorer.exists():
        raise FileNotFoundError(f"Chunked APEX scorer not found: {scorer}")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    workdir = output_path.parent / f"{output_path.stem}_apex_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(scorer),
        "--candidates",
        str(input_path),
        "--output-dir",
        str(workdir),
        "--batch-size",
        str(args.batch_size),
        "--min-batch-size",
        str(args.min_batch_size),
        "--device",
        args.device,
        "--top-fasta-count",
        str(args.top_fasta_count),
    ]
    if args.apex_root:
        cmd.extend(["--apex-root", args.apex_root])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    print("[V4B-APEX] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    scored_csv = workdir / "apex_scored_v3_candidates.csv"
    if not scored_csv.exists():
        raise FileNotFoundError(f"Expected APEX scored CSV was not produced: {scored_csv}")

    shutil.copyfile(scored_csv, output_path)

    # Keep APEX side outputs beside the requested output for traceability.
    sidecar_map = {
        "apex_scoring_summary.json": output_path.with_name(output_path.stem + "_apex_summary.json"),
        "apex_batch_logs.json": output_path.with_name(output_path.stem + "_apex_batch_logs.json"),
        "apex_top_v3_candidates.fasta": output_path.with_name(output_path.stem + "_top_apex_candidates.fasta"),
    }
    copied_sidecars = {}
    for source_name, dest in sidecar_map.items():
        source = workdir / source_name
        if source.exists():
            shutil.copyfile(source, dest)
            copied_sidecars[source_name] = str(dest)

    adapter_summary = {
        "input": str(input_path),
        "requested_output": str(output_path),
        "scorer": str(scorer),
        "workdir": str(workdir),
        "scored_csv_source": str(scored_csv),
        "copied_sidecars": copied_sidecars,
    }
    summary_path = output_path.with_name(output_path.stem + "_apex_adapter_summary.json")
    summary_path.write_text(json.dumps(adapter_summary, indent=2), encoding="utf-8")

    print(f"[V4B-APEX] Wrote scored output: {output_path}")


if __name__ == "__main__":
    main()

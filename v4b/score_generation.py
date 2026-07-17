#!/usr/bin/env python3
"""Score one V4B generation with an external APEX command or fallback passthrough.

APEX integration is intentionally command-template based because local APEX
installations often differ. Use:

  --apex-command 'python path/to/apex_score.py --input {input} --output {output}'

The placeholders {input} and {output} are required when a command is supplied.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from evolution_core import utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Pre-APEX candidate CSV.")
    parser.add_argument("--output", required=True, help="Scored candidate CSV to create.")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--apex-command", default=None, help="Command template with {input} and {output} placeholders.")
    parser.add_argument("--require-apex", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(output_path.stem + "_scoring_summary.json")

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.overwrite:
        print(f"[V4B] Scored output already exists: {output_path}")
        return

    command_template = args.apex_command or os.environ.get("APEX_SCORE_CMD", "").strip()
    mode = "fallback_passthrough_no_apex"
    command = ""

    if command_template:
        if "{input}" not in command_template or "{output}" not in command_template:
            raise ValueError("APEX command must include both {input} and {output} placeholders.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = command_template.format(input=shlex.quote(str(input_path)), output=shlex.quote(str(output_path)))
        print(f"[V4B] Running APEX command: {command}")
        subprocess.run(command, shell=True, check=True)
        mode = "external_apex_command"
    else:
        if args.require_apex:
            raise RuntimeError("REQUIRE_APEX was set, but no APEX_SCORE_CMD/--apex-command was provided.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, output_path)
        print("[V4B] WARNING: no APEX command provided. Copied pre-APEX candidates as scored fallback.")

    if not output_path.exists():
        raise FileNotFoundError(f"Expected scored output was not created: {output_path}")

    scored = pd.read_csv(output_path, low_memory=False)
    required = {"candidate_id", "sequence"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Scored output missing required columns: {missing}")

    summary = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "generation": int(args.generation),
        "mode": mode,
        "command_template": command_template,
        "executed_command": command,
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(scored)),
        "columns": list(scored.columns),
    }
    write_json(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()

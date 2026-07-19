#!/usr/bin/env python3
"""Run V4C one-digit predicted-MIC panels for every canonical APEX species.

The canonical inventory is produced by ``25_global_one_digit_panel.py``. Each species
is passed to ``26_species_one_digit_panel.py`` using an exact canonical-label regex.
Completed species summaries are skipped by default, making the batch resumable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
        "--inventory",
        default=(
            "v4c/results/final_funnel/05_global_one_digit/"
            "v4c_inferred_species_inventory.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="v4c/results/final_funnel/06_species_panels",
    )
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--heatmap-top-n", type=int, default=100)
    parser.add_argument("--label-top-n", type=int, default=100)
    parser.add_argument("--rest-seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "species"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    inventory_path = Path(args.inventory)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for path in (input_path, inventory_path):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    inventory = pd.read_csv(inventory_path)
    if "inferred_species" not in inventory.columns:
        raise ValueError("Inventory must contain inferred_species.")
    if inventory["inferred_species"].eq("Unresolved").any():
        unresolved = inventory.loc[
            inventory["inferred_species"].eq("Unresolved")
        ]
        raise RuntimeError(
            "Species inventory still contains Unresolved entries; rerun the corrected "
            "global panel before batch species analysis.\n"
            + unresolved.to_string(index=False)
        )

    species_labels = sorted(inventory["inferred_species"].dropna().astype(str).unique())
    if not species_labels:
        raise ValueError("Species inventory is empty.")

    repo_root = Path(__file__).resolve().parents[1]
    panel_script = repo_root / "v4c" / "26_species_one_digit_panel.py"
    if not panel_script.exists():
        raise FileNotFoundError(panel_script)

    rows: list[dict[str, object]] = []
    for index, species in enumerate(species_labels, start=1):
        slug = safe_slug(species)
        species_dir = output_root / slug
        summary_path = species_dir / f"{slug}_analysis_summary.json"

        if summary_path.exists() and not args.overwrite:
            print(
                f"[V4C-SPECIES] {index}/{len(species_labels)} {species}: "
                "completed summary exists; skipping"
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "species": species,
                    "status": "skipped_existing",
                    "n_models": payload.get("n_matched_models"),
                    "n_one_digit_any_model": payload.get(
                        "n_candidates_one_digit_any_model"
                    ),
                    "best_predicted_species_MIC": payload.get(
                        "best_predicted_species_MIC"
                    ),
                    "summary": str(summary_path),
                }
            )
            continue

        species_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(panel_script),
            "--input", str(input_path),
            "--species-regex", rf"^{re.escape(species)}$",
            "--species-label", species,
            "--output-dir", str(species_dir),
            "--cutoff", str(args.cutoff),
            "--heatmap-top-n", str(args.heatmap_top_n),
            "--label-top-n", str(args.label_top_n),
        ]

        print(
            f"[V4C-SPECIES] {index}/{len(species_labels)} {species}: starting"
        )
        subprocess.run(command, check=True)
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Species panel completed without expected summary: {summary_path}"
            )

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "species": species,
                "status": "completed",
                "n_models": payload.get("n_matched_models"),
                "n_one_digit_any_model": payload.get(
                    "n_candidates_one_digit_any_model"
                ),
                "best_predicted_species_MIC": payload.get(
                    "best_predicted_species_MIC"
                ),
                "summary": str(summary_path),
            }
        )
        print(
            f"[V4C-SPECIES] {species}: complete; "
            f"one-digit candidates={payload.get('n_candidates_one_digit_any_model')}"
        )

        if index < len(species_labels) and args.rest_seconds > 0:
            print(
                f"[V4C-SPECIES] Resting {args.rest_seconds:g}s before next species"
            )
            time.sleep(args.rest_seconds)

    batch = pd.DataFrame(rows).sort_values("species").reset_index(drop=True)
    batch_csv = output_root / "v4c_species_panel_batch_summary.csv"
    batch.to_csv(batch_csv, index=False)

    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "all_species_one_digit_predicted_MIC_panels",
        "input_file": str(input_path),
        "inventory_file": str(inventory_path),
        "n_species": int(len(species_labels)),
        "species": species_labels,
        "cutoff_uM_exclusive": float(args.cutoff),
        "output_root": str(output_root),
        "batch_summary": str(batch_csv),
    }
    manifest_path = output_root / "v4c_species_panel_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nV4C ALL-SPECIES PANEL BATCH SUMMARY")
    print(batch.to_string(index=False))
    print(f"\nSaved: {batch_csv}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()

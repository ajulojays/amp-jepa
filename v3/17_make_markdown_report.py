#!/usr/bin/env python3
"""Create a small Markdown report from v3 outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def table_preview(path: Path, n: int = 10) -> str:
    if not path.exists():
        return f"Missing: `{path}`\n"
    df = pd.read_csv(path)
    return f"`{path}`: {len(df):,} rows, {len(df.columns):,} columns\n\n" + df.head(n).to_markdown(index=False) + "\n"


def main() -> None:
    out = Path("v3/results/v3_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# AMP-JEPA-Hybrid v3 Report\n",
        "## Ranked candidates\n" + table_preview(Path("v3/results/ranked_candidates_v3.csv")),
        "## Top panel\n" + table_preview(Path("v3/results/top_panel_v3.csv")),
        "## APEX comparator\n" + table_preview(Path("v3/results/apex_comparator_v3.csv")),
    ]
    out.write_text("\n\n".join(sections))
    print(f"[DONE] Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a manifest for v3 candidate files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    paths = sorted(Path("v3/results").glob("*.csv"))
    rows = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            rows.append({"file": str(path), "rows": len(df), "columns": len(df.columns)})
        except Exception as exc:
            rows.append({"file": str(path), "rows": None, "columns": None, "error": str(exc)})
    out = Path("v3/results/candidate_manifest_v3.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[DONE] Wrote {out}")


if __name__ == "__main__":
    main()

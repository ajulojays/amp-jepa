#!/usr/bin/env python3
"""Print a simple index of v3 scripts."""

from pathlib import Path


def main() -> None:
    scripts = sorted(Path("v3").glob("*.py"))
    for path in scripts:
        first = ""
        for line in path.read_text(errors="ignore").splitlines():
            if line.strip().startswith('"""'):
                first = line.strip().strip('"')
                break
        print(f"- `{path}` — {first}")


if __name__ == "__main__":
    main()

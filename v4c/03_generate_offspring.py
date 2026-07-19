#!/usr/bin/env python3
"""Run the frozen V4B offspring generator while assigning V4C candidate IDs."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path


def load_module_from_path(module_name: str, module_path: Path):
    """Load a module safely, including support for dataclass type resolution."""
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")

    module = importlib.util.module_from_spec(spec)

    # dataclasses and postponed annotations resolve the defining module through
    # sys.modules while the class body is being executed. Register first.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def load_v4b_generator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "v4b" / "03_generate_offspring.py"
    return load_module_from_path("v4b_generate_offspring", module_path)


def load_v3_module_safe(repo_root: Path):
    """Replacement for the older V4B loader that omitted sys.modules registration."""
    module_path = repo_root / "v3" / "ampjepa_hybrid_v3.py"
    return load_module_from_path("ampjepa_hybrid_v3", module_path)


def v4c_stable_id(sequence: str, generation: int) -> str:
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    return f"V4C_G{generation:02d}_{digest}"


def main() -> None:
    module = load_v4b_generator()
    module.stable_id = v4c_stable_id
    module.load_v3_module = load_v3_module_safe
    module.main()


if __name__ == "__main__":
    main()

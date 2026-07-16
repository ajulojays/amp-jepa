#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-v4/results}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-v4/archives}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NAME="v4a_complete_${STAMP}"
WORKDIR="${ARCHIVE_ROOT}/${NAME}"

mkdir -p "${WORKDIR}"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "${WORKDIR}/${dest}")"
    cp -a "$src" "${WORKDIR}/${dest}"
  else
    echo "[WARN] Missing: $src" >&2
  fi
}

# Core candidate and scoring outputs.
copy_if_exists "${RESULTS_ROOT}/seed_pool" "results/seed_pool"
copy_if_exists "${RESULTS_ROOT}/landscape" "results/landscape"
copy_if_exists "${RESULTS_ROOT}/optimization" "results/optimization"
copy_if_exists "${RESULTS_ROOT}/rescue" "results/rescue"
copy_if_exists "${RESULTS_ROOT}/robustness" "results/robustness"
copy_if_exists "${RESULTS_ROOT}/final_panel" "results/final_panel"
copy_if_exists "${RESULTS_ROOT}/logs" "results/logs"

# Model/config/code snapshot needed to reproduce the frozen milestone.
copy_if_exists "v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt" "model/amp_jepa_hybrid_v3_qc_core.pt"
copy_if_exists "v3/data/processed/peptide_corpus_v3_qc_core.csv" "model/peptide_corpus_v3_qc_core.csv"
copy_if_exists "v4/configs" "code/configs"
copy_if_exists "v4/README.md" "code/README.md"
copy_if_exists "v4/V4A_ARCHITECTURE.md" "code/V4A_ARCHITECTURE.md"
copy_if_exists "v4/V4A_FINDINGS_2026-07-16.md" "code/V4A_FINDINGS_2026-07-16.md"
copy_if_exists "v4/G_RESCUE.md" "code/G_RESCUE.md"
copy_if_exists "v4/run_v4a_fullscale.sh" "code/run_v4a_fullscale.sh"

# Include all executable V4A scripts.
mkdir -p "${WORKDIR}/code/scripts"
find v4 -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) -print0 | while IFS= read -r -d '' f; do
  cp -a "$f" "${WORKDIR}/code/scripts/"
done

# Environment and repository provenance.
{
  echo "archive_name=${NAME}"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo NA)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || echo NA)"
  echo "python=$(python --version 2>&1 || true)"
  echo "pytorch=$(python - <<'PY' 2>/dev/null || true
import torch
print(torch.__version__)
PY
)"
  echo "cuda_available=$(python - <<'PY' 2>/dev/null || true
import torch
print(torch.cuda.is_available())
PY
)"
} > "${WORKDIR}/PROVENANCE.txt"

python - <<PY
from pathlib import Path
import hashlib, json
root = Path("${WORKDIR}")
rows = []
for p in sorted(root.rglob('*')):
    if p.is_file():
        h = hashlib.sha256()
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                h.update(chunk)
        rows.append({"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size, "sha256": h.hexdigest()})
(root / "MANIFEST.sha256.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Manifested {len(rows)} files")
PY

tar -czf "${ARCHIVE_ROOT}/${NAME}.tar.gz" -C "${ARCHIVE_ROOT}" "${NAME}"
sha256sum "${ARCHIVE_ROOT}/${NAME}.tar.gz" > "${ARCHIVE_ROOT}/${NAME}.tar.gz.sha256"

echo "V4A archive created:"
echo "  ${ARCHIVE_ROOT}/${NAME}.tar.gz"
echo "  ${ARCHIVE_ROOT}/${NAME}.tar.gz.sha256"

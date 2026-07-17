#!/usr/bin/env bash
set -euo pipefail

V4A_CANDIDATES="${V4A_CANDIDATES:-v4/results/final_panel/v4a_candidate_groups_all.csv}"
CHECKPOINT="${CHECKPOINT:-v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt}"
OUTDIR="${OUTDIR:-v4b/results/generation_00}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

echo "[V4B] Generation 0: importing frozen V4A population"
echo "[V4B] Source:     ${V4A_CANDIDATES}"
echo "[V4B] Checkpoint: ${CHECKPOINT}"
echo "[V4B] Output:     ${OUTDIR}"

python v4b/00_import_frozen_v4a.py \
  --input "${V4A_CANDIDATES}" \
  --outdir "${OUTDIR}" \
  "${EXTRA_ARGS[@]}"

echo "[V4B] Generation 0: encoding frozen population"
python v4b/01_encode_generation_00.py \
  --input "${OUTDIR}/generation_00_candidates.csv" \
  --checkpoint "${CHECKPOINT}" \
  --outdir "${OUTDIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"

echo "[V4B] Generation 0 complete"
echo "[V4B] Candidate table: ${OUTDIR}/generation_00_candidates.csv"
echo "[V4B] Latent vectors:  ${OUTDIR}/latent_vectors.npz"
echo "[V4B] Manifest:        ${OUTDIR}/generation_manifest.json"
echo "[V4B] Encoding report: ${OUTDIR}/latent_encoding_summary.json"

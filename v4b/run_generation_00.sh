#!/usr/bin/env bash
set -euo pipefail

V4A_CANDIDATES="${V4A_CANDIDATES:-v4/results/final_panel/v4a_candidate_groups_all.csv}"
CHECKPOINT="${CHECKPOINT:-v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt}"
OUTDIR="${OUTDIR:-v4b/results/generation_00}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
OVERWRITE="${OVERWRITE:-0}"

CANDIDATE_TABLE="${OUTDIR}/generation_00_candidates.csv"
MANIFEST="${OUTDIR}/generation_manifest.json"
LATENT_VECTORS="${OUTDIR}/latent_vectors.npz"
LATENT_METADATA="${OUTDIR}/latent_metadata.csv"
LATENT_SUMMARY="${OUTDIR}/latent_encoding_summary.json"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

echo "[V4B] Generation 0"
echo "[V4B] Source:     ${V4A_CANDIDATES}"
echo "[V4B] Checkpoint: ${CHECKPOINT}"
echo "[V4B] Output:     ${OUTDIR}"

if [[ -f "${CANDIDATE_TABLE}" && -f "${MANIFEST}" && "${OVERWRITE}" != "1" ]]; then
  echo "[V4B] Import already complete; resuming from existing Generation 0 candidate table."
else
  echo "[V4B] Importing frozen V4A population"
  python v4b/00_import_frozen_v4a.py \
    --input "${V4A_CANDIDATES}" \
    --outdir "${OUTDIR}" \
    "${EXTRA_ARGS[@]}"
fi

if [[ ! -f "${CANDIDATE_TABLE}" ]]; then
  echo "[V4B][ERROR] Candidate table was not created: ${CANDIDATE_TABLE}" >&2
  exit 1
fi

if [[ -f "${LATENT_VECTORS}" && -f "${LATENT_METADATA}" && -f "${LATENT_SUMMARY}" && "${OVERWRITE}" != "1" ]]; then
  echo "[V4B] Latent encoding already complete; nothing to do."
else
  echo "[V4B] Encoding frozen population"
  python v4b/01_encode_generation_00.py \
    --input "${CANDIDATE_TABLE}" \
    --checkpoint "${CHECKPOINT}" \
    --outdir "${OUTDIR}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
fi

for required in "${CANDIDATE_TABLE}" "${MANIFEST}" "${LATENT_VECTORS}" "${LATENT_METADATA}" "${LATENT_SUMMARY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[V4B][ERROR] Required Generation 0 output missing: ${required}" >&2
    exit 1
  fi
done

echo "[V4B] Generation 0 complete"
echo "[V4B] Candidate table: ${CANDIDATE_TABLE}"
echo "[V4B] Latent vectors:  ${LATENT_VECTORS}"
echo "[V4B] Manifest:        ${MANIFEST}"
echo "[V4B] Encoding report: ${LATENT_SUMMARY}"

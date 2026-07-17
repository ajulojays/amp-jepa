#!/usr/bin/env bash
set -euo pipefail

SAFETY_INPUT="${SAFETY_INPUT:-v4b/data/safety_reference.csv}"
CHECKPOINT="${CHECKPOINT:-v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt}"
OUTDIR="${OUTDIR:-v4b/results/safety_manifold_pilot}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTDIR}"

echo "=== V4B PHASE 0: SAFETY MANIFOLD PILOT ==="
echo "SAFETY_INPUT: ${SAFETY_INPUT}"
echo "CHECKPOINT:   ${CHECKPOINT}"
echo "OUTDIR:      ${OUTDIR}"
echo "DEVICE:      ${DEVICE}"
echo

python v4b/00_prepare_safety_reference.py \
  --input "${SAFETY_INPUT}" \
  --output "${OUTDIR}/safety_reference_clean.csv" \
  --summary "${OUTDIR}/safety_reference_summary.json"

python v4b/01_encode_safety_manifold.py \
  --input "${OUTDIR}/safety_reference_clean.csv" \
  --checkpoint "${CHECKPOINT}" \
  --outdir "${OUTDIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}"

python v4b/02_evaluate_safety_manifold.py \
  --metadata "${OUTDIR}/latent_metadata.csv" \
  --latent "${OUTDIR}/latent_mu.npy" \
  --outdir "${OUTDIR}"

echo
echo "=== SAFETY MANIFOLD PILOT COMPLETE ==="
echo "Summary: ${OUTDIR}/safety_manifold_summary.json"
echo "Metrics: ${OUTDIR}/safety_manifold_metrics.csv"

#!/usr/bin/env bash
# ==============================================================================
# AMP-JEPA-HYBRID V4A: FULL-SCALE APEX-ONLY OPTIMIZATION PIPELINE
# ==============================================================================
# V4A = V3 backbone + whole-population optimization + failure-aware G-Rescue.
#
# This runner uses the existing v3 APEX scorer for all APEX inference and adds
# V4A landscape mapping, sequence-level optimization, G-Rescue gain analysis,
# and Pareto final panel selection.
# ==============================================================================

set -euo pipefail

APEX_ROOT="${APEX_ROOT:-/home/julojays/apex}"
RESULTS_ROOT="${RESULTS_ROOT:-v4/results}"
V3_RESULTS="${V3_RESULTS:-v3/results}"
MAX_SEEDS="${MAX_SEEDS:-20000}"
MAX_PARENTS="${MAX_PARENTS:-1200}"
MAX_VARIANTS="${MAX_VARIANTS:-8000}"
TOP_FASTA_COUNT="${TOP_FASTA_COUNT:-50}"

mkdir -p \
  "${RESULTS_ROOT}/seed_pool" \
  "${RESULTS_ROOT}/landscape" \
  "${RESULTS_ROOT}/rescue" \
  "${RESULTS_ROOT}/optimization" \
  "${RESULTS_ROOT}/robustness" \
  "${RESULTS_ROOT}/final_panel" \
  "${RESULTS_ROOT}/logs"

echo "=== AMP-JEPA-HYBRID V4A FULL-SCALE PIPELINE ==="
echo "APEX_ROOT:    ${APEX_ROOT}"
echo "V3_RESULTS:   ${V3_RESULTS}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "MAX_SEEDS:    ${MAX_SEEDS}"
echo "MAX_PARENTS:  ${MAX_PARENTS}"
echo "MAX_VARIANTS: ${MAX_VARIANTS}"
echo

# ------------------------------------------------------------------------------
echo "[V4A-00] Prepare seed pool from existing v3 outputs"
# ------------------------------------------------------------------------------
python v4/00_prepare_v4a_seed_pool.py \
  --v3-results "${V3_RESULTS}" \
  --output "${RESULTS_ROOT}/seed_pool/v4a_seed_candidates.csv" \
  --summary "${RESULTS_ROOT}/seed_pool/v4a_seed_summary.json" \
  --max-seeds "${MAX_SEEDS}" \
  2>&1 | tee "${RESULTS_ROOT}/logs/00_prepare_seed_pool.log"

# ------------------------------------------------------------------------------
echo "[V4A-01] APEX-score seed pool using proven v3 scorer"
# ------------------------------------------------------------------------------
python v3/25_score_v3_candidates_with_apex.py \
  --candidates "${RESULTS_ROOT}/seed_pool/v4a_seed_candidates.csv" \
  --output-dir "${RESULTS_ROOT}/seed_pool/apex_seed_scoring" \
  --apex-root "${APEX_ROOT}" \
  --top-fasta-count "${TOP_FASTA_COUNT}" \
  2>&1 | tee "${RESULTS_ROOT}/logs/01_score_seed_pool.log"

# ------------------------------------------------------------------------------
echo "[V4A-02] Map candidate landscape and assign A/B/C/D/E/F/G/G5 classes"
# ------------------------------------------------------------------------------
python v4/02_map_candidate_landscape.py \
  --input "${RESULTS_ROOT}/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv" \
  --outdir "${RESULTS_ROOT}" \
  2>&1 | tee "${RESULTS_ROOT}/logs/02_map_landscape.log"

# ------------------------------------------------------------------------------
echo "[V4A-03] Optimize full population and perform G-Rescue"
# ------------------------------------------------------------------------------
python v4/03_optimize_candidate_population.py \
  --landscape "${RESULTS_ROOT}/landscape/candidate_landscape.csv" \
  --output "${RESULTS_ROOT}/optimization/optimized_variants.csv" \
  --g-output "${RESULTS_ROOT}/rescue/g_rescue_variants.csv" \
  --max-parents "${MAX_PARENTS}" \
  --max-variants "${MAX_VARIANTS}" \
  2>&1 | tee "${RESULTS_ROOT}/logs/03_optimize_population.log"

# ------------------------------------------------------------------------------
echo "[V4A-04] APEX-score optimized/rescued variants using v3 scorer"
# ------------------------------------------------------------------------------
python v3/25_score_v3_candidates_with_apex.py \
  --candidates "${RESULTS_ROOT}/optimization/optimized_variants.csv" \
  --output-dir "${RESULTS_ROOT}/optimization/apex_optimized_scoring" \
  --apex-root "${APEX_ROOT}" \
  --top-fasta-count "${TOP_FASTA_COUNT}" \
  2>&1 | tee "${RESULTS_ROOT}/logs/04_score_optimized_variants.log"

# ------------------------------------------------------------------------------
echo "[V4A-05] Compute optimization gains and G-Rescue successes"
# ------------------------------------------------------------------------------
python v4/04_compute_optimization_gains.py \
  --parent-landscape "${RESULTS_ROOT}/landscape/candidate_landscape.csv" \
  --scored-variants "${RESULTS_ROOT}/optimization/apex_optimized_scoring/apex_scored_v3_candidates.csv" \
  --output "${RESULTS_ROOT}/optimization/optimized_variants_with_gains.csv" \
  --g-success-output "${RESULTS_ROOT}/rescue/g_rescue_successes.csv" \
  2>&1 | tee "${RESULTS_ROOT}/logs/05_compute_optimization_gains.log"

# ------------------------------------------------------------------------------
echo "[V4A-06] Select final V4A Pareto panel"
# ------------------------------------------------------------------------------
python v4/05_select_v4a_pareto_panel.py \
  --seed-scored "${RESULTS_ROOT}/seed_pool/apex_seed_scoring/apex_scored_v3_candidates.csv" \
  --variant-gains "${RESULTS_ROOT}/optimization/optimized_variants_with_gains.csv" \
  --outdir "${RESULTS_ROOT}/final_panel" \
  2>&1 | tee "${RESULTS_ROOT}/logs/06_select_final_panel.log"

echo
echo "=== V4A FULL-SCALE PIPELINE COMPLETE ==="
echo "Final panel: ${RESULTS_ROOT}/final_panel/v4a_top50_panel.csv"
echo "Final FASTA: ${RESULTS_ROOT}/final_panel/v4a_top50_panel.fasta"
echo "G-Rescue successes: ${RESULTS_ROOT}/rescue/g_rescue_successes.csv"

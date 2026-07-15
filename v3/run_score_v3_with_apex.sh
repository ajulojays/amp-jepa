#!/usr/bin/env bash
# ==============================================================================
# Score v3-generated AMP-JEPA candidates with the real APEX MIC ensemble.
#
# The pasted APEX/ApexOracle table is used only as a benchmark comparator.
# It is not used to assign MIC values to v3-generated candidates.
# ==============================================================================
set -euo pipefail

CANDIDATES="${1:-v3/results/top_panel_v3.csv}"
BENCHMARK_APEX="${2:-v3/data/external/apex_oracle_ranked_summary.csv}"
OUTPUT_DIR="${3:-v3/results/apex_scored_v3}"

python v3/05_score_v3_candidates_with_apex.py \
  --candidates "$CANDIDATES" \
  --benchmark-apex "$BENCHMARK_APEX" \
  --output-dir "$OUTPUT_DIR" \
  --max-candidates 50

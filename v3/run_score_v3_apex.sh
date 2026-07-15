#!/usr/bin/env bash
set -euo pipefail

python v3/25_score_v3_candidates_with_apex.py \
  --candidates v3/results/top_panel_v3.csv \
  --output-dir v3/results/apex_scored_v3 \
  --oracle v3/data/external/apex_oracle_ranked_summary.csv

python v3/26_select_apex_aware_panel.py \
  --scored-candidates v3/results/apex_scored_v3/apex_scored_v3_candidates.csv \
  --output-dir v3/results/apex_scored_v3 \
  --top-n 20

#!/usr/bin/env bash
set -euo pipefail

python v3/26_select_apex_aware_panel.py \
  --scored-candidates v3/results/apex_scored_v3/apex_scored_v3_candidates.csv \
  --output-dir v3/results/apex_scored_v3 \
  --top-n 20

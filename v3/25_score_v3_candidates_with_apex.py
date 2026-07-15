#!/usr/bin/env python3
"""Score AMP-JEPA-Hybrid v3 candidates with the APEX MIC ensemble.

This is the v3 equivalent of the older AMP-JEPA discovery scoring step.
It does real APEX ensemble inference for v3-generated peptides, then adds
cross-organism MIC summaries so v3 candidates can be compared against the
bundled APEX/ApexOracle rows.

Typical use:
    python v3/25_score_v3_candidates_with_apex.py \
      --candidates v3/results/top_panel
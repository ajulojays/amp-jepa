# v3 run order

For a quick wiring check:

```bash
pip install -r v3/requirements-v3.txt
bash v3/run_tiny_demo.sh
```

For a tiny debug train using the bundled APEX sequences only:

```bash
bash v3/run_tiny_debug_training.sh
```

For the real v3 run:

```bash
# first place your curated APD/dbAMP/DRAMP/CAMPR FASTA/CSV here or edit INPUTS in run_v3_hybrid.sh
# v3/data/raw/peptides.fasta
export APEX_ROOT=/home/julojays/apex
bash v3/run_v3_hybrid.sh
```

If your local APEX checkout exists at `$APEX_ROOT` or `/home/julojays/apex`, `run_v3_hybrid.sh` now automatically:

1. trains v3,
2. generates candidates,
3. ranks them by v3 heuristic filters,
4. scores the top panel with the APEX MIC ensemble,
5. selects an APEX-aware final panel using MIC, breadth, worst-case MIC, novelty, and sequence developability filters.

To run only the v3 APEX/MIC scoring and final selection after candidates already exist:

```bash
export APEX_ROOT=/home/julojays/apex
bash v3/run_score_v3_apex.sh
```

To rerun only the final APEX-aware panel selector after MIC scoring already exists:

```bash
bash v3/run_select_apex_aware_panel.sh
```

Main MIC outputs:

```text
v3/results/apex_scored_v3/apex_scored_v3_candidates.csv
v3/results/apex_scored_v3/apex_scored_v3_benchmarks.csv
v3/results/apex_scored_v3/apex_scored_v3_combined.csv
v3/results/apex_scored_v3/apex_scored_v3_vs_oracle.csv
v3/results/apex_scored_v3/apex_top_v3_candidates.fasta
v3/results/apex_scored_v3/apex_scoring_summary.json
```

Main APEX-aware final-panel outputs:

```text
v3/results/apex_scored_v3/apex_aware_ranked_v3.csv
v3/results/apex_scored_v3/apex_aware_top_panel_v3.csv
v3/results/apex_scored_v3/apex_aware_top_panel_v3.fasta
v3/results/apex_scored_v3/apex_aware_selection_summary.json
```

For ApexOracle-9 parent-variant scan:

```bash
bash v3/run_parent_scan_apex9.sh
```

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
bash v3/run_v3_hybrid.sh
```

If your local APEX checkout exists at `$APEX_ROOT` or `/home/julojays/apex`, `run_v3_hybrid.sh` now automatically scores the top v3 panel with the APEX MIC ensemble after top-panel export.

To run only the v3 APEX/MIC scoring step after candidates already exist:

```bash
export APEX_ROOT=/home/julojays/apex
bash v3/run_score_v3_apex.sh
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

For ApexOracle-9 parent-variant scan:

```bash
bash v3/run_parent_scan_apex9.sh
```

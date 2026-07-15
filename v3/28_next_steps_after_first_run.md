# Next steps after the first real v3 run

After `bash v3/run_v3_hybrid.sh` finishes on the real ~20k AMP corpus:

1. inspect `v3/results/top_panel_v3.csv`;
2. inspect `v3/results/diverse_panel_v3.csv` if generated;
3. run external APEX/hemolysis/toxicity scoring if available;
4. merge external scores using `v3/16_merge_external_scores.py`;
5. export candidate FASTA using `v3/11_export_candidates_fasta.py`;
6. select a top 20-50 candidate panel.

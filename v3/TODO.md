# v3 TODO

Immediate next steps:

1. Put the current 20k curated AMP FASTA/CSV at `v3/data/raw/peptides.fasta` or edit `INPUTS` in `v3/run_v3_hybrid.sh`.
2. Run the APEX-only smoke test:

```bash
bash v3/run_apex_smoke_test.sh
```

3. Run the full v3 pipeline:

```bash
bash v3/run_v3_hybrid.sh
```

4. Inspect:

```text
v3/results/ranked_candidates_v3.csv
v3/results/top_panel_v3.csv
v3/results/apex_comparator_v3.csv
```

5. Upgrade path:

- add ESM2 conditioning;
- add real APEX batch scoring;
- add toxicity / hemolysis predictors;
- add parent-peptide improvement mode;
- add cluster-held-out validation.

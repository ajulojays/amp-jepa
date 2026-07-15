# v3 usage

## 1. Install

```bash
pip install -r v3/requirements-v3.txt
```

## 2. Place input data

Default:

```text
v3/data/raw/peptides.fasta
```

Or edit `INPUTS` in:

```text
v3/run_v3_hybrid.sh
```

## 3. Smoke test

```bash
bash v3/run_apex_smoke_test.sh
```

This scores the bundled APEX table with v3 heuristic filters only.

## 4. Full run

```bash
bash v3/run_v3_hybrid.sh
```

## 5. Key outputs

```text
v3/results/ranked_candidates_v3.csv
v3/results/top_panel_v3.csv
v3/results/apex_comparator_v3.csv
```

## 6. Best early interpretation

Do not start with “v3 beats APEX.” Start with:

> AMP-JEPA-Hybrid v3 generates novel AMP-like candidates and prioritizes them using latent generation, physicochemical constraints, novelty, and external APEX-style comparator support.

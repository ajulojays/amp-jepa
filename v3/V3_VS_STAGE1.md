# v3 vs Stage 1 foundation

These are complementary tracks.

## v3: improve the original hybrid architecture

Purpose:

```text
train on curated ~20k AMPs → generate candidates → rank/filter → external comparator
```

Best early output:

```text
top_panel_v3.csv
```

## Stage 1 foundation: true JEPA embedding route

Purpose:

```text
large peptide corpus → teacher/student latent prediction → reusable AMP embeddings
```

Best early output:

```text
AMP-JEPA embeddings and embedding benchmarks
```

## Recommended order

1. Harden v3 first because it already produces candidates.
2. Use v3 candidates for APEX/hemolysis/toxicity filtering.
3. Use Stage 1 foundation later to improve the latent space and benchmarking.

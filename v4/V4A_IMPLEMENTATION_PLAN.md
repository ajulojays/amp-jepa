# AMP-JEPA-Hybrid V4A Implementation Plan

## Implementation target

V4A starts as an APEX-only optimization system around the existing V3 backbone.

The implementation goal is:

```text
APEX-only, no DBAASP,
whole-pool optimization,
failure-aware G-Rescue,
robustness testing,
Pareto final panel.
```

## Phase 0: Inputs

Expected inputs from V3:

```text
v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt
v3/data/processed/peptide_corpus_v3_qc_core.csv
v3/results/ranked_candidates_v3_qc_core.csv
v3/results/global_qc_core_apex_panel/global_qc_core_apex_passed_ranked.csv
```

Expected APEX root:

```text
/home/julojays/apex
```

## Phase 1: Seed pool generation

Goal:

Generate a large candidate pool using the trained V3 generator.

Recommended first run:

```text
50,000 to 200,000 raw samples
```

Outputs:

```text
v4/results/seed_pool/v4a_seed_candidates.csv
```

Minimum columns:

```text
candidate_id
sequence
length
net_charge_KR_minus_DE
hydrophobic_fraction
cysteine_count
tryptophan_count
aromatic_fraction
source_run
generation_temperature
```

## Phase 2: APEX scoring

Goal:

Score all seed candidates using APEX.

Outputs:

```text
v4/results/seed_pool/v4a_seed_apex_scored.csv
```

Required columns:

```text
APEX_mean_MIC
APEX_median_MIC
APEX_worst_MIC
organisms_MIC_le_32
organisms_MIC_le_64
organisms_MIC_le_80
organisms_MIC_le_128
mean_GN_pred_MIC
mean_GP_pred_MIC
```

## Phase 3: Landscape mapping

Goal:

Turn all scored candidates into a design landscape.

Outputs:

```text
v4/results/landscape/candidate_landscape.csv
```

Metrics:

- potency component
- breadth component
- worst-case component
- novelty component
- developability component
- robustness placeholder
- failure-mode flags
- candidate class

## Phase 4: Candidate class assignment

Candidate classes:

```text
A  Elite
B  Near-pass
C  High-novelty
D  Broad-spectrum
E  Worst-case robust
F  Developable
G  Rescue candidate
G5 True junk
```

Output:

```text
v4/results/landscape/candidate_class_assignments.csv
```

## Phase 5: Clustering

Goal:

Avoid motif collapse by organizing candidates into families.

Possible first-pass clustering:

- sequence identity/difflib similarity
- length + charge + hydrophobicity bins
- k-mer Jaccard similarity

Output:

```text
v4/results/landscape/candidate_clusters.csv
```

## Phase 6: Population-wide optimization

Goal:

Generate variants from A-F and G candidates.

Optimization operators:

```text
single_substitution
double_substitution
terminal_trim
terminal_extension
KR_charge_tuning
ILV_hydrophobic_tuning
aromatic_reduction
cysteine_control
motif_preserving_mutation
```

Output:

```text
v4/results/optimization/optimized_variants.csv
```

## Phase 7: G-Rescue

Goal:

Diagnose and repair promising failed candidates.

Outputs:

```text
v4/results/rescue/g_rescue_candidates.csv
v4/results/rescue/g_rescue_failure_modes.csv
v4/results/rescue/g_rescue_variants.csv
```

Success output:

```text
v4/results/rescue/g_rescue_successes.csv
```

## Phase 8: Rescore optimized variants

Goal:

Run APEX on optimized and rescued variants.

Output:

```text
v4/results/optimization/optimized_variants_apex_scored.csv
```

## Phase 9: Robustness testing

Goal:

Stress-test final candidates by generating local variants and asking if the local neighborhood remains strong.

Robustness variant types:

```text
KR_swap
ILV_swap
single_neutral_substitution
terminal_trim
terminal_extension
aromatic_reduction_variant
charge_density_variant
```

Output:

```text
v4/results/robustness/robustness_scores.csv
```

Example robustness metric:

```text
local_robustness_score = number of acceptable local variants / number of local variants tested
```

## Phase 10: Pareto final panel

Goal:

Select a final balanced panel.

Outputs:

```text
v4/results/final_panel/v4a_pareto_front.csv
v4/results/final_panel/v4a_top20_panel.csv
v4/results/final_panel/v4a_top50_panel.csv
v4/results/final_panel/v4a_top50_panel.fasta
```

Panel categories:

- best overall
- best predicted median MIC
- best predicted worst-case MIC
- best predicted organism breadth
- best high-novelty candidate
- best short/simple candidate
- best low-W candidate
- best low-C candidate
- best G-Rescue success
- best balanced developability candidate

## Recommended implementation order

1. Create seed pool wrapper from existing V3 generation scripts.
2. Reuse existing APEX scoring script for seed candidates.
3. Build candidate class/failure-mode mapper.
4. Build simple sequence-level optimization operators.
5. Build G-Rescue operator logic.
6. Rescore variants with APEX.
7. Add robustness scoring.
8. Add Pareto selection.

## Notes

- Do not claim experimental MIC validation.
- Always write APEX-predicted MIC, not true MIC.
- Keep DBAASP out of V4A for now.
- Preserve all intermediate tables for auditability.
- Report failures as information, not only as discarded junk.

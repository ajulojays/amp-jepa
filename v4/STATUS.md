# V4A Status

## Current status

AMP-JEPA-Hybrid V4A has progressed from an architecture plan to a completed full-scale APEX-guided computational optimization run.

```text
Design: documented
Config: implemented
Full-scale runner: implemented
Chunked APEX scoring: implemented
Population optimization: completed
G-Rescue: completed
Pareto selection: completed
Elite/Pareto/spectrum annotation: completed
Experimental validation: not yet performed
```

## Architecture

```text
V3 latent AMP generator
+ whole-population candidate landscape mapping
+ APEX-guided multiobjective optimization
+ failure-aware G-Rescue
+ parent-to-variant gain analysis
+ independent Elite and Pareto labels
+ potency and spectrum classification
+ final lead-panel selection
```

## Full-scale V4A findings

- V3-derived seed sequences: **20,000**
- Optimized/rescued variants scored: **6,795**
- Local optimization successes: **4,399**
- Local optimization success rate: **64.7%**
- G-Rescue successes: **1,781**
- Combined seed-plus-variant candidates: **26,795**
- Final sanity-filtered candidate pool: **18,745**
- Elite candidates: **177**
- Pareto candidates: **76**
- Elite-Pareto candidates: **23**
- Potent against at least one organism: **507**
- Narrow-spectrum potent specialists: **1**
- Broad-spectrum candidates: **1,334**

The full findings report is available at:

```text
v4/V4A_FINDINGS_2026-07-16.md
```

## Current candidate taxonomy

Elite and Pareto are independent labels.

```text
                Optimized variants
                      |
        +-------------+-------------+
        |                           |
   Elite candidates          Pareto candidates
        |                           |
        +-------------+-------------+
                      |
          Lead candidates for synthesis
```

Additional independent labels include:

```text
is_potent_any_organism
is_narrow_spectrum_specialist
is_broad_spectrum
is_elite_pareto
```

## Current definitions

```python
is_potent_any_organism = APEX_best_MIC < 5

is_narrow_spectrum_specialist = (
    APEX_best_MIC < 5
    and fraction_MIC_le_64 <= 0.35
)

is_broad_spectrum = fraction_MIC_le_64 >= 0.70
```

Elite status currently requires:

```text
APEX_best_MIC <= 20
APEX_mean_MIC <= 80
APEX_median_MIC <= 32
APEX_worst_MIC <= 512
fraction_MIC_le_64 >= 0.60
developability_component >= 0.70
```

## Representative V4A lead

`IHWILIKKSLAKL`

```text
APEX best MIC:      2.661
APEX mean MIC:     54.015
APEX median MIC:   18.791
APEX worst MIC:   375.346
organisms <=64:        29
V4A score:          0.876
```

This sequence arose from a `B_near_pass` parent through `worst_case_charge_tuning+conservative` optimization.

## DBAASP status

DBAASP is intentionally excluded from V4A. Harmonized DBAASP material is preserved for a future real-label layer, but V4A remains APEX-only.

## Scientific framing

V4A must be described as an **APEX-guided computational optimization system**.

Use:

```text
APEX-predicted MIC
APEX-guided optimization
computational AMP lead panel
in silico candidate landscape
local optimization success
```

Do not use:

```text
experimentally validated MIC
confirmed antimicrobial activity
clinically active peptide
```

## Immediate next work

1. Summarize gain distributions and operator success rates for the 4,399 local optimization successes.
2. Add Gram-negative and Gram-positive group-specific summary metrics.
3. Audit Elite-Pareto sequences for redundancy and training-set similarity.
4. Build a diverse synthesis-priority lead panel.
5. Plan matched experimental MIC and safety validation.

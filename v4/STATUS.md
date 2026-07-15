# V4A Status

## Current status

V4A has been formally added as the next architecture direction for AMP-JEPA.

Current implementation state:

```text
Design: documented
Config: drafted
Runner scaffold: added
Executable modules: pending implementation
```

## Files added

```text
v4/README.md
v4/V4A_ARCHITECTURE.md
v4/V4A_IMPLEMENTATION_PLAN.md
v4/G_RESCUE.md
v4/configs/v4a_apex_only.yaml
v4/run_v4a_fullscale.sh
```

## V4A architecture commitment

V4A is defined as:

```text
V3 latent AMP generator
+ whole-population candidate landscape mapping
+ APEX-guided multiobjective optimization
+ failure-aware G-Rescue
+ robustness stress testing
+ Pareto final lead-panel selection
```

## DBAASP status

DBAASP is intentionally not used in V4A.

DBAASP harmonized batch 1 is preserved separately for future V5 or later activity-aware training.

## Immediate next work

Implement the V4A modules in this order:

```text
00_generate_v4a_seed_pool.py
01_score_v4a_seed_pool_with_apex.py
02_map_candidate_landscape.py
03_cluster_candidate_families.py
04_diagnose_failure_modes.py
05_optimize_candidate_population.py
06_score_v4a_optimized_variants.py
07_compute_local_robustness.py
08_select_v4a_pareto_panel.py
```

## Scientific framing

V4A should always be described as an **APEX-guided computational optimization system**.

Use:

```text
APEX-predicted MIC
APEX-guided optimization
computational AMP lead panel
in silico candidate landscape
```

Do not use:

```text
experimentally validated MIC
confirmed antimicrobial activity
clinically active peptide
```

# AMP-JEPA-Hybrid V4B

V4B extends the frozen V4A milestone into an iterative, self-improving latent design loop.

## Core cycle

```text
JEPA latent manifold
      ↓
Generate peptides
      ↓
APEX-guided sequence optimization
      ↓
Re-encode optimized peptides
      ↓
Update the latent manifold
      ↓
Generate again
```

V4B adds safety-aware multiobjective design:

- APEX-predicted organism-specific MIC
- overall, Gram-negative, and Gram-positive potency/breadth
- hemolysis prediction
- cytotoxicity prediction
- sequence novelty
- developability
- local robustness
- Elite and Pareto status
- potent-any-organism and spectrum labels

## Scientific definition

> AMP-JEPA-Hybrid V4B is an iterative latent-manifold optimization system that learns from its own optimized descendants while jointly balancing predicted antimicrobial activity, hemolysis, cytotoxicity, breadth, novelty, developability, and robustness.

V4B does not replace V4A. V4A is the frozen baseline and candidate archive. V4B starts from the V4A candidate pool and optimization history.

## Data policy

For now, activity remains APEX-guided. Hemolysis and cytotoxicity predictors must be treated as separate prediction modules with model provenance recorded. Missing safety labels must remain missing; they must not be silently interpreted as safe.

## V4B generations

Each iteration is a generation:

```text
Generation 0: frozen V4A seeds, optimized variants, Elite/Pareto candidates
Generation 1: re-encoded V4A descendants and newly decoded peptides
Generation 2+: descendants selected from multiobjective safety/activity fitness
```

Every generated sequence must retain:

- generation number
- parent sequence and parent ID
- source class
- latent vector ID
- optimization operator
- activity predictions
- safety predictions
- novelty/developability metrics
- Elite/Pareto/specialist labels

## Candidate taxonomy

Elite, Pareto, and potent-specialist labels remain independent.

```text
                    V4B descendants
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
      Elite              Pareto        Potent specialist
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
             safety-qualified lead panel
```

A candidate can be Elite only, Pareto only, specialist only, or any intersection.

## Safety-aware lead principle

A peptide is not called safety-qualified merely because safety predictions are missing. A lead requires explicit model outputs or experimentally measured values for the safety criteria being claimed.

## Planned modules

```text
v4b/
├── README.md
├── V4B_ARCHITECTURE.md
├── V4B_IMPLEMENTATION_PLAN.md
├── configs/v4b_apex_safety.yaml
├── 00_import_frozen_v4a.py
├── 01_encode_v4a_descendants.py
├── 02_fit_activity_safety_surrogates.py
├── 03_update_latent_manifold.py
├── 04_generate_next_generation.py
├── 05_score_activity.py
├── 06_score_hemolysis_toxicity.py
├── 07_select_multitask_elite_pareto.py
├── 08_build_next_generation.py
└── run_v4b_generation.sh
```

## Current status

Architecture and implementation plan are now defined. The next engineering step is to implement Generation 0 import and latent re-encoding using the frozen V4A candidates and the V3 encoder checkpoint.

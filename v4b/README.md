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

V4B runs for ten generations and optimizes:

- APEX-predicted organism-specific MIC
- overall, Gram-negative and Gram-positive potency/breadth
- sequence novelty
- developability
- local robustness
- latent and sequence diversity
- Elite and Pareto status
- potent-any-organism and spectrum labels

Hemolysis and cytotoxicity are excluded from manifold evolution. They will be applied only to the final shortlisted panel before synthesis.

## Scientific definition

> AMP-JEPA-Hybrid V4B is a ten-generation latent-manifold evolution system that learns from optimized descendants while balancing predicted antimicrobial activity, spectrum, novelty, developability, robustness and diversity.

V4A remains the frozen baseline and Generation 0 archive.

## Generations

```text
Generation 0: frozen V4A seeds and optimized variants
Generation 1: re-encoded V4A descendants and newly decoded peptides
Generation 2–9: activity-guided descendant populations
Generation 10: final evolved computational population
```

Every sequence retains:

- generation number
- parent sequence and parent ID
- source class
- latent vector ID
- optimization operator
- activity predictions
- novelty and developability metrics
- robustness metrics
- Elite, Pareto and specialist labels

## Candidate taxonomy

Elite, Pareto and specialist labels remain independent.

```text
                    V4B descendants
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
      Elite              Pareto        Potent specialist
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
               diverse final lead panel
```

## Final validation before synthesis

After Generation 10, a small structurally diverse lead panel will undergo separate validation for hemolysis, cytotoxicity, stability, solubility, aggregation and experimental MIC. These endpoints are final selection gates rather than V4B optimization objectives.

## Planned modules

```text
v4b/
├── README.md
├── WORKFLOW.md
├── V4B_ARCHITECTURE.md
├── configs/v4b_apex.yaml
├── 00_import_frozen_v4a.py
├── 01_encode_v4a_descendants.py
├── 02_fit_activity_surrogate.py
├── 03_update_latent_manifold.py
├── 04_generate_next_generation.py
├── 05_score_activity.py
├── 06_score_design_quality.py
├── 07_select_elite_pareto.py
├── 08_build_next_generation.py
└── run_v4b_generation.sh
```

## Current status

The V4A archive is frozen. The next engineering step is Generation 0 import and latent re-encoding using the frozen V4A candidates and V3 encoder checkpoint.

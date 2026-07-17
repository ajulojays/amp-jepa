# AMP-JEPA V4B workflow

## Scope

V4B is a ten-generation closed-loop extension of V4A. It re-encodes successful descendants, updates the learned peptide manifold, generates again, and optimizes antimicrobial activity while preserving novelty, developability, robustness, spectrum diversity, and latent diversity.

V4A remains frozen as Generation 0.

Hemolysis and cytotoxicity are **not** V4B optimization objectives. They are reserved for final validation of the shortlisted synthesis panel after Generation 10.

```text
V4A Generation 0
      ↓
Re-encode selected candidates into latent μ
      ↓
Generate latent-neighborhood offspring
      ↓
Sequence optimization and G-Rescue
      ↓
APEX organism-specific activity scoring
      ↓
Novelty, developability and robustness scoring
      ↓
Elite, Pareto, spectrum and specialist classification
      ↓
Select diverse parents
      ↓
Update/retrain latent manifold
      ↓
Generation n + 1
```

## Ten-generation protocol

```text
Generation 0  = frozen V4A
Generation 1  = first descendant population
Generation 2
Generation 3
Generation 4
Generation 5
Generation 6
Generation 7
Generation 8
Generation 9
Generation 10 = final evolved population
```

Every generation is archived and never overwritten.

## Per-generation objectives

### Activity

- APEX best MIC
- APEX mean MIC
- APEX median MIC
- APEX worst MIC
- fraction of organisms with predicted MIC ≤20, ≤32 and ≤64
- Gram-negative potency and breadth
- Gram-positive potency and breadth

### Design quality

- sequence novelty
- nearest-neighbor identity
- length, charge and hydrophobicity
- developability
- local sequence robustness
- local latent robustness
- sequence-cluster diversity
- latent-cluster diversity

### Candidate taxonomy

- `is_optimization_success`
- `is_g_rescue`
- `is_elite`
- `is_pareto`
- `is_elite_pareto`
- `is_potent_any_organism`
- `is_narrow_spectrum_specialist`
- `is_broad_spectrum`
- `is_lead`

Elite and Pareto remain parallel classifications. Specialist and spectrum labels remain independent.

## Parent selection

The next generation is assembled from a controlled mixture of:

- Elite candidates
- Pareto candidates
- Elite-Pareto candidates
- potent specialists
- G-Rescue successes
- structurally diverse V4A descendants
- random exploration candidates

Family caps, sequence-cluster quotas, latent-cluster quotas and a novelty floor are required to prevent collapse.

## Required outputs for every generation

```text
generation_XX/
├── generation_manifest.json
├── generated_candidates.csv
├── optimized_candidates.csv
├── activity_predictions.csv
├── candidate_lineage.csv
├── latent_vectors.npz
├── candidate_groups.csv
├── elite_candidates.csv
├── pareto_candidates.csv
├── elite_pareto_candidates.csv
├── specialists.csv
├── next_generation_parents.csv
├── model_checkpoint.pt
└── metrics.json
```

## Final validation gate before synthesis

Only after Generation 10 will a small, diverse synthesis panel be evaluated for:

- hemolysis
- mammalian-cell cytotoxicity
- serum/plasma stability
- protease stability
- solubility and aggregation
- organism-specific experimental MIC

These final validation results do not retroactively define the V4B training objective. They determine which computational leads proceed to synthesis and wet-lab testing.

## Scientific boundary

APEX values are model predictions, not experimental MIC measurements. Candidates should be described as predicted-active, predicted broad-spectrum, predicted specialist, Elite, Pareto, or computational leads until experimentally validated.

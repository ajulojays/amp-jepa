# V4B Architecture Specification

## Frozen baseline

V4A is immutable input to V4B. V4B must never overwrite V4A result tables. Generation 0 imports copies and records source hashes.

## Iterative latent loop

```text
Frozen V4A population
      ↓
Encode full sequences with AMP-JEPA encoder
      ↓
Construct Generation-g latent table (mu, logvar, z)
      ↓
Attach activity and design-quality phenotype vectors
      ↓
Select diverse parents from Elite, Pareto, specialists and G-Rescue successes
      ↓
Update latent manifold / fit conditional latent proposal model
      ↓
Decode Generation-(g+1) sequences
      ↓
APEX activity scoring + design-quality scoring
      ↓
Multiobjective classification and selection
      ↓
Re-encode selected descendants
      ↺
```

The loop runs from frozen Generation 0 through Generation 10.

## Phenotype vector

Each peptide is represented by a multiobjective phenotype vector.

### Activity

- APEX_best_MIC
- APEX_mean_MIC
- APEX_median_MIC
- APEX_worst_MIC
- fraction_MIC_le_20/32/64
- Gram-negative min/mean/median/worst/breadth
- Gram-positive min/mean/median/worst/breadth

### Design quality

- novelty and nearest-neighbor identity
- length, charge and hydrophobicity
- cysteine, tryptophan and aromatic burden
- developability score
- local sequence robustness
- local latent robustness
- sequence-cluster membership
- latent-cluster membership

Hemolysis and cytotoxicity are not phenotype objectives during V4B evolution. They are final validation gates for the shortlisted Generation-10 synthesis panel.

## Latent update options

### V4B-1: weighted latent resampling

Encode selected parents, estimate weighted centers/covariances, and sample near high-fitness but diverse regions. This is the first implementation because it is transparent and robust.

### V4B-2: conditional latent proposal network

Train a proposal model to predict latent shifts from current phenotype and desired phenotype. The decoder remains the V3/V4 backbone.

### V4B-3: joint encoder/decoder refinement

Fine-tune the latent model using selected descendants while retaining reconstruction, KL, JEPA consistency and property losses. Use replay from the original QC-core corpus to reduce catastrophic forgetting.

## Multiobjective selection

Elite and Pareto are independent.

Pareto objectives include activity, breadth, novelty, developability and robustness.

Specialist labels remain independent:

```python
is_potent_any_organism = APEX_best_MIC < 5
is_narrow_spectrum_specialist = (
    (APEX_best_MIC < 5)
    & (fraction_MIC_le_64 <= 0.35)
)
is_broad_spectrum = fraction_MIC_le_64 >= 0.70
```

## Preventing optimizer collapse

V4B must preserve:

- sequence-cluster quotas
- latent-cluster quotas
- parent-family caps
- novelty floor
- replay of V4A and QC-core peptides
- random exploration fraction
- specialist preservation
- G-Rescue lineage preservation

## Required audit trail

Each generation writes:

```text
generation_manifest.json
candidate_lineage.csv
latent_vectors.npz
activity_predictions.csv
design_quality.csv
candidate_groups.csv
elite_candidates.csv
pareto_candidates.csv
elite_pareto_candidates.csv
specialists.csv
next_generation_parents.csv
model_checkpoint.pt
metrics.json
```

## Final validation gate

After Generation 10, select a small panel using activity, novelty, structural diversity and robustness. Evaluate hemolysis, cytotoxicity, stability, solubility, aggregation and experimental MIC before synthesis prioritization.

## Scientific boundary

All APEX MIC values are predictions and must not be described as experimental measurements. Hemolysis and cytotoxicity are not inferred from the V4B manifold and must be measured or assessed separately during final validation.

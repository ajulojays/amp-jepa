# AMP-JEPA-Hybrid V4A

**AMP-JEPA-Hybrid V4A** is the APEX-only, full-scale optimization extension of the AMP-JEPA-Hybrid v3 latent peptide generator.

V4A does not replace the v3 backbone. It turns v3 into a full design system.

```text
V3:
  generate AMP-like candidates -> APEX screen -> select top hits

V4A:
  generate candidates -> score all candidates -> map the landscape -> diagnose failures -> optimize/rescue -> rescore -> stress-test -> classify candidates -> select leads
```

## Core definition

> AMP-JEPA-Hybrid V4A extends the v3 latent AMP generator into a population-wide, failure-aware, optimization-guided peptide design system using APEX-predicted multi-organism activity and multiobjective Pareto selection.

V4A is intentionally **APEX-only for now**. DBAASP is preserved as a future real-label layer, but it is not required for this version.

## Why V4A exists

V3 already showed that the QC-core corpus improved generated candidate quality. However, v3 mostly performs broad latent sampling followed by post-generation screening.

V4A asks a stronger biological question:

```text
Can the entire generated peptide landscape be understood, repaired, optimized, and stress-tested?
```

This makes V4A different from a simple top-hit selector. V4A treats failed candidates as biological information. Weak candidates may contain recoverable motifs, immature scaffolds, or correctable physicochemical imbalance.

## Design philosophy

V4A follows the principle:

```text
Out of ashes comes beauty.
```

A low-scoring candidate is not automatically discarded. It is diagnosed first.

A candidate may fail because it is:

- too hydrophobic
- too weakly cationic
- too strongly cationic
- too short or too long
- too similar to training sequences
- too narrow in predicted organism coverage
- poor only in worst-case predicted MIC
- chemically/developability imbalanced
- locally fragile but biologically promising

Only true degenerate junk is discarded immediately.

## V4A system architecture

```text
QC-core trained AMP-JEPA-Hybrid v3
          |
          v
Large seed candidate generation
          |
          v
APEX full-panel scoring
          |
          v
Whole-population landscape mapping
          |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
Elite refinement          Near-pass rescue          G-Rescue
          |                      |                      |
          +----------------------+----------------------+
                                 |
                                 v
                    Optimized variant pool
                                 |
                                 v
                    APEX rescoring
                                 |
                                 v
                    Independent candidate labels
                                 |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
   Elite candidates       Pareto candidates      Potent specialists
          |                      |                      |
          +----------------------+----------------------+
                                 |
                                 v
                    Lead candidates for synthesis
```

Elite, Pareto, and potent-specialist status are independent labels. A candidate may belong to one, two, or all three groups.

## Candidate classes

V4A classifies generated candidates into design roles:

| Class | Meaning | Action |
|---|---|---|
| A | Elite candidates | refine and preserve |
| B | Near-pass candidates | rescue failed criterion |
| C | High-novelty candidates | improve activity while preserving novelty |
| D | Broad-spectrum candidates | improve potency/developability |
| E | Worst-case robust candidates | improve median/mean while preserving robustness |
| F | Developable candidates | improve activity while preserving clean properties |
| G | Rescue candidates | diagnose and repair hidden biological structure |
| G5 | True junk | discard |

## Independent candidate labels

The final selector writes the following Boolean labels to every sanity-filtered candidate:

```python
df["is_potent_any_organism"] = (
    df["APEX_best_MIC"] < 5
)

df["is_narrow_spectrum_specialist"] = (
    (df["APEX_best_MIC"] < 5)
    & (df["fraction_MIC_le_64"] <= 0.35)
)

df["is_broad_spectrum"] = (
    df["fraction_MIC_le_64"] >= 0.70
)
```

Interpretation:

- `is_potent_any_organism`: predicted MIC below 5 against at least one organism.
- `is_narrow_spectrum_specialist`: predicted MIC below 5 against at least one organism but predicted MIC <=64 against no more than 35% of the panel.
- `is_broad_spectrum`: predicted MIC <=64 against at least 70% of the panel.
- `is_elite`: passes absolute potency, breadth, worst-case, and developability thresholds.
- `is_pareto`: is non-dominated across the V4A multiobjective selection space.
- `is_elite_pareto`: is both elite and Pareto.

A potent-any candidate is not automatically narrow-spectrum. Broad-spectrum candidates can also have `APEX_best_MIC < 5`.

## Default elite thresholds

The current defaults are configurable from the command line:

```text
APEX_best_MIC <= 20
APEX_mean_MIC <= 80
APEX_median_MIC <= 32
APEX_worst_MIC <= 512
fraction_MIC_le_64 >= 0.60
developability_component >= 0.70
```

Elite and Pareto remain parallel labels rather than sequential filters.

## Optimization objectives

V4A optimizes multiple objectives at once:

- minimize APEX-predicted median MIC
- minimize APEX-predicted mean MIC
- minimize APEX-predicted worst MIC
- maximize number of organisms with predicted MIC <= 64
- maximize novelty against the training corpus
- maximize sequence diversity
- maximize local robustness
- keep length biologically reasonable
- keep charge biologically reasonable
- keep hydrophobicity biologically reasonable
- avoid excessive cysteine burden
- avoid excessive tryptophan/aromatic burden
- avoid simple memorization of known training peptides

V4A is not a lowest-MIC-at-all-cost system. It is a robust biological lead-design system.

## Output directories

```text
v4/results/
├── seed_pool/
│   ├── v4a_seed_candidates.csv
│   └── apex_seed_scoring/
├── landscape/
│   └── candidate_landscape.csv
├── rescue/
│   ├── g_rescue_variants.csv
│   └── g_rescue_successes.csv
├── optimization/
│   ├── optimized_variants.csv
│   ├── optimized_variants_with_gains.csv
│   └── apex_optimized_scoring/
├── robustness/
│   └── robustness_scores.csv
└── final_panel/
    ├── v4a_all_sanity_filtered_candidates.csv
    ├── v4a_elite_candidates.csv
    ├── v4a_pareto_front.csv
    ├── v4a_elite_pareto_candidates.csv
    ├── v4a_potent_any_organism.csv
    ├── v4a_narrow_spectrum_specialists.csv
    ├── v4a_broad_spectrum_candidates.csv
    ├── v4a_top20_panel.csv
    ├── v4a_top50_panel.csv
    └── v4a_top50_panel.fasta
```

## Version ladder

```text
V3:
  latent AMP generator + APEX post-screening

V4A:
  V3 backbone + APEX-guided full-population optimization + G-Rescue

V5:
  future activity-aware training with DBAASP/real-label integration
```

## Current status

V4A is implemented as an APEX-only, whole-population optimization system with failure-aware G-Rescue, independent Elite/Pareto/spectrum labels, and synthesis-oriented output panels.

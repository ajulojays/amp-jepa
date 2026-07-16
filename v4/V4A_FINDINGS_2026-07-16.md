# AMP-JEPA-Hybrid V4A Findings

**Status date:** 2026-07-16  
**Branch:** `v3-hybrid-improved`  
**Scope:** APEX-guided computational optimization; no DBAASP supervision in V4A.

## Executive finding

V4A successfully extended the V3 generator into a population-wide peptide optimization system. It generated, rescored, improved, rescued, and multiobjective-ranked candidate peptides rather than relying only on broad random generation followed by screening.

All activity values below are **APEX-predicted MIC values**, not experimentally measured MICs.

## Full-scale run

- V3-derived seed sequences: **20,000**
- Optimized/rescued variants scored: **6,795**
- Local optimization successes: **4,399**
- G-Rescue successes: **1,781**
- Combined seed-plus-variant candidates: **26,795**
- Candidates retained after final sequence sanity filters: **18,745**
- Pareto-front candidates: **76**

The local optimization success rate among scored variants was approximately **64.7%** (`4,399 / 6,795`). A local optimization success means that a variant improved relative to its parent under the implemented gain criteria; it does not automatically mean that the variant is Elite or Pareto-optimal.

## Candidate taxonomy

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

A candidate may be Elite only, Pareto only, both, or neither.

### Current group counts

- Candidate pool: **18,745**
- Elite candidates: **177**
- Pareto candidates: **76**
- Elite-Pareto candidates: **23**
- Potent against at least one organism: **507**
- Narrow-spectrum potent specialists: **1**
- Broad-spectrum candidates: **1,334**

### Potency and spectrum definitions

```python
is_potent_any_organism = APEX_best_MIC < 5

is_narrow_spectrum_specialist = (
    APEX_best_MIC < 5
    and fraction_MIC_le_64 <= 0.35
)

is_broad_spectrum = fraction_MIC_le_64 >= 0.70
```

The low narrow-specialist count reflects both the strict breadth cutoff and V4A's current optimization pressure toward broad multi-organism activity. It should not be interpreted as evidence that only one candidate has target selectivity under every possible definition.

## Elite definition

An Elite candidate must pass all current absolute thresholds:

- `APEX_best_MIC <= 20`
- `APEX_mean_MIC <= 80`
- `APEX_median_MIC <= 32`
- `APEX_worst_MIC <= 512`
- `fraction_MIC_le_64 >= 0.60`
- `developability_component >= 0.70`

The developability component favors:

- length: 8-35 residues
- net charge: +2 to +10
- hydrophobic fraction: 0.25-0.62
- cysteine count: no more than 2 before penalty
- tryptophan count: no more than 3 before penalty
- aromatic fraction: no more than 0.30 before penalty

## Pareto result

The Pareto front contains candidates for which no alternative candidate is at least as good across all selected objectives and strictly better in at least one. V4A currently minimizes APEX-predicted mean, median, and worst MIC while maximizing breadth, novelty, and developability.

Pareto membership does not automatically imply Elite status. The **23 Elite-Pareto candidates** are currently the most defensible primary lead group because they pass absolute quality thresholds and also represent non-dominated tradeoffs.

## Representative leading candidates

### Highest composite V4A score

`IHWILIKKSLAKL`

- source: optimized/rescued variant
- parent class: `B_near_pass`
- operator: `worst_case_charge_tuning+conservative`
- APEX best MIC: **2.661**
- APEX mean MIC: **54.015**
- APEX median MIC: **18.791**
- APEX worst MIC: **375.346**
- organisms with MIC <=64: **29**
- V4A score: **0.876**

### Strong worst-case robustness

`GIIGLIKTVSKLIKHT`

- APEX mean MIC: **45.657**
- APEX median MIC: **31.934**
- APEX worst MIC: **153.394**
- organisms with MIC <=64: **27**

### Optimized descendant of the prior V3 lead region

V3 lead: `RLLSISSKLLSRL`  
V4A optimized variant: `RLLSISSKLISRL`

The V4A variant retained broad coverage while reducing the APEX-predicted worst MIC from approximately **210.46** to **167.24**, with similar mean activity and a modest median-MIC tradeoff.

## Interpretation relative to V3

Within the shared APEX-guided computational framework, V4A improved on V3 as a design system by adding:

- population-wide optimization
- parent-to-variant gain tracking
- failure-aware G-Rescue
- independent Elite and Pareto labels
- specialist and broad-spectrum labels
- multiobjective lead selection

The evidence supports the statement that **V4A outperformed V3 computationally within the APEX scoring framework**. It does not yet establish experimentally superior antimicrobial activity.

## Comparison caution

Comparisons with APEX-GO or ApexOracle must distinguish:

- APEX-predicted MIC from experimental MIC
- overall summary metrics from organism-specific metrics
- uncensored values from experimental `>64` right-censored values

No claim of experimental superiority should be made until V4A leads are synthesized and tested under matched assay conditions.

## Primary output files

```text
v4/results/final_panel/v4a_candidate_groups_all.csv
v4/results/final_panel/v4a_elite_candidates.csv
v4/results/final_panel/v4a_pareto_candidates_annotated.csv
v4/results/final_panel/v4a_elite_pareto_candidates.csv
v4/results/final_panel/v4a_potent_any_organism.csv
v4/results/final_panel/v4a_narrow_spectrum_specialists.csv
v4/results/final_panel/v4a_broad_spectrum_candidates.csv
v4/results/final_panel/v4a_candidate_group_summary.json
v4/results/rescue/g_rescue_successes.csv
v4/results/optimization/optimized_variants_with_gains.csv
```

## Immediate next analyses

1. Summarize all 4,399 local optimization successes by operator and gain distribution.
2. Add separate Gram-negative and Gram-positive min, mean, median, worst, and breadth metrics.
3. Audit the 23 Elite-Pareto candidates for sequence-family redundancy and nearest-training similarity.
4. Build a small, diverse lead panel representing balanced, broad-spectrum, robust, novelty-first, and target-selective design roles.
5. Treat experimental MIC, hemolysis, cytotoxicity, and stability testing as the next evidence layer.

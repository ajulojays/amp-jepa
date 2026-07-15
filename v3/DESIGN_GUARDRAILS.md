# v3 design guardrails

1. v3 is a candidate-generation architecture, not a clinical claim.
2. APEX/ApexOracle is an external comparator, not wet-lab ground truth.
3. Candidate ranking should combine novelty, AMP-like constraints, and diversity; do not rank only by model generation probability.
4. Near-duplicates of training peptides should be flagged, not celebrated as novel.
5. Generated candidates should be treated as hypotheses requiring experimental validation.
6. The next strongest upgrade is not more generations; it is better filters: hemolysis, cytotoxicity, serum stability, and salt tolerance.

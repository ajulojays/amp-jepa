# AMP-JEPA-Hybrid v3 method summary

## Core claim

AMP-JEPA-Hybrid v3 is not positioned as a pure JEPA foundation model. It is positioned as an improved predictive-generative AMP design architecture derived from the original AMP-JEPA pilot.

## Input

A curated AMP sequence corpus, ideally 10k-50k validated or high-confidence AMP sequences.

## Model

The model learns a latent AMP design space using:

- Transformer encoder;
- VAE latent bottleneck;
- sequence reconstruction decoder;
- masked-view latent prediction loss;
- physicochemical regularization.

## Candidate selection

Generated peptides are filtered and ranked by:

- sequence validity;
- length;
- net charge;
- hydrophobic fraction;
- novelty against training peptides;
- AMP-like developability constraints;
- optional APEX/ApexOracle support.

## Experimental interpretation

A v3 candidate should be called computationally promising only if it satisfies multiple criteria:

1. produced by AMP-JEPA-Hybrid v3;
2. passes design filters;
3. is not a near-duplicate of training AMPs;
4. is supported by an external oracle or comparator;
5. is part of a diverse candidate panel.

APEX support is not experimental validation.

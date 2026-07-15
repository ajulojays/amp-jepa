# v3 implementation notes

## Architecture

AMP-JEPA-Hybrid v3 keeps the useful idea from the earlier pilot: a compact generative AMP model can already learn useful design signals from about 20k curated peptide sequences.

The v3 model contains:

1. a Transformer sequence encoder;
2. a VAE latent space;
3. a non-autoregressive sequence decoder;
4. a JEPA-inspired masked-view latent prediction term;
5. a physicochemical property regularizer;
6. post-generation novelty and developability filters.

## Loss

```text
loss = reconstruction
     + beta_kl * KL
     + jepa_weight * masked_view_latent_MSE
     + property_weight * physicochemical_feature_MSE
```

## Why this is not the same as Stage 1 foundation

Stage 1 foundation is the long-term pure embedding route. v3 is the fast design route:

```text
v3: generate and rank candidates now
Stage 1 foundation: learn a more rigorous AMP-specific JEPA embedding space later
```

## Next upgrades

- plug in ESM2 embeddings as conditioning features;
- add a real hemolysis/cytotoxicity predictor;
- add APEX batch-scoring integration when available locally;
- add family/cluster-held-out evaluation;
- add parent-peptide variant optimization mode.

# Next v3 upgrade: ESM2 conditioning

The current v3 scaffold is sequence-only so it can run immediately. The next improvement should add optional ESM2 embeddings as a conditioning vector:

```text
sequence tokens ───────► sequence encoder ──► VAE latent
ESM2 embedding ─────────► projection ─────────┘
```

Recommended implementation:

1. export ESM2 mean-pooled peptide embeddings to NPZ;
2. add `--esm2-npz` to `01_train_v3_hybrid.py`;
3. concatenate projected ESM2 embedding with the sequence pooled embedding before `to_mu` and `to_logvar`;
4. compare sequence-only v3 vs ESM2-conditioned v3 on candidate quality and reconstruction/generation behavior.

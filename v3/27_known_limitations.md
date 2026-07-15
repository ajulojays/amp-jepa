# Known v3 limitations

- The first v3 model is sequence-only; ESM2 conditioning is noted as the next upgrade.
- The decoder is non-autoregressive and should be treated as a fast prototype generator.
- Heuristic novelty uses simple sequence identity, not full clustering or alignment.
- APEX is not called directly; the bundled table is only a comparator.
- No wet-lab validation is implied by v3 scores.

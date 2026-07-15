# v3 run order

For a quick wiring check:

```bash
pip install -r v3/requirements-v3.txt
bash v3/run_tiny_demo.sh
```

For a tiny debug train using the bundled APEX sequences only:

```bash
bash v3/run_tiny_debug_training.sh
```

For the real v3 run:

```bash
# first place your curated 20k AMP FASTA/CSV here or edit INPUTS in run_v3_hybrid.sh
# v3/data/raw/peptides.fasta
bash v3/run_v3_hybrid.sh
```

For ApexOracle-9 parent-variant scan:

```bash
bash v3/run_parent_scan_apex9.sh
```

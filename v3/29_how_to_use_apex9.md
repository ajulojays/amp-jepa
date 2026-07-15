# How ApexOracle-9 is used in v3

ApexOracle-9 from the uploaded APEX table is treated as a comparator anchor:

```text
IQKLKFLRLAAQAQKLLLKLGIARRSLASK
```

You can score it with v3 filters:

```bash
bash v3/run_score_apex9.sh
```

You can generate simple in silico nearby variants:

```bash
bash v3/run_parent_scan_apex9.sh
```

This does not mean v3 has validated or improved ApexOracle-9 experimentally. It only gives a fast computational comparison path.

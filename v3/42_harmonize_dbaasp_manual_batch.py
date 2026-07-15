#!/usr/bin/env python3
"""Harmonize manually downloaded DBAASP batch files for AMP-JEPA.

Expected input files in --input-dir:
  peptides.csv
  activity-against-target-species.csv
  hemolytic-and-cytotoxic-activities.csv
  peptides-antibiofilm-activities.csv
  peptides-fasta.txt

Outputs:
  dbaasp_peptides_master_harmonized.csv
  dbaasp_activity_target_species_long.csv
  dbaasp_mic_labels_long.csv
  dbaasp_hemolysis_cytotoxicity_long.csv
  dbaasp_antibiofilm_long.csv
  dbaasp_trainable_single_chain_sequences.csv
  dbaasp_trainable_single_chain_sequences.fasta
  dbaasp_fasta_id_sequence_map.csv
  dbaasp_harmonization_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_col(c: str) -> str:
    return str(c).strip().strip('"').strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [clean_col(c) for c in df.columns]
    return df


def clean_sequence(seq: Any) -> str:
    if pd.isna(seq):
        return ""
    return re.sub(r"\s+", "", str(seq).upper())


def is_canonical(seq: Any) -> bool:
    seq = clean_sequence(seq)
    return bool(seq) and all(aa in CANONICAL_AA for aa in seq)


def split_chains(seq: Any) -> list[str]:
    if pd.isna(seq):
        return []
    raw = str(seq).strip().upper()
    parts = [p.strip() for p in re.split(r"\s+", raw) if p.strip()]
    return [p for p in parts if is_canonical(p)]


def unit_norm(u: Any) -> str:
    if pd.isna(u) or str(u).strip() == "":
        return ""
    x = str(u).strip().replace("μ", "µ")
    low = x.lower().replace(" ", "")
    if low in {"µm", "um"}:
        return "uM"
    if low in {"µg/ml", "ug/ml", "µgperml", "ugperml"}:
        return "ug_per_ml"
    if low == "nmol/g":
        return "nmol_per_g"
    return x


def parse_numeric_value(x: Any) -> pd.Series:
    if pd.isna(x):
        return pd.Series({"value_num": np.nan, "value_qualifier": "", "value_raw": ""})
    raw = str(x).strip()
    if raw == "":
        return pd.Series({"value_num": np.nan, "value_qualifier": "", "value_raw": raw})
    qualifier = ""
    mqual = re.match(r"^\s*(<=|>=|<|>|≤|≥|=|~)", raw)
    if mqual:
        qualifier = mqual.group(1).replace("≤", "<=").replace("≥", ">=")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw.replace(",", "."))
    value = float(nums[0]) if nums else np.nan
    return pd.Series({"value_num": value, "value_qualifier": qualifier, "value_raw": raw})


def parse_percent_from_text(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    m = re.search(r"([-+]?\d*\.?\d+)\s*%", str(x))
    return float(m.group(1)) if m else np.nan


def measurement_class(measure: Any) -> str:
    if pd.isna(measure):
        return "unknown"
    s = str(measure).strip().upper()
    if s.startswith("MIC"):
        return "MIC"
    if s.startswith("MBC"):
        return "MBC"
    if s.startswith("MBIC"):
        return "MBIC"
    if s.startswith("MBEC"):
        return "MBEC"
    for key in ["IC50", "EC50", "LC50", "LD50", "LD90", "ED50", "CC50"]:
        if key in s:
            return key
    if "HEMOLYSIS" in s:
        return "hemolysis"
    if "CYTOTOX" in s:
        return "cytotoxicity"
    if "CELL DEATH" in s:
        return "cell_death"
    if "INHIBITION" in s:
        return "percent_inhibition"
    if s in {"-", "NA", "NAN"}:
        return "unknown"
    return "other"


def parse_target_taxon(target: Any) -> pd.Series:
    if pd.isna(target):
        return pd.Series({"target_genus": "", "target_binomial": ""})
    s = str(target).strip()
    words = re.findall(r"[A-Za-z][A-Za-z\.-]*", s)
    genus = words[0] if words else ""
    binomial = ""
    if len(words) >= 2 and words[1] and words[1][0].islower():
        binomial = f"{words[0]} {words[1]}"
    return pd.Series({"target_genus": genus, "target_binomial": binomial})


def parse_fasta(path: Path) -> pd.DataFrame:
    records = []
    header = None
    seq_lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, clean_sequence("".join(seq_lines))))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        records.append((header, clean_sequence("".join(seq_lines))))

    rows = []
    for h, seq in records:
        m = re.search(r"DBAASP([RS])_(\d+)", h)
        rows.append({
            "dbaasp_id": int(m.group(2)) if m else np.nan,
            "fasta_record_class": f"DBAASP{m.group(1)}" if m else "",
            "fasta_header": h,
            "fasta_name": re.sub(r"^DBAASP[RS]_\d+\s*", "", h).strip(),
            "sequence_fasta": seq,
            "sequence_fasta_is_canonical": is_canonical(seq),
            "sequence_fasta_length": len(seq) if seq else np.nan,
        })
    return pd.DataFrame(rows).drop_duplicates("dbaasp_id")


def harmonize(input_dir: Path, output_dir: Path, batch_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = {
        "peptides": input_dir / "peptides.csv",
        "target_activity": input_dir / "activity-against-target-species.csv",
        "toxicity": input_dir / "hemolytic-and-cytotoxic-activities.csv",
        "antibiofilm": input_dir / "peptides-antibiofilm-activities.csv",
        "fasta": input_dir / "peptides-fasta.txt",
    }
    missing = [str(p) for p in input_files.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing required input files:\n" + "\n".join(missing))

    fasta_df = parse_fasta(input_files["fasta"])

    pep = read_csv_clean(input_files["peptides"])
    pep = pep.rename(columns={
        "id": "dbaasp_id",
        "target_group": "target_group_raw",
        "target_object": "target_object_raw",
    })
    pep["dbaasp_id"] = pep["dbaasp_id"].astype(int)
    pep["sequence_metadata_raw"] = pep["sequence"].astype(str)
    pep["metadata_chains"] = pep["sequence"].apply(lambda x: "|".join(split_chains(x)))
    pep["metadata_chain_count"] = pep["sequence"].apply(lambda x: len(split_chains(x)))
    pep["sequence_metadata_joined"] = pep["sequence"].apply(lambda x: "".join(split_chains(x)))
    pep["sequence_metadata_is_canonical_joined"] = pep["sequence_metadata_joined"].apply(is_canonical)

    master = pep.merge(fasta_df, on="dbaasp_id", how="left")

    def choose_single_sequence(row: pd.Series) -> str:
        if isinstance(row.get("sequence_fasta"), str) and is_canonical(row["sequence_fasta"]):
            return row["sequence_fasta"]
        chains = str(row.get("metadata_chains", "")).split("|") if str(row.get("metadata_chains", "")) else []
        chains = [c for c in chains if is_canonical(c)]
        return chains[0] if len(chains) == 1 else ""

    master["sequence_clean"] = master.apply(choose_single_sequence, axis=1)
    master["sequence_length"] = master["sequence_clean"].apply(lambda s: len(s) if s else np.nan)
    master["is_multichain_metadata"] = (master["metadata_chain_count"] > 1) | master.get("complexity", "").astype(str).str.contains("multi", case=False, na=False)
    master["has_fasta_record"] = master["sequence_fasta"].notna()
    master["sequence_is_canonical"] = master["sequence_clean"].apply(is_canonical)
    master["trainable_single_chain"] = (
        master["sequence_is_canonical"]
        & (~master["is_multichain_metadata"])
        & master["sequence_length"].between(5, 64, inclusive="both")
    )

    act = read_csv_clean(input_files["target_activity"])
    act = act.rename(columns={
        "peptide_id": "dbaasp_id",
        "peptide_sequence": "sequence_reported",
        "target_species": "target_name",
        "activity": "activity_value",
        "ionic_strength_mm": "ionic_strength_mM",
    })
    act["dbaasp_id"] = act["dbaasp_id"].astype(int)
    act["sequence_reported_clean"] = act["sequence_reported"].apply(clean_sequence)
    act = act.merge(master[["dbaasp_id", "sequence_clean"]], on="dbaasp_id", how="left")
    act["sequence_clean"] = np.where(act["sequence_reported_clean"].apply(is_canonical), act["sequence_reported_clean"], act["sequence_clean"])
    act = pd.concat([act, act["activity_value"].apply(parse_numeric_value)], axis=1)
    act["unit_norm"] = act.get("unit", "").apply(unit_norm)
    act["measurement_class"] = act["activity_measure"].apply(measurement_class)
    act["measure_percent"] = act["activity_measure"].apply(parse_percent_from_text)
    act = pd.concat([act, act["target_name"].apply(parse_target_taxon)], axis=1)
    act["source_table"] = "activity_against_target_species"

    tox = read_csv_clean(input_files["toxicity"])
    tox = tox.rename(columns={
        "peptide_id": "dbaasp_id",
        "peptide_sequence": "sequence_reported",
        "activity_measure_for_lysis": "lysis_measure",
        "peptide_concentration": "concentration",
    })
    tox["dbaasp_id"] = tox["dbaasp_id"].astype(int)
    tox["sequence_reported_clean"] = tox["sequence_reported"].apply(clean_sequence)
    tox = tox.merge(master[["dbaasp_id", "sequence_clean"]], on="dbaasp_id", how="left")
    tox["sequence_clean"] = np.where(tox["sequence_reported_clean"].apply(is_canonical), tox["sequence_reported_clean"], tox["sequence_clean"])
    parsed_tox = tox["concentration"].apply(parse_numeric_value).rename(columns={
        "value_num": "concentration_num",
        "value_qualifier": "concentration_qualifier",
        "value_raw": "concentration_raw",
    })
    tox = pd.concat([tox, parsed_tox], axis=1)
    tox["unit_norm"] = tox.get("unit", "").apply(unit_norm)
    tox["toxicity_class"] = tox["lysis_measure"].apply(measurement_class)
    tox["lysis_percent"] = tox["lysis_measure"].apply(parse_percent_from_text)
    tox["source_table"] = "hemolytic_and_cytotoxic_activities"

    bio = read_csv_clean(input_files["antibiofilm"])
    bio = bio.rename(columns={
        "peptide_id": "dbaasp_id",
        "peptide_sequence": "sequence_reported",
        "target_species": "target_name",
        "activity": "activity_value",
    })
    bio["dbaasp_id"] = bio["dbaasp_id"].astype(int)
    bio["sequence_reported_clean"] = bio["sequence_reported"].apply(clean_sequence)
    bio = bio.merge(master[["dbaasp_id", "sequence_clean"]], on="dbaasp_id", how="left")
    bio["sequence_clean"] = np.where(bio["sequence_reported_clean"].apply(is_canonical), bio["sequence_reported_clean"], bio["sequence_clean"])
    bio = pd.concat([bio, bio["activity_value"].apply(parse_numeric_value)], axis=1)
    bio["unit_norm"] = bio.get("unit", "").apply(unit_norm)
    bio["measurement_class"] = bio["activity_measure"].apply(measurement_class)
    bio["measure_percent"] = bio["activity_measure"].apply(parse_percent_from_text)
    bio = pd.concat([bio, bio["target_name"].apply(parse_target_taxon)], axis=1)
    bio["source_table"] = "peptides_antibiofilm_activities"

    counts = pd.DataFrame({"dbaasp_id": master["dbaasp_id"]})
    for name, df in [("activity", act), ("toxicity", tox), ("antibiofilm", bio)]:
        c = df.groupby("dbaasp_id").size().rename(f"n_{name}_rows").reset_index()
        counts = counts.merge(c, on="dbaasp_id", how="left")

    mic = act[(act["measurement_class"] == "MIC") & act["value_num"].notna()].copy()
    best_mic = mic.groupby(["dbaasp_id", "unit_norm"])["value_num"].min().unstack().rename(columns={
        "uM": "best_mic_uM",
        "ug_per_ml": "best_mic_ug_per_ml",
        "nmol_per_g": "best_mic_nmol_per_g",
    }).reset_index()
    target_counts = act.groupby("dbaasp_id")["target_name"].nunique().rename("n_unique_activity_targets").reset_index()
    mic_target_counts = mic.groupby("dbaasp_id")["target_name"].nunique().rename("n_unique_mic_targets").reset_index()

    master_h = master.merge(counts, on="dbaasp_id", how="left")
    master_h = master_h.merge(best_mic, on="dbaasp_id", how="left")
    master_h = master_h.merge(target_counts, on="dbaasp_id", how="left")
    master_h = master_h.merge(mic_target_counts, on="dbaasp_id", how="left")
    for c in ["n_activity_rows", "n_toxicity_rows", "n_antibiofilm_rows", "n_unique_activity_targets", "n_unique_mic_targets"]:
        master_h[c] = master_h[c].fillna(0).astype(int)

    master_cols = [
        "dbaasp_id", "fasta_record_class", "name", "complexity", "synthesis_type",
        "n_terminus", "c_terminus", "sequence_clean", "sequence_length",
        "sequence_is_canonical", "trainable_single_chain", "is_multichain_metadata",
        "metadata_chain_count", "metadata_chains", "sequence_metadata_raw",
        "has_fasta_record", "fasta_header", "target_group_raw", "target_object_raw",
        "n_activity_rows", "n_unique_activity_targets", "n_unique_mic_targets",
        "best_mic_uM", "best_mic_ug_per_ml", "best_mic_nmol_per_g",
        "n_toxicity_rows", "n_antibiofilm_rows",
    ]
    master_out = master_h[[c for c in master_cols if c in master_h.columns]].sort_values("dbaasp_id")

    act_cols = [
        "dbaasp_id", "sequence_clean", "target_name", "target_genus", "target_binomial",
        "activity_measure", "measurement_class", "activity_value", "value_num",
        "value_qualifier", "unit", "unit_norm", "measure_percent",
        "pH", "ionic_strength_mM", "salt_type", "medium", "cfu", "note", "reference", "source_table",
    ]
    tox_cols = [
        "dbaasp_id", "sequence_clean", "target_cell", "lysis_measure", "toxicity_class",
        "lysis_percent", "concentration", "concentration_num", "concentration_qualifier",
        "unit", "unit_norm", "note", "reference", "source_table",
    ]
    bio_cols = [
        "dbaasp_id", "sequence_clean", "target_name", "target_genus", "target_binomial",
        "activity_measure", "measurement_class", "measure_percent", "activity_value",
        "value_num", "value_qualifier", "unit", "unit_norm", "medium", "cfu", "note", "reference", "source_table",
    ]

    act_out = act[[c for c in act_cols if c in act.columns]].sort_values(["dbaasp_id", "target_name"])
    tox_out = tox[[c for c in tox_cols if c in tox.columns]].sort_values(["dbaasp_id"])
    bio_out = bio[[c for c in bio_cols if c in bio.columns]].sort_values(["dbaasp_id", "target_name"])

    trainable = master_out[master_out["trainable_single_chain"]].drop_duplicates("sequence_clean").copy()
    trainable_corpus = trainable[[
        "dbaasp_id", "sequence_clean", "sequence_length", "name", "synthesis_type",
        "complexity", "target_group_raw", "n_activity_rows", "n_toxicity_rows", "n_antibiofilm_rows",
    ]].rename(columns={"sequence_clean": "sequence"})

    mic_labels = mic[[
        "dbaasp_id", "sequence_clean", "target_name", "target_genus", "target_binomial",
        "activity_measure", "value_num", "value_qualifier", "unit_norm", "medium", "reference",
    ]].rename(columns={"value_num": "mic_value", "unit_norm": "mic_unit"})
    mic_labels = mic_labels[mic_labels["sequence_clean"].apply(is_canonical)].copy()

    master_out.to_csv(output_dir / "dbaasp_peptides_master_harmonized.csv", index=False)
    act_out.to_csv(output_dir / "dbaasp_activity_target_species_long.csv", index=False)
    tox_out.to_csv(output_dir / "dbaasp_hemolysis_cytotoxicity_long.csv", index=False)
    bio_out.to_csv(output_dir / "dbaasp_antibiofilm_long.csv", index=False)
    trainable_corpus.to_csv(output_dir / "dbaasp_trainable_single_chain_sequences.csv", index=False)
    mic_labels.to_csv(output_dir / "dbaasp_mic_labels_long.csv", index=False)
    fasta_df.to_csv(output_dir / "dbaasp_fasta_id_sequence_map.csv", index=False)

    with open(output_dir / "dbaasp_trainable_single_chain_sequences.fasta", "w") as f:
        for _, r in trainable_corpus.iterrows():
            name = "" if pd.isna(r.get("name")) else str(r.get("name")).replace("\n", " ").strip()
            f.write(f">DBAASP_{int(r['dbaasp_id'])} {name}\n{r['sequence']}\n")

    summary = {
        "batch_label": batch_label,
        "input_files": {k: str(v) for k, v in input_files.items()},
        "row_counts": {
            "peptides_metadata_rows": int(len(pep)),
            "fasta_records": int(len(fasta_df)),
            "activity_against_target_species_rows": int(len(act)),
            "hemolytic_cytotoxic_rows": int(len(tox)),
            "antibiofilm_rows": int(len(bio)),
        },
        "harmonized_counts": {
            "master_peptides": int(len(master_out)),
            "with_fasta_record": int(master_out["has_fasta_record"].sum()),
            "missing_fasta_record": int((~master_out["has_fasta_record"]).sum()),
            "canonical_single_chain_trainable_unique_sequences": int(len(trainable_corpus)),
            "mic_label_rows_with_canonical_sequence": int(len(mic_labels)),
            "activity_rows_with_missing_sequence_after_join": int((act_out["sequence_clean"].fillna("") == "").sum()),
        },
        "notes": [
            "No MIC unit conversion was performed. uM, ug_per_ml, and nmol_per_g are kept separate.",
            "Multimer and multi-peptide records are preserved in the master table but excluded from the single-chain trainable sequence FASTA/CSV.",
            "Activity tables are long-format and joined to sequence_clean by DBAASP numeric ID where possible.",
            "value_num is the first parsed numeric value from the raw activity field; value_qualifier preserves <, <=, >, >= when present.",
        ],
    }
    (output_dir / "dbaasp_harmonization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[DONE] Wrote harmonized DBAASP batch to: {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--batch-label", default="dbaasp_manual_batch")
    args = p.parse_args()
    harmonize(args.input_dir, args.output_dir, args.batch_label)


if __name__ == "__main__":
    main()

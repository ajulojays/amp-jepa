#!/usr/bin/env python3
"""Utilities for identifying APEX organism-model columns and canonical species.

The APEX output mixes wide organism/strain MIC predictions with numeric ranking and
metadata fields. Detection therefore requires both a numeric MIC-like column and a
recognized organism name. This prevents score fields such as
``v3_rank_score_pre_apex`` from entering potency panels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

SUMMARY_PREFIXES = (
    "APEX_", "mean_", "median_", "min_", "max_", "organisms_",
    "fraction_", "n_pathogens_", "criteria_", "score_", "rank_",
    "v3_", "v4c_", "audit_", "self_", "broad_", "qc_", "layer1_",
)

FIXED_METADATA = {
    "candidate_id", "apex_candidate_id", "sequence", "sequence_clean",
    "generation", "generation_source", "lead_tier", "record_type",
    "parent_candidate_id", "second_parent_candidate_id", "generation_operator",
    "parent_selection_stratum", "hydrophobicity_zone", "ranking_note",
    "length", "sequence_length", "net_charge_KR_minus_DE",
    "hydrophobic_fraction", "proposal_index", "lineage_depth",
    "latent_sigma", "decode_temperature", "APEX_rank",
    "dual_novelty_pass", "median_MIC_le_32", "self_similarity_status",
    "v4c_pre_self_similarity_rank", "v4c_self_nonredundant_rank",
    "v3_rank_score_pre_apex",
}

# Canonical labels for abbreviated organism names in the frozen APEX matrix.
# Keys are (genus initial, species epithet as represented in the source column).
ABBREVIATED_SPECIES = {
    ("E", "coli"): "Escherichia coli",
    ("P", "aeruginosa"): "Pseudomonas aeruginosa",
    ("S", "aureus"): "Staphylococcus aureus",
    ("K", "pneumoniae"): "Klebsiella pneumoniae",
    ("A", "baumannii"): "Acinetobacter baumannii",
    ("A", "muciniphila"): "Akkermansia muciniphila",
    ("B", "fragilis"): "Bacteroides fragilis",
    ("B", "vulgatus"): "Bacteroides vulgatus",
    ("C", "aerofaciens"): "Collinsella aerofaciens",
    ("C", "scindens"): "Clostridium scindens",
    ("B", "thetaiotaomicron"): "Bacteroides thetaiotaomicron",
    ("B", "uniformis"): "Bacteroides uniformis",
    ("B", "eggerthi"): "Bacteroides eggerthii",
    ("C", "spiroforme"): "Clostridium spiroforme",
    ("P", "distasonis"): "Parabacteroides distasonis",
    ("P", "copri"): "Prevotella copri",
    ("B", "ovatus"): "Bacteroides ovatus",
    ("E", "rectale"): "Eubacterium rectale",
    ("C", "symbiosum"): "Clostridium symbiosum",
    ("R", "obeum"): "Ruminococcus obeum",
    ("R", "torques"): "Ruminococcus torques",
    ("E", "faecalis"): "Enterococcus faecalis",
    ("E", "faecium"): "Enterococcus faecium",
    ("L", "monocytogenes"): "Listeria monocytogenes",
}

FULL_BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z.-]{2,})\b")
ABBREVIATED_BINOMIAL = re.compile(r"\b([A-Z])\.\s*([a-z][a-z.-]{2,})\b")


def infer_species_label(column: str) -> str:
    """Return a canonical species label from an APEX model column name."""
    name = str(column).strip()

    full = FULL_BINOMIAL.search(name)
    if full:
        return f"{full.group(1)} {full.group(2)}"

    abbreviated = ABBREVIATED_BINOMIAL.search(name)
    if abbreviated:
        key = (abbreviated.group(1), abbreviated.group(2))
        return ABBREVIATED_SPECIES.get(key, "Unresolved")

    return "Unresolved"


def is_metadata_or_summary(column: str) -> bool:
    name = str(column)
    return name in FIXED_METADATA or name.startswith(SUMMARY_PREFIXES)


def detect_apex_model_columns(
    df: pd.DataFrame,
    minimum_numeric_fraction: float = 0.95,
) -> tuple[list[str], list[str]]:
    """Return recognized APEX MIC columns and excluded positive numeric columns."""
    model_columns: list[str] = []
    excluded_numeric: list[str] = []

    for column in df.columns:
        name = str(column)
        if is_metadata_or_summary(name):
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        fraction = float(numeric.notna().mean())
        if fraction < minimum_numeric_fraction:
            continue

        valid = numeric.dropna()
        if valid.empty or not valid.gt(0).all():
            continue

        if infer_species_label(name) == "Unresolved":
            excluded_numeric.append(name)
            continue

        model_columns.append(name)

    if not model_columns:
        raise ValueError("No recognized APEX organism/strain MIC columns were detected.")

    return model_columns, excluded_numeric


def match_species_columns(
    df: pd.DataFrame,
    species_regex: str,
    minimum_numeric_fraction: float = 0.95,
) -> tuple[list[str], list[str]]:
    """Match a regex against raw model names or their canonical species labels."""
    pattern = re.compile(species_regex, flags=re.IGNORECASE)
    model_columns, excluded_numeric = detect_apex_model_columns(
        df,
        minimum_numeric_fraction=minimum_numeric_fraction,
    )
    matched = [
        column
        for column in model_columns
        if pattern.search(column) or pattern.search(infer_species_label(column))
    ]
    return matched, excluded_numeric


def build_species_inventory(model_columns: Iterable[str]) -> pd.DataFrame:
    rows = [
        {"inferred_species": infer_species_label(column), "model_column": column}
        for column in model_columns
    ]
    return (
        pd.DataFrame(rows)
        .groupby("inferred_species", dropna=False)
        .agg(
            n_models=("model_column", "count"),
            model_columns=("model_column", lambda values: " | ".join(map(str, values))),
        )
        .reset_index()
        .sort_values("inferred_species")
        .reset_index(drop=True)
    )

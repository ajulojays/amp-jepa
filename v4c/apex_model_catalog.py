#!/usr/bin/env python3
"""Exact catalog for organism/strain columns in the frozen V4C APEX matrix.

The APEX output contains 34 organism/strain MIC variables plus numeric metadata and
ranking variables. Species analyses must use only the organism/strain variables
actually present in that matrix. This module therefore uses an explicit catalog for
the observed columns, while retaining conservative abbreviation handling for minor
spacing/capitalization differences.
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

# Exact source-column prefixes and their canonical species labels. Rules are ordered
# from more specific to more general and cover all 34 frozen APEX MIC variables.
MODEL_SPECIES_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^Salmonella\s+enterica\b", re.I), "Salmonella enterica"),
    (re.compile(r"^E\.\s*coli\b", re.I), "Escherichia coli"),
    (re.compile(r"^P\.\s*aeruginosa\b", re.I), "Pseudomonas aeruginosa"),
    (re.compile(r"^S\.\s*aureus\b", re.I), "Staphylococcus aureus"),
    (re.compile(r"^K\.\s*pneumoniae\b", re.I), "Klebsiella pneumoniae"),
    (re.compile(r"^A\.\s*baumannii\b", re.I), "Acinetobacter baumannii"),
    (re.compile(r"^A\.\s*muciniphila\b", re.I), "Akkermansia muciniphila"),
    (re.compile(r"^B\.\s*fragilis\b", re.I), "Bacteroides fragilis"),
    (re.compile(r"^B\.\s*vulgatus\b", re.I), "Bacteroides vulgatus"),
    (re.compile(r"^C\.\s*aerofaciens\b", re.I), "Collinsella aerofaciens"),
    (re.compile(r"^C\.\s*scindens\b", re.I), "Clostridium scindens"),
    (re.compile(r"^B\.\s*thetaiotaomicron\b", re.I), "Bacteroides thetaiotaomicron"),
    (re.compile(r"^B\.\s*uniformis\b", re.I), "Bacteroides uniformis"),
    (re.compile(r"^B\.\s*eggerthi\b", re.I), "Bacteroides eggerthii"),
    (re.compile(r"^C\.\s*spiroforme\b", re.I), "Clostridium spiroforme"),
    (re.compile(r"^P\.\s*distasonis\b", re.I), "Parabacteroides distasonis"),
    (re.compile(r"^P\.\s*copri\b", re.I), "Prevotella copri"),
    (re.compile(r"^B\.\s*ovatus\b", re.I), "Bacteroides ovatus"),
    (re.compile(r"^E\.\s*rectale\b", re.I), "Eubacterium rectale"),
    (re.compile(r"^C\.\s*symbiosum\b", re.I), "Clostridium symbiosum"),
    (re.compile(r"^R\.\s*obeum\b", re.I), "Ruminococcus obeum"),
    (re.compile(r"^R\.\s*torques\b", re.I), "Ruminococcus torques"),
    (re.compile(r"^(?:vancomycin-resistant\s+)?E\.\s*faecalis\b", re.I), "Enterococcus faecalis"),
    (re.compile(r"^(?:vancomycin-resistant\s+)?E\.\s*faecium\b", re.I), "Enterococcus faecium"),
    (re.compile(r"^L\.\s*monocytogenes\b", re.I), "Listeria monocytogenes"),
)

# Canonical species aliases accepted by the species-specific command. This lets users
# request either a full species name or the abbreviation used in the source matrix.
SPECIES_ALIASES: dict[str, str] = {
    "e. coli": "Escherichia coli",
    "p. aeruginosa": "Pseudomonas aeruginosa",
    "s. aureus": "Staphylococcus aureus",
    "k. pneumoniae": "Klebsiella pneumoniae",
    "a. baumannii": "Acinetobacter baumannii",
    "a. muciniphila": "Akkermansia muciniphila",
    "b. fragilis": "Bacteroides fragilis",
    "b. vulgatus": "Bacteroides vulgatus",
    "c. aerofaciens": "Collinsella aerofaciens",
    "c. scindens": "Clostridium scindens",
    "b. thetaiotaomicron": "Bacteroides thetaiotaomicron",
    "b. uniformis": "Bacteroides uniformis",
    "b. eggerthi": "Bacteroides eggerthii",
    "c. spiroforme": "Clostridium spiroforme",
    "p. distasonis": "Parabacteroides distasonis",
    "p. copri": "Prevotella copri",
    "b. ovatus": "Bacteroides ovatus",
    "e. rectale": "Eubacterium rectale",
    "c. symbiosum": "Clostridium symbiosum",
    "r. obeum": "Ruminococcus obeum",
    "r. torques": "Ruminococcus torques",
    "e. faecalis": "Enterococcus faecalis",
    "e. faecium": "Enterococcus faecium",
    "l. monocytogenes": "Listeria monocytogenes",
}


def infer_species_label(column: str) -> str:
    """Return the canonical species for one frozen APEX model column."""
    name = str(column).strip()
    for pattern, species in MODEL_SPECIES_RULES:
        if pattern.search(name):
            return species
    return "Unresolved"


def normalize_species_query(value: str) -> str:
    """Normalize a canonical or abbreviated species query for exact matching."""
    stripped = re.sub(r"\s+", " ", str(value).strip())
    return SPECIES_ALIASES.get(stripped.lower(), stripped)


def is_metadata_or_summary(column: str) -> bool:
    name = str(column)
    return name in FIXED_METADATA or name.startswith(SUMMARY_PREFIXES)


def detect_apex_model_columns(
    df: pd.DataFrame,
    minimum_numeric_fraction: float = 0.95,
) -> tuple[list[str], list[str]]:
    """Return exact recognized APEX MIC variables and excluded numeric variables."""
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
    species_query: str,
    minimum_numeric_fraction: float = 0.95,
) -> tuple[list[str], list[str]]:
    """Match a species query against exact canonical labels and raw model names."""
    requested = normalize_species_query(species_query)
    pattern = re.compile(species_query, flags=re.IGNORECASE)
    model_columns, excluded_numeric = detect_apex_model_columns(
        df,
        minimum_numeric_fraction=minimum_numeric_fraction,
    )

    matched = [
        column
        for column in model_columns
        if infer_species_label(column).casefold() == requested.casefold()
        or pattern.search(column)
        or pattern.search(infer_species_label(column))
    ]
    return matched, excluded_numeric


def build_species_inventory(model_columns: Iterable[str]) -> pd.DataFrame:
    rows = [
        {"inferred_species": infer_species_label(column), "model_column": column}
        for column in model_columns
    ]
    inventory = (
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
    return inventory


def available_species(df: pd.DataFrame, minimum_numeric_fraction: float = 0.95) -> list[str]:
    model_columns, _ = detect_apex_model_columns(
        df,
        minimum_numeric_fraction=minimum_numeric_fraction,
    )
    return sorted({infer_species_label(column) for column in model_columns})

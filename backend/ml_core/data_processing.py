from __future__ import annotations

from pathlib import Path

import pandas as pd

from .state import MODEL_STATE


def normalize_symptom_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def resolve_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "Training_2.csv"


def load_and_clean_data() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(resolve_dataset_path())

    df.columns = df.columns.astype(str).str.strip()
    df = df.drop_duplicates().dropna(axis=1, how="all").copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    prognosis_candidates = [
        c for c in df.columns if c.lower() == "prognosis" or "prognosis" in c.lower()
    ]
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if prognosis_candidates:
        target_column = prognosis_candidates[0]
    elif object_cols:
        target_column = object_cols[0]
    else:
        target_column = df.columns[-1]

    return df, target_column


def ensure_feature_metadata() -> None:
    if MODEL_STATE["feature_names"]:
        return

    df, target_column = load_and_clean_data()
    feature_names = df.drop(columns=[target_column]).columns.tolist()
    symptom_lookup = {normalize_symptom_name(symptom): symptom for symptom in feature_names}

    MODEL_STATE.update(
        {
            "feature_names": feature_names,
            "symptom_lookup": symptom_lookup,
        }
    )

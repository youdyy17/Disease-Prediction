from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data_processing import ensure_feature_metadata, normalize_symptom_name
from .model_training import initialize_model
from .state import MODEL_STATE


def get_symptoms() -> list[str]:
    ensure_feature_metadata()
    return MODEL_STATE["feature_names"]


def get_model_summary() -> dict[str, Any]:
    initialize_model()
    return {
        "selected_model": MODEL_STATE["model_name"],
        "selected_model_accuracy": round(MODEL_STATE["model_accuracy"] * 100, 2),
        "all_model_scores": {
            name: round(score * 100, 2)
            for name, score in MODEL_STATE["model_scores"].items()
        },
        "symptom_count": len(MODEL_STATE["feature_names"]),
    }


def predict_top_diseases(selected_symptoms: list[str], top_n: int = 5) -> dict[str, Any]:
    initialize_model()

    model = MODEL_STATE["model"]
    feature_names = MODEL_STATE["feature_names"]
    symptom_lookup = MODEL_STATE["symptom_lookup"]
    label_encoder: LabelEncoder = MODEL_STATE["label_encoder"]

    input_row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    selected: list[str] = []
    unknown: list[str] = []

    for symptom in selected_symptoms:
        normalized = normalize_symptom_name(symptom)
        actual = symptom_lookup.get(normalized)
        if actual is None:
            unknown.append(symptom)
            continue
        input_row.at[0, actual] = 1
        selected.append(actual)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_row)[0]
    else:
    # Softmax Function
        decision = model.decision_function(input_row)
        decision = np.array(decision).ravel()
        exp_scores = np.exp(decision - np.max(decision))
        probabilities = exp_scores / exp_scores.sum()

    top_indices = np.argsort(probabilities)[::-1][:top_n]

    predictions = [
        {
            "disease": str(label_encoder.inverse_transform([idx])[0]),
            "probability": round(float(probabilities[idx]) * 100, 2),
        }
        for idx in top_indices
    ]
    
    print({
        "selected_symptoms": selected,
        "unknown_symptoms": unknown,
        "predictions": predictions,
        "model": MODEL_STATE["model_name"],
        "accuracy": round(MODEL_STATE["model_accuracy"] * 100, 2),
        "total_symptoms_selected": len(selected),
    })

    return {
        "selected_symptoms": selected,
        "unknown_symptoms": unknown,
        "predictions": predictions,
        "model": MODEL_STATE["model_name"],
        "accuracy": round(MODEL_STATE["model_accuracy"] * 100, 2),
        "total_symptoms_selected": len(selected),
    }

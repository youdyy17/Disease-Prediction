from __future__ import annotations

from typing import Any

MODEL_STATE: dict[str, Any] = {
    "trained": False,
    "model": None,
    "model_name": None,
    "model_accuracy": 0.0,
    "model_scores": {},
    "feature_names": [],
    "symptom_lookup": {},
    "label_encoder": None,
}

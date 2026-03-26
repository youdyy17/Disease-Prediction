from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


_MODEL_STATE: dict[str, Any] = {
	"trained": False,
	"model": None,
	"model_name": None,
	"model_accuracy": 0.0,
	"model_scores": {},
	"feature_names": [],
	"symptom_lookup": {},
	"label_encoder": None,
}


def _normalize_symptom_name(value: str) -> str:
	return str(value).strip().lower().replace(" ", "_")


def _resolve_dataset_path() -> Path:
	return Path(__file__).resolve().parent / "data" / "Training_2.csv"


def _load_and_clean_data() -> tuple[pd.DataFrame, str]:
	df = pd.read_csv(_resolve_dataset_path())

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


def _ensure_feature_metadata() -> None:
	if _MODEL_STATE["feature_names"]:
		return

	df, target_column = _load_and_clean_data()
	feature_names = df.drop(columns=[target_column]).columns.tolist()
	symptom_lookup = {_normalize_symptom_name(symptom): symptom for symptom in feature_names}

	_MODEL_STATE.update(
		{
			"feature_names": feature_names,
			"symptom_lookup": symptom_lookup,
		}
	)


def initialize_model() -> None:
	if _MODEL_STATE["trained"]:
		return

	_ensure_feature_metadata()

	df, target_column = _load_and_clean_data()
	X = df.drop(columns=[target_column]).copy()
	y = df[target_column].copy()

	for col in X.select_dtypes(include=["object"]).columns:
		col_encoder = LabelEncoder()
		X[col] = col_encoder.fit_transform(X[col].astype(str))

	X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y)

	class_counts = pd.Series(y_encoded).value_counts()
	split_stratify = y_encoded if class_counts.min() >= 2 else None

	sample_size = min(4000, len(X))
	if sample_size < len(X):
		sample_stratify = y_encoded if class_counts.min() >= 2 else None
		X_run, _, y_run, _ = train_test_split(
			X,
			y_encoded,
			train_size=sample_size,
			shuffle=True,
			stratify=sample_stratify,
			random_state=42,
		)
	else:
		X_run, y_run = X, y_encoded

	class_counts_run = pd.Series(y_run).value_counts()
	split_stratify_run = y_run if class_counts_run.min() >= 2 else None

	X_train, X_test, y_train, y_test = train_test_split(
		X_run,
		y_run,
		test_size=0.2,
		shuffle=True,
		stratify=split_stratify_run,
		random_state=42,
	)

	candidates = {
		"Random Forest": RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1),
		"Naive Bayes": GaussianNB(),
		"SVM": SVC(kernel="linear", probability=False, random_state=42),
	}

	model_scores: dict[str, float] = {}
	for name, model in candidates.items():
		model.fit(X_train, y_train)
		preds = model.predict(X_test)
		model_scores[name] = float(accuracy_score(y_test, preds))

	best_model_name = max(model_scores, key=model_scores.get)
	best_model = candidates[best_model_name]
	best_model.fit(X_run, y_run)

	_MODEL_STATE.update(
		{
			"trained": True,
			"model": best_model,
			"model_name": best_model_name,
			"model_accuracy": model_scores[best_model_name],
			"model_scores": model_scores,
			"label_encoder": label_encoder,
		}
	)


def get_symptoms() -> list[str]:
	_ensure_feature_metadata()
	return _MODEL_STATE["feature_names"]


def get_model_summary() -> dict[str, Any]:
	initialize_model()
	return {
		"selected_model": _MODEL_STATE["model_name"],
		"selected_model_accuracy": round(_MODEL_STATE["model_accuracy"] * 100, 2),
		"all_model_scores": {
			name: round(score * 100, 2)
			for name, score in _MODEL_STATE["model_scores"].items()
		},
		"symptom_count": len(_MODEL_STATE["feature_names"]),
	}


def predict_top_diseases(selected_symptoms: list[str], top_n: int = 5) -> dict[str, Any]:
	initialize_model()

	model = _MODEL_STATE["model"]
	feature_names = _MODEL_STATE["feature_names"]
	symptom_lookup = _MODEL_STATE["symptom_lookup"]
	label_encoder: LabelEncoder = _MODEL_STATE["label_encoder"]

	input_row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

	selected: list[str] = []
	unknown: list[str] = []

	for symptom in selected_symptoms:
		normalized = _normalize_symptom_name(symptom)
		actual = symptom_lookup.get(normalized)
		if actual is None:
			unknown.append(symptom)
			continue
		input_row.at[0, actual] = 1
		selected.append(actual)

	if hasattr(model, "predict_proba"):
		probabilities = model.predict_proba(input_row)[0]
	else:
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

	return {
		"selected_symptoms": selected,
		"unknown_symptoms": unknown,
		"predictions": predictions,
		"model": _MODEL_STATE["model_name"],
		"accuracy": round(_MODEL_STATE["model_accuracy"] * 100, 2),
		"total_symptoms_selected": len(selected),
	}

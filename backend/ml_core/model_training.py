from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .data_processing import ensure_feature_metadata, load_and_clean_data
from .state import MODEL_STATE


def initialize_model() -> None:
    if MODEL_STATE["trained"]:
        return

    ensure_feature_metadata()

    df, target_column = load_and_clean_data()
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
        X_run, _, y_run, _ = train_test_split(
            X,
            y_encoded,
            train_size=sample_size,
            shuffle=True,
            stratify=split_stratify,
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

    MODEL_STATE.update(
        {
            "trained": True,
            "model": best_model,
            "model_name": best_model_name,
            "model_accuracy": model_scores[best_model_name],
            "model_scores": model_scores,
            "label_encoder": label_encoder,
        }
    )

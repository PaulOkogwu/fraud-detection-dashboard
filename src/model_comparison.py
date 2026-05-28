"""
Model comparison utilities for FraudLens.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import logger


def _optional_xgboost_model(random_state: int) -> tuple[Any, str | None]:
    """
    Returns an XGBoost model when available; otherwise returns a skip reason.
    """
    try:
        from xgboost import XGBClassifier  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"XGBoost unavailable: {exc}"

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=random_state,
        eval_metric="logloss",
    )
    return model, None


def _build_model_registry(random_state: int) -> list[dict[str, Any]]:
    xgb_model, xgb_skip_reason = _optional_xgboost_model(random_state)
    registry = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state
            ),
            "skip_reason": None,
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "skip_reason": None,
        },
        {
            "name": "XGBoost",
            "model": xgb_model,
            "skip_reason": xgb_skip_reason,
        },
    ]
    return registry


def compare_models(
    data: pd.DataFrame,
    target_col: str = "isFraud",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Train/evaluate multiple models on the same split and return metrics + details.
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    result_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "target_column": target_col,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "models": {},
    }

    for item in _build_model_registry(random_state):
        model_name = item["name"]
        model = item["model"]
        skip_reason = item["skip_reason"]

        if skip_reason is not None or model is None:
            logger.warning("%s skipped: %s", model_name, skip_reason)
            result_rows.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "reason": skip_reason,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "roc_auc": np.nan,
                }
            )
            details["models"][model_name] = {
                "status": "skipped",
                "reason": skip_reason,
            }
            continue

        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                probs = preds

            cm = confusion_matrix(y_test, preds, labels=[0, 1])
            row = {
                "model": model_name,
                "status": "trained",
                "reason": "",
                "accuracy": float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1_score": float(f1_score(y_test, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probs)),
            }
            result_rows.append(row)
            details["models"][model_name] = {
                "status": "trained",
                "confusion_matrix": cm.tolist(),
                "metrics": row,
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Model comparison failed for %s: %s", model_name, exc)
            result_rows.append(
                {
                    "model": model_name,
                    "status": "error",
                    "reason": str(exc),
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "roc_auc": np.nan,
                }
            )
            details["models"][model_name] = {
                "status": "error",
                "reason": str(exc),
            }

    results_df = pd.DataFrame(result_rows)
    return results_df, details


def save_comparison_results(
    results_df: pd.DataFrame,
    details: dict[str, Any],
    output_dir: str = "reports",
    prefix: str = "model_comparison",
) -> dict[str, str]:
    """
    Saves model comparison outputs as CSV + JSON for reuse.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    details_path = os.path.join(output_dir, f"{prefix}_details.json")

    results_df.to_csv(csv_path, index=False)
    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    return {"summary_csv": csv_path, "details_json": details_path}

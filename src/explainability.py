"""
Explainability helpers for FraudLens dashboard.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def align_features_for_model(features: pd.DataFrame, model_feature_names: list[str] | None) -> pd.DataFrame:
    """
    Align feature columns to match the model's training schema.
    """
    if not model_feature_names:
        return features
    return features.reindex(columns=model_feature_names, fill_value=0)


def get_tree_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return sorted feature importance for tree-based estimators.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Loaded model does not expose tree-based feature importances.")

    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.shape[0] != len(feature_names):
        raise ValueError("Feature importance length does not match feature name count.")

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def compute_optional_shap_importance(
    model: Any,
    features: pd.DataFrame,
    max_samples: int = 200,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Compute optional SHAP mean absolute contribution by feature.
    Returns (result_df, error_message). If SHAP is unavailable, result_df is None.
    """
    try:
        import shap  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"SHAP unavailable: {exc}"

    if features.empty:
        return None, "No feature rows available for SHAP computation."

    sampled = features.sample(min(max_samples, len(features)), random_state=42)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sampled)

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])
        else:
            values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[:, :, -1]

        mean_abs = np.abs(values).mean(axis=0)
        result = pd.DataFrame(
            {"feature": sampled.columns.tolist(), "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        return result.reset_index(drop=True), None
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"SHAP computation failed: {exc}"

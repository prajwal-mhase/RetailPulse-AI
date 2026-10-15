
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_rows: int = 200,
) -> Optional[tuple[np.ndarray, list[str]]]:
    """
    Compute SHAP values for XGBoost model.
    Returns (shap_values_array, feature_names) or None if SHAP not available.
    Limits rows to `max_rows` for performance.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping explainability.")
        return None

    sample = X.head(max_rows).fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    return shap_vals, list(sample.columns)


def shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Aggregate SHAP values into mean absolute importance per feature.
    Returns DataFrame sorted by importance descending.
    """
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    return df.reset_index(drop=True)


def get_subcat_shap(
    model,
    feature_df: pd.DataFrame,
    feature_names: list[str],
    subcat: str,
    key: str = "Sub-Category",
    n_rows: int = 52,
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP importance for a specific sub-category's most recent rows.
    Returns importance DataFrame or None.
    """
    subset = feature_df[feature_df[key] == subcat].tail(n_rows)
    avail = [c for c in feature_names if c in subset.columns]
    X = subset[avail].fillna(0)
    if len(X) == 0:
        return None
    result = compute_shap_values(model, X)
    if result is None:
        return None
    shap_vals, feats = result
    return shap_feature_importance(shap_vals, feats)


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical coverage of prediction intervals.
    Returns the fraction of actuals that fall within [lower, upper].
    """
    covered = np.sum((y_true >= lower) & (y_true <= upper))
    return float(covered / len(y_true)) if len(y_true) > 0 else 0.0

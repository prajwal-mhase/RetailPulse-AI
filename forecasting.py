
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import (
    AGGS_KEY,
    ARTIFACTS_DIR,
    FORECAST_HORIZONS,
    RANDOM_STATE,
    ROLLING_WINDOWS,
    XGBOOST_PARAMS,
    LAG_WEEKS,
)

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = (
    ["subcat_id", "year", "month", "week_of_year", "quarter", "is_q4"]
    + [f"lag_{l}w" for l in LAG_WEEKS]
    + [f"roll_mean_{w}w" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}w" for w in ROLLING_WINDOWS]
)
TARGET_COL: str = "quantity"


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, WAPE, and forecast bias."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "Bias": bias}


# ── Baseline models ───────────────────────────────────────────────────────

def naive_forecast(train: pd.DataFrame, test: pd.DataFrame, key: str = AGGS_KEY) -> np.ndarray:
    """Predict the last observed value per sub-category."""
    last_vals = (
        train.sort_values("week").groupby(key)[TARGET_COL].last().to_dict()
    )
    return test[key].map(last_vals).fillna(0).values


def seasonal_naive_forecast(
    train: pd.DataFrame, test: pd.DataFrame, key: str = AGGS_KEY
) -> np.ndarray:
    """Predict the same week one year ago (52 weeks lag)."""
    ref = (
        train.sort_values("week")
        .groupby([key, "week_of_year"])[TARGET_COL]
        .mean()
        .to_dict()
    )
    preds = []
    for _, row in test.iterrows():
        k = (row[key], int(row["week_of_year"]))
        preds.append(ref.get(k, 0.0))
    return np.array(preds)


def moving_average_forecast(
    train: pd.DataFrame, test: pd.DataFrame, window: int = 4, key: str = AGGS_KEY
) -> np.ndarray:
    """Predict the rolling-window mean of recent training data."""
    ma_vals = (
        train.sort_values("week")
        .groupby(key)[TARGET_COL]
        .apply(lambda x: x.iloc[-window:].mean())
        .to_dict()
    )
    return test[key].map(ma_vals).fillna(0).values


# ── ML helpers ────────────────────────────────────────────────────────────

def _get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) dropping rows with NaN in any feature column."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset = df[available + [TARGET_COL]].dropna()
    return subset[available], subset[TARGET_COL]


# ── ML model training ─────────────────────────────────────────────────────

def train_ridge(
    train: pd.DataFrame,
) -> tuple[Ridge, StandardScaler, list[str]]:
    X_tr, y_tr = _get_xy(train)
    feature_names = list(X_tr.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    model = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    model.fit(X_scaled, y_tr)
    logger.info("Ridge trained on %d samples, %d features", len(y_tr), len(feature_names))
    return model, scaler, feature_names


def train_random_forest(
    train: pd.DataFrame,
) -> tuple[RandomForestRegressor, list[str]]:
    X_tr, y_tr = _get_xy(train)
    feature_names = list(X_tr.columns)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    logger.info("RandomForest trained on %d samples", len(y_tr))
    return model, feature_names


def train_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    use_early_stopping: bool = True,
) -> tuple[xgb.XGBRegressor, list[str]]:
    X_tr, y_tr = _get_xy(train)
    X_val, y_val = _get_xy(val)
    feature_names = list(X_tr.columns)

    if use_early_stopping and len(X_val) > 0:
        model = xgb.XGBRegressor(
            **XGBOOST_PARAMS,
            early_stopping_rounds=30,
            eval_metric="mae",
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best = int(getattr(model, "best_iteration", 0) or 0)
        # If early stopping fired at step 0 or 1, it means the dataset has very
        # low autocorrelation; fall back to a fixed reasonable n_estimators.
        if best <= 1:
            logger.warning(
                "XGBoost early stopping fired at iteration %d; "
                "refitting with fixed n_estimators=200.", best
            )
            model = xgb.XGBRegressor(**{**XGBOOST_PARAMS, "n_estimators": 200})
            model.fit(X_tr, y_tr, verbose=False)
    else:
        model = xgb.XGBRegressor(**{**XGBOOST_PARAMS, "n_estimators": 200})
        model.fit(X_tr, y_tr, verbose=False)

    logger.info(
        "XGBoost trained on %d samples; n_estimators=%s",
        len(y_tr),
        getattr(model, "n_estimators", "N/A"),
    )
    return model, feature_names



# ── Prediction helpers ────────────────────────────────────────────────────

def predict_ml(
    model: Any,
    df: pd.DataFrame,
    feature_names: list[str],
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Generate predictions for any row in df that has complete features."""
    avail = [c for c in feature_names if c in df.columns]
    X = df[avail].fillna(0)
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    return np.clip(preds, 0, None)  # demand cannot be negative


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_validation_selection(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Evaluate all candidate models strictly on the validation split (trained on train only).
    Used to justify model selection prior to viewing test set data.
    """
    results: dict[str, dict] = {}
    y_val = val[TARGET_COL].values

    # Baselines
    results["Naive"] = compute_metrics(y_val, naive_forecast(train, val))
    results["Seasonal Naive"] = compute_metrics(y_val, seasonal_naive_forecast(train, val))
    results["Moving Average (4w)"] = compute_metrics(y_val, moving_average_forecast(train, val, window=4))

    # Ridge
    ridge, scaler, ridge_feats = train_ridge(train)
    y_ridge = predict_ml(ridge, val, ridge_feats, scaler=scaler)
    results["Ridge"] = compute_metrics(y_val, y_ridge)

    # Random Forest
    rf, rf_feats = train_random_forest(train)
    y_rf = predict_ml(rf, val, rf_feats)
    results["Random Forest"] = compute_metrics(y_val, y_rf)

    # XGBoost
    xgb_model, xgb_feats = train_xgboost(train, val, use_early_stopping=True)
    y_xgb = predict_ml(xgb_model, val, xgb_feats)
    results["XGBoost"] = compute_metrics(y_val, y_xgb)

    return results


def evaluate_all_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Train all models on train+val (all data prior to test split), evaluate on untouched test split.
    Returns dict of model_name → test metrics dict.
    """
    results: dict[str, dict] = {}
    combined = pd.concat([train, val])
    y_test = test[TARGET_COL].values

    # ── Baselines (using all history up to test period)
    y_naive = naive_forecast(combined, test)
    results["Naive"] = compute_metrics(y_test, y_naive)

    y_snaive = seasonal_naive_forecast(combined, test)
    results["Seasonal Naive"] = compute_metrics(y_test, y_snaive)

    y_ma = moving_average_forecast(combined, test, window=4)
    results["Moving Average (4w)"] = compute_metrics(y_test, y_ma)

    # ── Ridge
    ridge, scaler, ridge_feats = train_ridge(combined)
    y_ridge = predict_ml(ridge, test, ridge_feats, scaler=scaler)
    results["Ridge"] = compute_metrics(y_test, y_ridge)

    # ── Random Forest
    rf, rf_feats = train_random_forest(combined)
    y_rf = predict_ml(rf, test, rf_feats)
    results["Random Forest"] = compute_metrics(y_test, y_rf)

    # ── XGBoost (using validation-selected best iteration, refit on train+val)
    xgb_val, _ = train_xgboost(train, val, use_early_stopping=True)
    best_iter = max(int(getattr(xgb_val, "best_iteration", 200) or 200), 50)
    final_params = {**XGBOOST_PARAMS, "n_estimators": best_iter}
    X_all, y_all = _get_xy(combined)
    final_xgb = xgb.XGBRegressor(**final_params)
    final_xgb.fit(X_all, y_all, verbose=False)
    y_xgb = predict_ml(final_xgb, test, list(X_all.columns))
    results["XGBoost"] = compute_metrics(y_test, y_xgb)

    # Log summary
    for name, m in results.items():
        logger.info(
            "%-25s MAE=%.1f  RMSE=%.1f  WAPE=%.3f  Bias=%.1f",
            name, m["MAE"], m["RMSE"], m["WAPE"], m["Bias"],
        )
    return results


# ── Forecast for future horizons ──────────────────────────────────────────

def forecast_horizon(
    model: xgb.XGBRegressor,
    feature_names: list[str],
    grid: pd.DataFrame,
    cat_map: dict,
    subcat: str,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Recursive multi-step forecast for a single sub-category.
    Each step predicts the next week; the prediction is fed back as lag_1w.
    Returns DataFrame with columns: week, forecast, lower_80, upper_80.

    Uncertainty intervals are derived from bootstrap residuals computed on
    the most recent 26 training weeks (historical residuals, not parametric).
    """
    horizon_weeks = max(1, int(np.ceil(horizon_days / 7)))
    subcat_data = (
        grid[grid[AGGS_KEY] == subcat]
        .sort_values("week")
        .copy()
    )

    # Build residual-based uncertainty from the most recent 26 weeks
    history = subcat_data.tail(52).copy()
    history = _enrich_features(history, cat_map)
    avail = [c for c in feature_names if c in history.columns]
    X_hist = history[avail].fillna(0)
    y_hist = history[TARGET_COL].values
    if len(X_hist) > 0:
        y_hist_pred = np.clip(model.predict(X_hist), 0, None)
        residuals = y_hist - y_hist_pred
        resid_std = np.std(residuals) if len(residuals) > 1 else 0.0
    else:
        resid_std = 0.0

    # Iterative forecasting
    last_row = subcat_data.iloc[-1].copy()
    last_week = subcat_data["week"].iloc[-1]
    history_vals = subcat_data[TARGET_COL].values.tolist()

    forecasts = []
    current_lags = {f"lag_{l}w": _get_lag(history_vals, l) for l in LAG_WEEKS}
    rolling_series = history_vals[:]

    for step in range(1, horizon_weeks + 1):
        next_week = last_week + pd.Timedelta(weeks=step)
        row = _build_forecast_row(
            next_week=next_week,
            subcat=subcat,
            cat_map=cat_map,
            current_lags=current_lags,
            rolling_series=rolling_series,
        )
        feat_row = pd.DataFrame([row])
        avail = [c for c in feature_names if c in feat_row.columns]
        X_row = feat_row[avail].fillna(0)
        pred = float(np.clip(model.predict(X_row)[0], 0, None))

        # CI based on accumulated horizon uncertainty (grows with horizon)
        z80 = 1.28
        interval = z80 * resid_std * np.sqrt(step)
        forecasts.append(
            {
                "week": next_week,
                "forecast": round(pred, 2),
                "lower_80": max(0.0, round(pred - interval, 2)),
                "upper_80": round(pred + interval, 2),
            }
        )
        # Feed prediction back
        rolling_series.append(pred)
        for l in LAG_WEEKS:
            current_lags[f"lag_{l}w"] = _get_lag(rolling_series, l)

    return pd.DataFrame(forecasts)


def _get_lag(series: list, lag: int) -> float:
    """Return the value lag steps back in a list, or 0 if not available."""
    idx = len(series) - lag
    return float(series[idx]) if idx >= 0 else 0.0


def _enrich_features(df: pd.DataFrame, cat_map: dict) -> pd.DataFrame:
    """Add calendar and encoded-category columns to an existing aggregated df."""
    from data_pipeline import (
        add_calendar_features,
        add_lag_features,
        add_rolling_features,
    )
    df = add_calendar_features(df)
    df = add_lag_features(df, target=TARGET_COL)
    df = add_rolling_features(df, target=TARGET_COL)
    df["subcat_id"] = df[AGGS_KEY].map(cat_map).fillna(0).astype(int)
    return df


def _build_forecast_row(
    next_week: pd.Timestamp,
    subcat: str,
    cat_map: dict,
    current_lags: dict,
    rolling_series: list,
) -> dict:
    """Build a feature dictionary for one future week."""
    from config import ROLLING_WINDOWS as RW
    row = {
        "subcat_id": cat_map.get(subcat, 0),
        "year": next_week.year,
        "month": next_week.month,
        "week_of_year": int(next_week.isocalendar()[1]),
        "quarter": next_week.quarter,
        "is_q4": int(next_week.quarter == 4),
    }
    row.update(current_lags)
    for w in RW:
        tail = rolling_series[-w:] if len(rolling_series) >= w else rolling_series
        row[f"roll_mean_{w}w"] = float(np.mean(tail)) if tail else 0.0
        row[f"roll_std_{w}w"] = float(np.std(tail)) if len(tail) > 1 else 0.0
    return row


# ── Artefact persistence ──────────────────────────────────────────────────

def save_artifacts(
    xgb_model: xgb.XGBRegressor,
    feature_names: list[str],
    cat_map: dict,
    metrics: dict,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Persist trained model, feature names, category map, and metrics."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    import json

    with open(artifacts_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(artifacts_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(artifacts_dir / "cat_map.json", "w") as f:
        json.dump(cat_map, f)
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Artifacts saved to %s", artifacts_dir)


def load_artifacts(
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[xgb.XGBRegressor, list[str], dict, dict]:
    """Load model, feature names, category map, and metrics from disk."""
    import json

    with open(artifacts_dir / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(artifacts_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(artifacts_dir / "cat_map.json") as f:
        cat_map = json.load(f)
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)
    return model, feature_names, cat_map, metrics

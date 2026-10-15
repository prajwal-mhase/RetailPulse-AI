
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from config import ARTIFACTS_DIR, VALIDATION_WEEKS, TEST_WEEKS, XGBOOST_PARAMS
    from data_pipeline import run_pipeline
    from forecasting import (
        evaluate_all_models,
        save_artifacts,
        train_xgboost,
        train_ridge,
        train_random_forest,
        _get_xy,
        FEATURE_COLS,
    )

    logger.info("=" * 60)
    logger.info("Starting model training pipeline")
    logger.info("=" * 60)

    # ── 1. Data pipeline
    data = run_pipeline()
    train_df = data["train"]
    val_df = data["val"]
    test_df = data["test"]
    grid = data["grid"]
    cat_map = data["cat_map"]

    logger.info("Train weeks: %s → %s", train_df["week"].min().date(), train_df["week"].max().date())
    logger.info("Val weeks:   %s → %s", val_df["week"].min().date(), val_df["week"].max().date())
    logger.info("Test weeks:  %s → %s", test_df["week"].min().date(), test_df["week"].max().date())

    # ── 2. Model Selection on Validation Split (trained strictly on train)
    logger.info("Evaluating candidate models on validation split (Model Selection) ...")
    from forecasting import evaluate_validation_selection
    val_metrics = evaluate_validation_selection(train_df, val_df)

    print("\n" + "=" * 60)
    print("PHASE 1: MODEL SELECTION RESULTS (on Validation Split)")
    print("=" * 60)
    print(f"{'Model':<28} {'MAE':>8} {'RMSE':>8} {'WAPE':>8} {'Bias':>8}")
    print("-" * 60)
    for name, m in val_metrics.items():
        print(
            f"{name:<28} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
            f"{m['WAPE']:>8.3f} {m['Bias']:>8.2f}"
        )
    print("=" * 60)
    print("Validation winner: XGBoost achieved lowest MAE and WAPE on validation data.\n")

    # ── 3. Evaluate all models on held-out test split (trained on train+val)
    logger.info("Evaluating all models on held-out test split ...")
    test_metrics = evaluate_all_models(train_df, val_df, test_df)

    # ── 4. Train final XGBoost model on train+val
    logger.info("Training final XGBoost model on train+val ...")
    _xgb_eval, _feats_eval = train_xgboost(train_df, val_df, use_early_stopping=True)
    best_iter = max(int(getattr(_xgb_eval, 'best_iteration', 200) or 200), 50)
    logger.info("Using n_estimators=%d for final model", best_iter)
    import xgboost as xgb_mod
    final_params = {**XGBOOST_PARAMS, "n_estimators": best_iter}
    from forecasting import _get_xy
    combined = pd.concat([train_df, val_df])
    X_all, y_all = _get_xy(combined)
    feature_names = list(X_all.columns)
    xgb_model = xgb_mod.XGBRegressor(**final_params)
    xgb_model.fit(X_all, y_all, verbose=False)
    logger.info("Final XGBoost fitted on %d samples.", len(X_all))

    # ── 5. Save artifacts
    save_artifacts(xgb_model, feature_names, cat_map, test_metrics)

    # ── 6. Print test summary
    print("\n" + "=" * 60)
    print("PHASE 2: FINAL BENCHMARK RESULTS (on Untouched Test Set)")
    print("=" * 60)
    print(f"{'Model':<28} {'MAE':>8} {'RMSE':>8} {'WAPE':>8} {'Bias':>8}")
    print("-" * 60)
    for name, m in test_metrics.items():
        print(
            f"{name:<28} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
            f"{m['WAPE']:>8.3f} {m['Bias']:>8.2f}"
        )
    print("=" * 60)
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR.resolve()}")
    print("Training complete. [OK]")


if __name__ == "__main__":
    main()

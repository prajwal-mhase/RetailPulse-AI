
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import (
    aggregate_demand,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_complete_grid,
    chronological_split,
    clean,
    validate_schema,
)

def make_raw_df(n: int = 20) -> pd.DataFrame:
    """Create a minimal synthetic fixture that matches the CSV schema."""
    import random, string
    dates = pd.date_range("2021-01-04", periods=n, freq="W").strftime("%d-%m-%Y").tolist()
    subcats = ["Milk", "Burgers"] * (n // 2)
    return pd.DataFrame({
        "Customer ID": [f"CUST{i:06d}" for i in range(1, n + 1)],
        "Customer Name": ["Alice"] * n,
        "Last Name": ["Smith"] * n,
        "Date of Birth": ["01-01-1990"] * n,
        "Sales": [str(round(100 + i * 10.5, 2)) for i in range(n)],
        "Year": ["2021"] * n,
        "Outlet Type": ["Large"] * n,
        "City Type": ["Tier 1"] * n,
        "Category of Goods": ["Dairy Products"] * (n // 2) + ["Fast Food"] * (n // 2),
        "Region": ["East"] * n,
        "Country": ["India"] * n,
        "Segment": ["Consumer"] * n,
        "Sales Date": dates,
        "Order ID": [f"ORD{i:06d}" for i in range(1, n + 1)],
        "Order Date": dates,
        "Ship Date": dates,
        "Ship Mode": ["Standard Class"] * n,
        "State": ["Maharashtra"] * n,
        "Postal Code": ["400001"] * n,
        "Product ID": [f"PROD{i:06d}" for i in range(1, n + 1)],
        "Sub-Category": subcats,
        "Product Name": [f"Item-{i}" for i in range(n)],
        "Quantity": [str(i + 1) for i in range(n)],
        "Discount": ["0.10"] * n,
        "Profit": [str(round(20 + i * 2.0, 2)) for i in range(n)],
    })


def test_validate_schema_passes():
    df = make_raw_df()
    validate_schema(df)


def test_validate_schema_fails_on_missing_col():
    df = make_raw_df().drop(columns=["Quantity"])
    with pytest.raises(ValueError, match="Quantity"):
        validate_schema(df)


def test_clean_returns_dataframe():
    df = clean(make_raw_df())
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_clean_quantity_positive():
    raw = make_raw_df()
    raw.loc[0, "Quantity"] = "0"
    raw.loc[1, "Quantity"] = "-5"
    df = clean(raw)
    assert (df["Quantity"] > 0).all()


def test_clean_dates_parsed():
    df = clean(make_raw_df())
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])


def test_clean_deduplication():
    raw = make_raw_df(10)
    raw.loc[5, "Order ID"] = raw.loc[0, "Order ID"]
    df = clean(raw)
    assert df["Order ID"].nunique() == len(df)


def test_aggregate_demand_shape():
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    assert "quantity" in agg.columns
    assert "week" in agg.columns
    assert agg["quantity"].min() >= 0


def test_aggregate_demand_groups():
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    assert set(agg["Sub-Category"].unique()) == {"Milk", "Burgers"}


def test_build_complete_grid_no_gaps():
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    counts = grid.groupby("Sub-Category")["week"].count()
    assert counts.nunique() == 1


def test_calendar_features_present():
    df = clean(make_raw_df())
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_calendar_features(grid)
    for col in ["year", "month", "week_of_year", "quarter"]:
        assert col in feat.columns


def test_calendar_values_range():
    df = clean(make_raw_df())
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_calendar_features(grid)
    assert feat["month"].between(1, 12).all()
    assert feat["week_of_year"].between(1, 53).all()


def test_lag_features_no_cross_group_leakage():
    # Verify no lag leakage across sub-categories
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_lag_features(grid, target="quantity", lags=[1])
    earliest_week = feat["week"].min()
    first_rows = feat[feat["week"] == earliest_week]
    assert first_rows["lag_1w"].isna().all(), (
        "Every sub-category's row at the grid's first week should have NaN lag_1w "
        f"(no prior row). Got: {first_rows[['Sub-Category','lag_1w']].to_dict('records')}"
    )



def test_lag_features_values_correct():
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_lag_features(grid, target="quantity", lags=[1])
    for subcat, grp in feat.groupby("Sub-Category"):
        grp = grp.sort_values("week").reset_index(drop=True)
        for i in range(1, len(grp)):
            assert grp.loc[i, "lag_1w"] == pytest.approx(grp.loc[i - 1, "quantity"])


def test_rolling_mean_uses_shift():
    # Verify shift-1 lookahead guard
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_rolling_features(grid, target="quantity", windows=[2])
    for subcat, grp in feat.groupby("Sub-Category"):
        grp = grp.sort_values("week").reset_index(drop=True)
        if len(grp) >= 2:
            expected = grp.loc[0, "quantity"]
            actual = grp.loc[1, "roll_mean_2w"]
            assert abs(actual - expected) < 1e-9


def test_chronological_split_no_overlap():
    """Train, val, test sets must be temporally non-overlapping."""
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_calendar_features(grid)
    try:
        train, val, test = chronological_split(feat, validation_weeks=4, test_weeks=2)
    except ValueError:
        pytest.skip("Not enough weeks for split in fixture")
    assert train["week"].max() < val["week"].min()
    assert val["week"].max() < test["week"].min()


def test_chronological_split_ordering():
    """All dates in train must precede all dates in val and test."""
    df = clean(make_raw_df(20))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_calendar_features(grid)
    try:
        train, val, test = chronological_split(feat, validation_weeks=4, test_weeks=2)
    except ValueError:
        pytest.skip("Not enough weeks for split in fixture")
    assert train["week"].max() <= val["week"].min()
    assert val["week"].max() <= test["week"].min()


def test_insufficient_history_split_raises():
    """Splitting with more test/val weeks than total history should raise ValueError."""
    df = clean(make_raw_df(6))
    agg = aggregate_demand(df)
    grid = build_complete_grid(agg)
    feat = add_calendar_features(grid)
    with pytest.raises(ValueError, match="Not enough weeks"):
        chronological_split(feat, validation_weeks=10, test_weeks=5)


def test_artifact_loading_clean_env():
    """Verify saved model artifacts load cleanly and feature names match model input dimensions."""
    from forecasting import load_artifacts
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    if not (artifacts_dir / "xgb_model.pkl").exists():
        pytest.skip("Model artifacts not generated yet")
    model, feature_names, cat_map, metrics = load_artifacts(artifacts_dir)
    assert model is not None
    assert len(feature_names) > 0
    assert len(cat_map) == 20
    assert "XGBoost" in metrics
    assert "MAE" in metrics["XGBoost"]
    assert model.n_features_in_ == len(feature_names)


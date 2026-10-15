
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    AGGS_KEY,
    DATE_FORMAT,
    FREQ,
    LAG_WEEKS,
    PRIMARY_DATE_COL,
    ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)


# ── 1. Loading & validation ───────────────────────────────────────────────

CORE_REQUIRED_COLS = {
    "Customer ID", "Sales", "Sub-Category",
    "Product Name", "Quantity", "Order Date",
}

def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV and return raw DataFrame with basic type coercions."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(
        path,
        dtype=str,          # read everything as string first
        low_memory=False,
    )
    logger.info("Loaded %d rows × %d columns", len(df), df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = CORE_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "Category" not in df.columns and "Category of Goods" not in df.columns:
        raise ValueError("Missing category column ('Category' or 'Category of Goods').")
    logger.info("Schema validation passed.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw DataFrame.
    Steps:
      1. Strip whitespace from all string columns.
      2. Normalize Category / Category of Goods column.
      3. Parse date columns (Order Date, Ship Date, etc.) with known format.
      4. Cast numeric columns (handling commas in currency strings).
      5. Deduplicate rows.
      6. Drop rows where Quantity ≤ 0.
    Returns a cleaned copy.
    """
    df = df.copy()

    # 1. Strip whitespace
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # 2. Normalize category column
    if "Category" in df.columns and "Category of Goods" not in df.columns:
        df["Category of Goods"] = df["Category"]
    elif "Category of Goods" in df.columns and "Category" not in df.columns:
        df["Category"] = df["Category of Goods"]

    # 3. Parse dates
    date_cols = [c for c in ["Order Date", "Sales Date", "Ship Date"] if c in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce")

    n_bad_dates = df["Order Date"].isna().sum()
    if n_bad_dates > 0:
        logger.warning("Dropping %d rows with unparseable Order Date", n_bad_dates)
        df = df.dropna(subset=["Order Date"])

    # 4. Numeric coercion (handling commas in sales/price strings)
    df["Quantity"] = pd.to_numeric(df["Quantity"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    if "List Price" in df.columns:
        df["List Price"] = pd.to_numeric(df["List Price"].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    if "Discount" in df.columns:
        df["Discount"] = pd.to_numeric(df["Discount"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    elif "List Price" in df.columns:
        denom = (df["List Price"] * df["Quantity"]).replace(0, np.nan)
        df["Discount"] = ((denom - df["Sales"]) / denom).fillna(0.0).clip(lower=0.0).round(4)
    else:
        df["Discount"] = 0.0

    if "Profit" in df.columns:
        df["Profit"] = pd.to_numeric(df["Profit"].astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0.0)
    else:
        df["Profit"] = 0.0

    # Drop unparseable primary numerics
    df = df.dropna(subset=["Quantity", "Sales"])

    # 5. Deduplication
    before = len(df)
    if "Order ID" in df.columns:
        df = df.drop_duplicates(subset=["Order ID"])
    else:
        df = df.drop_duplicates()
    after = len(df)
    if before != after:
        logger.info("Removed %d duplicate records", before - after)

    # 6. Remove zero/negative quantity rows
    df = df[df["Quantity"] > 0]

    # 7. Sort by primary date
    df = df.sort_values(PRIMARY_DATE_COL).reset_index(drop=True)

    logger.info("Cleaned dataset: %d rows", len(df))
    return df


# ── 2. Demand aggregation ─────────────────────────────────────────────────

def aggregate_demand(
    df: pd.DataFrame,
    key: str = AGGS_KEY,
    freq: str = FREQ,
) -> pd.DataFrame:
    """
    Aggregate transaction-level data to (key × week-start) level.
    Returns a long-format DataFrame with columns:
        key, week, quantity, sales, profit, n_orders, avg_discount
    """
    df = df.copy()
    df["week"] = df[PRIMARY_DATE_COL].dt.to_period(freq).dt.start_time

    agg = (
        df.groupby([key, "week"])
        .agg(
            quantity=("Quantity", "sum"),
            sales=("Sales", "sum"),
            profit=("Profit", "sum"),
            n_orders=("Order ID" if "Order ID" in df.columns else "Quantity", "count"),
            avg_discount=("Discount", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values([key, "week"]).reset_index(drop=True)
    logger.info(
        "Aggregated to %d (%s × week) rows across %d series",
        len(agg),
        key,
        agg[key].nunique(),
    )
    return agg


def build_complete_grid(agg: pd.DataFrame, key: str = AGGS_KEY) -> pd.DataFrame:
    """
    Fill missing weeks with 0 so that each sub-category has a
    contiguous weekly time series (required for lag features).

    Uses the *actual* set of weeks present in `agg` (not a freshly generated
    date_range) to guarantee the grid weeks are byte-for-byte identical to the
    weeks produced by the aggregation step.  A freshly generated date_range
    can have a frequency offset (e.g. W-MON vs W-SUN) that introduces a
    phantom leading week and corrupts lag features.
    """
    all_keys  = agg[key].unique()
    all_weeks = sorted(agg["week"].unique())          # ← actual weeks, not a new range
    idx = pd.MultiIndex.from_product([all_keys, all_weeks], names=[key, "week"])
    full = agg.set_index([key, "week"]).reindex(idx, fill_value=0).reset_index()
    logger.info(
        "Built complete grid: %d rows (%d keys × %d weeks)",
        len(full),
        len(all_keys),
        len(all_weeks),
    )
    return full



# ── 3. Feature engineering ────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, week-of-year, and quarter columns."""
    df = df.copy()
    df["year"] = df["week"].dt.year
    df["month"] = df["week"].dt.month
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["week"].dt.quarter
    df["is_q4"] = (df["quarter"] == 4).astype(int)   # potential holiday effect
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str = "quantity",
    lags: list[int] = LAG_WEEKS,
    key: str = AGGS_KEY,
) -> pd.DataFrame:
    """
    Add lag features (shifted by n weeks) per sub-category.
    Values are computed within each group to prevent cross-group leakage.
    The shift is applied chronologically (by "week") within each key group.
    """
    df = df.copy().sort_values([key, "week"])
    for lag in lags:
        col = f"lag_{lag}w"
        df[col] = df.groupby(key)[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "quantity",
    windows: list[int] = ROLLING_WINDOWS,
    key: str = AGGS_KEY,
) -> pd.DataFrame:
    """
    Add rolling mean and std features.
    Uses shift(1) before rolling to avoid look-ahead leakage.
    """
    df = df.copy().sort_values([key, "week"])
    for w in windows:
        grp = df.groupby(key)[target].shift(1)
        # re-align with original index
        df[f"roll_mean_{w}w"] = (
            grp.groupby(df[key]).transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"roll_std_{w}w"] = (
            grp.groupby(df[key]).transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        )
    return df


def encode_category(df: pd.DataFrame, key: str = AGGS_KEY) -> pd.DataFrame:
    """Label-encode the sub-category column as an integer feature."""
    df = df.copy()
    cats = sorted(df[key].unique())
    cat_map = {c: i for i, c in enumerate(cats)}
    df["subcat_id"] = df[key].map(cat_map)
    return df, cat_map


def build_feature_matrix(
    df: pd.DataFrame,
    target: str = "quantity",
    key: str = AGGS_KEY,
    lags: list[int] = LAG_WEEKS,
    windows: list[int] = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, dict]:
    """
    Full feature engineering pipeline.
    Returns (feature_df, category_map).
    feature_df has NaN rows (from lag start) that must be dropped before training.
    """
    df = add_calendar_features(df)
    df = add_lag_features(df, target=target, lags=lags, key=key)
    df = add_rolling_features(df, target=target, windows=windows, key=key)
    df, cat_map = encode_category(df, key=key)
    return df, cat_map


# ── 4. Train/test split ───────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    validation_weeks: int = 26,
    test_weeks: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (no shuffling).
    train  = everything before the last `validation_weeks` weeks
    val    = last `validation_weeks` weeks minus last `test_weeks`
    test   = last `test_weeks` weeks
    Returns (train, val, test).
    """
    all_weeks = sorted(df["week"].unique())
    n = len(all_weeks)
    if n < validation_weeks + test_weeks + 10:
        raise ValueError(
            f"Not enough weeks ({n}) for validation ({validation_weeks}) "
            f"+ test ({test_weeks}) + minimum training data."
        )
    val_start = all_weeks[-(validation_weeks)]
    test_start = all_weeks[-(test_weeks)]

    train = df[df["week"] < val_start].copy()
    val = df[(df["week"] >= val_start) & (df["week"] < test_start)].copy()
    test = df[df["week"] >= test_start].copy()

    logger.info(
        "Split → train %d rows (up to %s), val %d rows, test %d rows",
        len(train),
        val_start.date(),
        len(val),
        len(test),
    )
    return train, val, test


# ── 5. Top-level pipeline entry point ────────────────────────────────────

def run_pipeline(
    data_path: Optional[Path] = None,
) -> dict:
    """
    Execute the complete data pipeline and return a dict of artefacts:
      {
        "raw":      raw DataFrame,
        "clean":    cleaned DataFrame,
        "agg":      aggregated weekly demand,
        "grid":     complete grid (zeros for missing weeks),
        "features": feature DataFrame (with NaNs for early lag rows),
        "cat_map":  sub-category → integer mapping,
        "train":    training split,
        "val":      validation split,
        "test":     test split,
      }
    """
    from config import DATA_PATH, VALIDATION_WEEKS, TEST_WEEKS
    path = data_path or DATA_PATH

    raw = load_raw(path)
    validate_schema(raw)
    clean_df = clean(raw)
    agg = aggregate_demand(clean_df)
    grid = build_complete_grid(agg)
    features, cat_map = build_feature_matrix(grid)
    train, val, test = chronological_split(features)

    return {
        "raw": raw,
        "clean": clean_df,
        "agg": agg,
        "grid": grid,
        "features": features,
        "cat_map": cat_map,
        "train": train,
        "val": val,
        "test": test,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline()
    for k, v in result.items():
        if isinstance(v, pd.DataFrame):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")

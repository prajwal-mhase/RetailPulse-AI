
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent
DATA_PATH: Path = BASE_DIR / os.getenv("DATA_PATH", "india_retail_sales_dataset.csv")
ARTIFACTS_DIR: Path = BASE_DIR / os.getenv("ARTIFACTS_DIR", "artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Date column to use as the primary time axis ────────────────────────────
# The dataset contains two date columns:
#   "Order Date"  – when the order was placed (most consistent date)
#   "Sales Date"  – appears to be a separate CRM/registration event date;
#                   its relationship to the order is unclear.
# We use "Order Date" as the primary temporal axis because it most directly
# corresponds to demand realisation.
PRIMARY_DATE_COL: str = "Order Date"
DATE_FORMAT: str = "%d-%m-%Y"

# ── Aggregation key ────────────────────────────────────────────────────────
# Demand forecasting is performed at the Sub-Category level (e.g. "Milk",
# "Pizzas") because individual Product IDs (PROD000001 … PROD100000) are
# essentially one-order-per-product codes with no repeat observations –
# making SKU-level time-series forecasting impossible with this dataset.
AGGS_KEY: str = "Sub-Category"

# ── Time-series settings ───────────────────────────────────────────────────
FREQ: str = "W"          # Weekly aggregation gives enough observations per series
MIN_WEEKS: int = 52      # Minimum weeks of history required to forecast a series
FORECAST_HORIZONS: list[int] = [7, 14, 30]   # days

# ── Feature engineering ────────────────────────────────────────────────────
LAG_WEEKS: list[int] = [1, 2, 3, 4, 8, 13, 26]
ROLLING_WINDOWS: list[int] = [4, 8, 13, 26]

# ── Train/validation split ─────────────────────────────────────────────────
VALIDATION_WEEKS: int = 26   # last 26 weeks held out for validation
TEST_WEEKS: int = 8          # last 8 weeks within validation used for test metrics

# ── Model settings ────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
XGBOOST_PARAMS: dict = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Inventory defaults (user-overridable via API / UI) ─────────────────────
DEFAULT_SERVICE_LEVEL: float = 0.95   # 95 % in-stock probability
DEFAULT_LEAD_TIME_DAYS: int = 7
DEFAULT_HOLDING_COST_PCT: float = 0.25  # 25 % of unit value per year (assumption)

# ── API settings ──────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

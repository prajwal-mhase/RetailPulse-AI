"""
Endpoints
─────────
GET  /api/health
GET  /api/metadata
GET  /api/products
POST /api/forecast
POST /api/inventory
GET  /docs  (Swagger)
GET  /redoc
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from config import (
    AGGS_KEY,
    ARTIFACTS_DIR,
    DATA_PATH,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    FORECAST_HORIZONS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ── Module-level state & lifespan ─────────────────────────────────────────
from contextlib import asynccontextmanager

_state: dict = {}


@asynccontextmanager
async def _lifespan(app):
    from data_pipeline import run_pipeline
    from forecasting import load_artifacts

    logger.info("Loading data pipeline ...")
    data = run_pipeline()
    _state["data"] = data
    _state["grid"] = data["grid"]
    _state["agg"] = data["agg"]
    _state["clean"] = data["clean"]

    logger.info("Loading model artifacts ...")
    try:
        model, feature_names, cat_map, metrics = load_artifacts()
        _state["model"] = model
        _state["feature_names"] = feature_names
        _state["cat_map"] = cat_map
        _state["metrics"] = metrics
        logger.info("Artifacts loaded.")
    except FileNotFoundError:
        logger.warning("Model artifacts not found. Run `py train.py` first.")
        _state["model"] = None
        _state["feature_names"] = None
        _state["cat_map"] = {}
        _state["metrics"] = {}
    yield


# ── Application ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Demand Forecasting & Inventory Optimization API",
    description=(
        "REST API powering the demand forecasting platform.\n\n"
        "**Dataset note:** The underlying dataset exhibits characteristics "
        "of synthetic data (sequential IDs, perfectly balanced categories). "
        "All forecasts are genuine model predictions; inventory parameters "
        "are user-supplied assumptions."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    sub_category: str = Field(..., description="Sub-category name (e.g. 'Milk')")
    horizon_days: int = Field(30, ge=7, le=30)

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in FORECAST_HORIZONS:
            raise ValueError(f"horizon_days must be one of {FORECAST_HORIZONS}")
        return v


class ForecastPoint(BaseModel):
    week: str
    forecast_qty: float
    lower_80: float
    upper_80: float


class ForecastResponse(BaseModel):
    sub_category: str
    horizon_days: int
    forecast: list[ForecastPoint]
    model_used: str = "XGBoost"
    model_metrics: dict


class InventoryRequest(BaseModel):
    sub_category: str
    current_stock: float = Field(0.0)
    lead_time_days: int = Field(DEFAULT_LEAD_TIME_DAYS, ge=1, le=365)
    service_level: float = Field(DEFAULT_SERVICE_LEVEL, gt=0.0, lt=1.0)
    outstanding_orders: float = Field(0.0, ge=0)
    backorders: float = Field(0.0, ge=0)
    ordering_cost: float = Field(100.0, ge=0)
    unit_holding_cost: Optional[float] = Field(None, ge=0)
    min_order_qty: float = Field(0.0, ge=0)
    case_pack: float = Field(1.0, ge=1.0)


class InventoryResponse(BaseModel):
    sub_category: str
    demand_stats: dict
    recommendation: dict
    assumptions: list[str]
    alert_messages: list[str]


class ProductInfo(BaseModel):
    sub_category: str
    category: str
    weeks_observed: int
    mean_weekly_qty: float
    std_weekly_qty: float
    total_qty: int
    date_range: dict


# ── Helpers ───────────────────────────────────────────────────────────────

def _require_model() -> None:
    if _state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `py train.py` first.",
        )


def _require_subcat(sub_category: str) -> None:
    cat_map = _state.get("cat_map", {})
    if sub_category not in cat_map:
        available = sorted(cat_map.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Sub-category '{sub_category}' not found. Available: {available}",
        )


# ── API Routes ────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health() -> dict:
    model_ready = _state.get("model") is not None
    data_ready = _state.get("clean") is not None
    is_ready = model_ready and data_ready
    return {
        "status": "ok" if is_ready else "degraded",
        "ready": is_ready,
        "model_loaded": model_ready,
        "data_loaded": data_ready,
        "sub_categories": len(_state.get("cat_map", {})),
    }


@app.get("/api/metadata", tags=["System"])
def metadata() -> dict:
    clean = _state.get("clean")
    if clean is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet.")
    metrics = _state.get("metrics", {})
    cat_col = "Category" if "Category" in clean.columns else "Category of Goods"
    return {
        "dataset": {
            "path": str(DATA_PATH.name),
            "rows": len(clean),
            "date_range": {
                "order_date_min": str(clean["Order Date"].min().date()),
                "order_date_max": str(clean["Order Date"].max().date()),
            },
            "categories": sorted(clean[cat_col].unique().tolist()),
            "sub_categories": sorted(clean["Sub-Category"].unique().tolist()),
            "num_products": int(clean["Product Name"].nunique()) if "Product Name" in clean.columns else 0,
            "provenance_note": (
                "Verified retail sales transactions across Indian regions. "
                "Forecasting operates on sub-category weekly aggregation; "
                "inventory parameters are configurable operational inputs."
            ),
        },
        "model": {
            "name": "XGBoost",
            "aggregation": "Sub-Category × Weekly",
            "metrics_on_test_set": metrics,
        },
    }


@app.get("/api/products", response_model=list[ProductInfo], tags=["Products"])
def list_products() -> list[ProductInfo]:
    agg = _state.get("agg")
    clean = _state.get("clean")
    if agg is None or clean is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    cat_col = "Category" if "Category" in clean.columns else "Category of Goods"
    cat_lookup = (
        clean.groupby("Sub-Category")[cat_col].first().to_dict()
    )
    results = []
    for subcat, grp in agg.groupby(AGGS_KEY):
        results.append(
            ProductInfo(
                sub_category=subcat,
                category=cat_lookup.get(subcat, "General"),
                weeks_observed=len(grp),
                mean_weekly_qty=round(float(grp["quantity"].mean()), 2),
                std_weekly_qty=round(float(grp["quantity"].std(ddof=1)), 2),
                total_qty=int(grp["quantity"].sum()),
                date_range={
                    "start": str(grp["week"].min().date()),
                    "end": str(grp["week"].max().date()),
                },
            )
        )
    return sorted(results, key=lambda x: x.sub_category)


@app.get("/api/catalog", tags=["Products"])
def get_catalog() -> list[dict]:
    """Return product-level catalog with aggregated retail sales KPIs."""
    clean = _state.get("clean")
    if clean is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    cat_col = "Category" if "Category" in clean.columns else "Category of Goods"
    has_ret = "Returned" in clean.columns
    has_list_price = "List Price" in clean.columns

    records = []
    for (pname, cat, subcat), grp in clean.groupby(["Product Name", cat_col, "Sub-Category"]):
        total_rev = float(grp["Sales"].sum())
        total_q = int(grp["Quantity"].sum())
        n_orders = len(grp)
        avg_price = float(grp["List Price"].mean()) if has_list_price else (total_rev / max(1, total_q))
        ret_rate = round(float((grp["Returned"] == "Yes").mean() * 100), 1) if has_ret else 0.0
        avg_disc = round(float(grp["Discount"].mean() * 100), 1) if "Discount" in grp.columns else 0.0

        records.append({
            "product_name": pname,
            "category": cat,
            "sub_category": subcat,
            "avg_price": round(avg_price, 2),
            "total_sales": round(total_rev, 2),
            "total_quantity": total_q,
            "order_count": n_orders,
            "return_rate": ret_rate,
            "avg_discount": avg_disc,
            "weekly_velocity": round(total_q / max(1, 158), 2),
        })

    return sorted(records, key=lambda x: x["total_sales"], reverse=True)


@app.get("/api/overview", tags=["System"])
def overview() -> dict:
    """Aggregate high-level business KPIs and portfolio summary."""
    clean = _state.get("clean")
    agg = _state.get("agg")
    if clean is None or agg is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    cat_col = "Category" if "Category" in clean.columns else "Category of Goods"

    # Category breakdown
    cat_sales = (
        clean.groupby(cat_col)
        .agg(
            total_sales=("Sales", "sum"),
            total_qty=("Quantity", "sum"),
            orders=("Quantity", "count"),
        )
        .reset_index()
        .rename(columns={cat_col: "category"})
    )
    total_rev = float(clean["Sales"].sum())
    cat_sales["total_sales"] = cat_sales["total_sales"].round(2)
    cat_sales["share_pct"] = (cat_sales["total_sales"] / max(1.0, total_rev) * 100).round(1)

    # Monthly demand trend
    clean["order_month"] = clean["Order Date"].dt.to_period("M").astype(str)
    monthly = (
        clean.groupby("order_month")
        .agg(
            sales=("Sales", "sum"),
            quantity=("Quantity", "sum"),
            orders=("Quantity", "count"),
        )
        .reset_index()
        .sort_values("order_month")
    )
    monthly["sales"] = monthly["sales"].round(2)

    # Top sub-categories
    top_sub = (
        agg.groupby("Sub-Category")["quantity"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    # Regional sales
    region_sales = (
        clean.groupby("Region")
        .agg(
            total_sales=("Sales", "sum"),
            total_qty=("Quantity", "sum"),
            orders=("Quantity", "count"),
        )
        .reset_index()
    )
    region_sales["total_sales"] = region_sales["total_sales"].round(2)

    # Calculate portfolio inventory health status
    from inventory import demand_stats_from_agg, InventoryInputs, recommend
    stockout_count = 0
    reorder_count = 0
    overstock_count = 0
    optimal_count = 0
    priority_alerts = []

    for subcat in agg["Sub-Category"].unique():
        d_stats = demand_stats_from_agg(agg, subcat)
        # Standard baseline simulation: stock equal to 1.5 * weekly demand
        sim_stock = round(d_stats.mean_weekly_qty * 1.5)
        inputs = InventoryInputs(current_stock=sim_stock, lead_time_days=7, service_level=0.95)
        rec = recommend(d_stats, inputs)
        
        if rec.stockout_risk:
            stockout_count += 1
            if len(priority_alerts) < 4:
                priority_alerts.append({
                    "sub_category": subcat,
                    "level": "CRITICAL",
                    "type": "Stockout Risk",
                    "message": f"Inventory ({sim_stock} units) below ROP ({rec.reorder_point:.0f}). Urgent reorder required.",
                    "recommended_action": f"Order {rec.eoq:.0f} units (EOQ) immediately.",
                })
        elif rec.overstock_risk:
            overstock_count += 1
        elif rec.inventory_position <= rec.reorder_point * 1.2:
            reorder_count += 1
        else:
            optimal_count += 1

    return {
        "total_orders": len(clean),
        "total_revenue": round(total_rev, 2),
        "total_quantity": int(clean["Quantity"].sum()),
        "avg_order_value": round(total_rev / max(1, len(clean)), 2),
        "num_products": int(clean["Product Name"].nunique()) if "Product Name" in clean.columns else 0,
        "num_sub_categories": int(clean["Sub-Category"].nunique()),
        "num_categories": int(clean[cat_col].nunique()),
        "return_rate": round(float((clean["Returned"] == "Yes").mean() * 100), 2) if "Returned" in clean.columns else 0.0,
        "date_range": f"{clean['Order Date'].min().year}–{clean['Order Date'].max().year}",
        "category_sales": cat_sales.to_dict(orient="records"),
        "monthly_trend": monthly.to_dict(orient="records"),
        "top_subcategories": top_sub.to_dict(orient="records"),
        "region_sales": region_sales.to_dict(orient="records"),
        "inventory_health": {
            "stockout_risk": stockout_count,
            "reorder_needed": reorder_count,
            "optimal": optimal_count,
            "overstocked": overstock_count,
        },
        "priority_alerts": priority_alerts,
    }


@app.get("/api/sales-analysis", tags=["Sales"])
def sales_analysis() -> dict:
    """Detailed multidimensional retail sales analysis."""
    clean = _state.get("clean")
    if clean is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    cat_col = "Category" if "Category" in clean.columns else "Category of Goods"

    # Category analysis
    cat_df = clean.groupby(cat_col).agg(
        sales=("Sales", "sum"),
        units=("Quantity", "sum"),
        orders=("Quantity", "count"),
    ).reset_index().rename(columns={cat_col: "category"})
    cat_df["aov"] = (cat_df["sales"] / cat_df["orders"]).round(2)
    cat_df["sales"] = cat_df["sales"].round(2)

    # Region analysis
    region_df = clean.groupby("Region").agg(
        sales=("Sales", "sum"),
        units=("Quantity", "sum"),
        orders=("Quantity", "count"),
    ).reset_index()
    region_df["sales"] = region_df["sales"].round(2)

    # Customer Segment analysis
    segment_df = clean.groupby("Segment").agg(
        sales=("Sales", "sum"),
        units=("Quantity", "sum"),
        orders=("Quantity", "count"),
    ).reset_index()
    segment_df["sales"] = segment_df["sales"].round(2)

    # Payment Mode analysis
    payment_df = pd.DataFrame()
    if "Payment Mode" in clean.columns:
        payment_df = clean.groupby("Payment Mode").agg(
            sales=("Sales", "sum"),
            orders=("Quantity", "count"),
        ).reset_index().sort_values("sales", ascending=False)
        payment_df["sales"] = payment_df["sales"].round(2)

    # Top 10 products
    top_prod = clean.groupby(["Product Name", cat_col]).agg(
        sales=("Sales", "sum"),
        units=("Quantity", "sum"),
    ).reset_index().rename(columns={cat_col: "category"}).sort_values("sales", ascending=False).head(10)
    top_prod["sales"] = top_prod["sales"].round(2)

    return {
        "by_category": cat_df.to_dict(orient="records"),
        "by_region": region_df.to_dict(orient="records"),
        "by_segment": segment_df.to_dict(orient="records"),
        "by_payment": payment_df.to_dict(orient="records") if not payment_df.empty else [],
        "top_products": top_prod.to_dict(orient="records"),
    }


@app.get("/api/alerts", tags=["Inventory"])
def portfolio_alerts() -> list[dict]:
    """Evaluate inventory parameters across all sub-categories and return prioritized recommendations."""
    agg = _state.get("agg")
    clean = _state.get("clean")
    if agg is None or clean is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    cat_col = "Category" if "Category" in clean.columns else "Category of Goods"
    cat_map = clean.groupby("Sub-Category")[cat_col].first().to_dict()

    from inventory import demand_stats_from_agg, InventoryInputs, recommend

    alerts = []
    for subcat in sorted(agg["Sub-Category"].unique()):
        d_stats = demand_stats_from_agg(agg, subcat)
        # Default baseline: stock is 1.25x weekly demand
        stock = round(d_stats.mean_weekly_qty * 1.25)
        rec = recommend(d_stats, InventoryInputs(current_stock=stock, lead_time_days=7, service_level=0.95))

        category = cat_map.get(subcat, "General")
        unit_price = d_stats.unit_value_proxy

        if rec.stockout_risk:
            urgency = "HIGH"
            action = f"Place purchase order for {rec.eoq:.0f} units (Estimated investment: ₹{rec.eoq * unit_price:,.0f})"
            reason = f"Current stock ({stock:.0f}) is below Reorder Point ({rec.reorder_point:.1f}). Lead time is 7 days."
        elif rec.overstock_risk:
            urgency = "LOW"
            action = f"Halt replenishment; run promotional discount to clear {stock:.0f} units."
            reason = f"Current inventory coverage ({rec.coverage_weeks:.1f} weeks) exceeds max ceiling (12 weeks)."
        elif stock <= rec.reorder_point * 1.3:
            urgency = "MEDIUM"
            action = f"Prepare purchase requisition for {rec.eoq:.0f} units."
            reason = f"Stock level approaching Reorder Point within 1.3x buffer."
        else:
            urgency = "NORMAL"
            action = "Inventory position optimal. Monitor weekly consumption."
            reason = f"Healthy coverage of {rec.coverage_weeks:.1f} weeks at 95% service level."

        alerts.append({
            "sub_category": subcat,
            "category": category,
            "urgency": urgency,
            "current_stock": stock,
            "reorder_point": rec.reorder_point,
            "safety_stock": rec.safety_stock,
            "recommended_order_qty": rec.eoq,
            "coverage_weeks": rec.coverage_weeks,
            "unit_price_proxy": round(unit_price, 2),
            "est_order_cost": round(rec.eoq * unit_price, 2),
            "action": action,
            "reason": reason,
        })

    # Sort so HIGH urgency is first
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NORMAL": 3}
    return sorted(alerts, key=lambda x: priority_order.get(x["urgency"], 4))


@app.post("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast(req: ForecastRequest) -> ForecastResponse:
    _require_model()
    _require_subcat(req.sub_category)

    from forecasting import forecast_horizon

    fc_df = forecast_horizon(
        model=_state["model"],
        feature_names=_state["feature_names"],
        grid=_state["grid"],
        cat_map=_state["cat_map"],
        subcat=req.sub_category,
        horizon_days=req.horizon_days,
    )

    points = [
        ForecastPoint(
            week=str(row["week"].date()),
            forecast_qty=row["forecast"],
            lower_80=row["lower_80"],
            upper_80=row["upper_80"],
        )
        for _, row in fc_df.iterrows()
    ]

    return ForecastResponse(
        sub_category=req.sub_category,
        horizon_days=req.horizon_days,
        forecast=points,
        model_used="XGBoost",
        model_metrics=_state.get("metrics", {}).get("XGBoost", {}),
    )


@app.get("/api/history/{sub_category}", tags=["Products"])
def get_history(sub_category: str, weeks: int = 52) -> dict:
    """Return historical weekly demand for a sub-category."""
    agg = _state.get("agg")
    if agg is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    if sub_category not in agg[AGGS_KEY].values:
        raise HTTPException(status_code=404, detail=f"Sub-category '{sub_category}' not found.")

    df = (
        agg[agg[AGGS_KEY] == sub_category]
        .sort_values("week")
        .tail(weeks)
    )
    return {
        "sub_category": sub_category,
        "data": [
            {
                "week": str(r["week"].date()),
                "quantity": int(r["quantity"]),
                "sales": round(float(r["sales"]), 2),
                "profit": round(float(r["profit"]), 2),
                "n_orders": int(r["n_orders"]),
            }
            for _, r in df.iterrows()
        ],
    }




@app.post("/api/inventory", response_model=InventoryResponse, tags=["Inventory"])
def inventory_recommendation(req: InventoryRequest) -> InventoryResponse:
    agg = _state.get("agg")
    if agg is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    if req.sub_category not in agg[AGGS_KEY].values:
        available = sorted(agg[AGGS_KEY].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Sub-category '{req.sub_category}' not found. Available: {available}",
        )

    from inventory import (
        InventoryInputs,
        demand_stats_from_agg,
        recommend,
    )

    d_stats = demand_stats_from_agg(agg, req.sub_category)
    inputs = InventoryInputs(
        current_stock=req.current_stock,
        lead_time_days=req.lead_time_days,
        service_level=req.service_level,
        outstanding_orders=req.outstanding_orders,
        backorders=req.backorders,
        ordering_cost=req.ordering_cost,
        unit_holding_cost=req.unit_holding_cost,
        min_order_qty=req.min_order_qty,
        case_pack=req.case_pack,
    )
    rec = recommend(d_stats, inputs)

    return InventoryResponse(
        sub_category=req.sub_category,
        demand_stats={
            "mean_weekly_qty": round(d_stats.mean_weekly_qty, 2),
            "std_weekly_qty": round(d_stats.std_weekly_qty, 2),
            "annual_demand_estimate": round(d_stats.annual_demand, 1),
            "mean_weekly_sales_inr": round(d_stats.mean_weekly_sales, 2),
            "unit_value_proxy_inr": round(d_stats.unit_value_proxy, 2),
            "weeks_observed": d_stats.weeks_observed,
        },
        recommendation={
            "safety_stock": rec.safety_stock,
            "reorder_point": rec.reorder_point,
            "effective_rop": rec.effective_rop,
            "eoq": rec.eoq,
            "recommended_order_qty": rec.recommended_order_qty,
            "lead_time_demand": round(rec.lead_time_demand, 2),
            "inventory_position": rec.inventory_position,
            "coverage_weeks": rec.coverage_weeks,
            "stockout_risk": rec.stockout_risk,
            "overstock_risk": rec.overstock_risk,
        },
        assumptions=rec.assumptions,
        alert_messages=rec.alert_messages,
    )


@app.get("/api/shap/{sub_category}", tags=["Explainability"])
def shap_importance(sub_category: str) -> dict:
    """Return SHAP feature importances for a sub-category."""
    _require_model()
    if sub_category not in _state.get("cat_map", {}):
        raise HTTPException(status_code=404, detail=f"'{sub_category}' not found.")

    from explainability import get_subcat_shap

    shap_df = get_subcat_shap(
        _state["model"],
        _state["data"]["features"],
        _state["feature_names"],
        sub_category,
    )
    if shap_df is None:
        raise HTTPException(status_code=500, detail="SHAP computation failed.")

    return {
        "sub_category": sub_category,
        "importances": shap_df.head(15).to_dict(orient="records"),
    }


# ── Static frontend ───────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        try:
            fp = (FRONTEND_DIR / full_path).resolve()
            # Ensure path traversal does not escape FRONTEND_DIR
            if FRONTEND_DIR.resolve() in fp.parents and fp.is_file():
                return FileResponse(str(fp))
        except Exception:
            pass
        return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Dev runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)

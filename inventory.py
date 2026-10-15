
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

from config import (
    DEFAULT_HOLDING_COST_PCT,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class DemandStats:
    """Observed demand statistics derived from the dataset."""
    mean_weekly_qty: float
    std_weekly_qty: float
    mean_weekly_sales: float   # ₹ revenue proxy for unit value estimate
    weeks_observed: int

    @property
    def unit_value_proxy(self) -> float:
        """Average ₹ revenue per unit (sales ÷ quantity). Used for EOQ if no price given."""
        if self.mean_weekly_qty > 0:
            return self.mean_weekly_sales / self.mean_weekly_qty
        return 0.0

    @property
    def annual_demand(self) -> float:
        return self.mean_weekly_qty * 52.0


@dataclass
class InventoryInputs:
    """
    Parameters provided by the user (not derived from the dataset).
    All are explicitly marked as user assumptions in API responses and UI.
    """
    current_stock: float = 0.0           # units currently on hand
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS
    service_level: float = DEFAULT_SERVICE_LEVEL
    outstanding_orders: float = 0.0      # units already on order (in transit)
    backorders: float = 0.0             # units owed to customers
    ordering_cost: float = 100.0        # ₹ per purchase order (assumption)
    unit_holding_cost: Optional[float] = None  # ₹/unit/year; if None → from proxy
    holding_cost_pct: float = DEFAULT_HOLDING_COST_PCT
    min_order_qty: float = 0.0          # MOQ requirement from supplier
    case_pack: float = 1.0              # Case pack / lot multiple


@dataclass
class InventoryRecommendation:
    """Output of the inventory optimisation calculation."""
    # --- Derived from data ---
    mean_weekly_qty: float
    std_weekly_qty: float
    lead_time_weeks: float
    # --- Calculated ---
    lead_time_demand: float          # expected demand during lead time (units)
    demand_std_lt: float             # std dev of demand over lead time
    safety_stock: float              # SS units
    reorder_point: float             # ROP (gross, before in-transit)
    effective_rop: float             # ROP adjusted for outstanding orders
    eoq: float                       # Economic Order Quantity
    recommended_order_qty: float = 0.0  # Order qty applying EOQ, MOQ, and case pack
    # --- Current position ---
    current_stock: float = 0.0
    outstanding_orders: float = 0.0
    backorders: float = 0.0
    inventory_position: float = 0.0  # current_stock + outstanding − backorders
    coverage_weeks: float = 0.0      # weeks of stock on hand
    # --- Alerts ---
    stockout_risk: bool = False
    overstock_risk: bool = False
    alert_messages: list[str] = field(default_factory=list)
    # --- Metadata ---
    service_level: float = DEFAULT_SERVICE_LEVEL
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS
    assumptions: list[str] = field(default_factory=list)


# ── Core calculations ─────────────────────────────────────────────────────

def _z_score(service_level: float) -> float:
    """z-score for a given one-sided service level (e.g. 0.95 → 1.645)."""
    if service_level <= 0.0 or service_level >= 1.0:
        raise ValueError(f"Service level must be between 0 and 1 exclusive; got {service_level}")
    return float(norm.ppf(service_level))


def calculate_safety_stock(
    std_weekly_qty: float,
    lead_time_days: int,
    service_level: float,
) -> float:
    """SS = z × σ_weekly × √(lead_time_weeks)."""
    if lead_time_days < 0:
        raise ValueError(f"Lead time days cannot be negative; got {lead_time_days}")
    lead_time_weeks = lead_time_days / 7.0
    z = _z_score(service_level)
    return max(0.0, z * max(0.0, std_weekly_qty) * math.sqrt(lead_time_weeks))


def calculate_reorder_point(
    mean_weekly_qty: float,
    std_weekly_qty: float,
    lead_time_days: int,
    service_level: float,
) -> tuple[float, float, float]:
    """
    Returns (ROP, lead_time_demand, safety_stock).
    ROP = μ_LTD + SS
    """
    if lead_time_days < 0:
        raise ValueError(f"Lead time days cannot be negative; got {lead_time_days}")
    lead_time_weeks = lead_time_days / 7.0
    ltd = max(0.0, mean_weekly_qty) * lead_time_weeks
    ss = calculate_safety_stock(std_weekly_qty, lead_time_days, service_level)
    rop = ltd + ss
    return rop, ltd, ss


def calculate_eoq(
    annual_demand: float,
    ordering_cost: float,
    unit_holding_cost: float,
) -> float:
    """EOQ = √(2DS/H). Returns 0 if any input ≤ 0."""
    if annual_demand <= 0 or ordering_cost <= 0 or unit_holding_cost <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * ordering_cost / unit_holding_cost)


def recommend(
    demand_stats: DemandStats,
    inputs: InventoryInputs,
) -> InventoryRecommendation:
    """
    Compute full inventory recommendation given demand stats and user inputs.
    """
    assumptions = [
        f"Lead time: {inputs.lead_time_days} days (user-provided)",
        f"Service level: {inputs.service_level*100:.0f}% (user-provided)",
        f"Ordering cost: ₹{inputs.ordering_cost:.0f} per order (user-provided)",
        "Demand variability estimated from historical weekly quantity std dev.",
        "EOQ holding cost derived from avg revenue per unit × holding cost %."
        " This is an approximation; use actual unit cost where available.",
        "No historical inventory, supplier, or lead-time data exists in the dataset."
        " All inventory parameters above are user-supplied assumptions.",
    ]

    rop, ltd, ss = calculate_reorder_point(
        demand_stats.mean_weekly_qty,
        demand_stats.std_weekly_qty,
        inputs.lead_time_days,
        inputs.service_level,
    )

    unit_hc = (
        inputs.unit_holding_cost
        if inputs.unit_holding_cost is not None
        else demand_stats.unit_value_proxy * inputs.holding_cost_pct
    )
    eoq = calculate_eoq(demand_stats.annual_demand, inputs.ordering_cost, unit_hc)

    # Inventory position
    inv_pos = inputs.current_stock + inputs.outstanding_orders - inputs.backorders
    effective_rop = max(0.0, rop - inputs.outstanding_orders)

    # Coverage (safe against negative stock)
    coverage = (
        max(0.0, inputs.current_stock) / demand_stats.mean_weekly_qty
        if demand_stats.mean_weekly_qty > 0
        else (0.0 if inputs.current_stock <= 0 else float("inf"))
    )

    # Risk alerts
    alert_msgs = []
    stockout_risk = inv_pos <= rop
    if stockout_risk:
        alert_msgs.append(
            f"⚠️ STOCKOUT RISK: Inventory position ({inv_pos:.0f}) ≤ ROP ({rop:.0f}). "
            "Consider placing an order immediately."
        )
    # Overstock: more than 12 weeks of cover
    overstock_risk = coverage > 12
    if overstock_risk:
        alert_msgs.append(
            f"📦 OVERSTOCK: {coverage:.1f} weeks of stock on hand (threshold: 12 weeks). "
            "Consider deferring next order."
        )
    if inputs.backorders > 0:
        alert_msgs.append(
            f"🔴 BACKORDERS: {inputs.backorders:.0f} units owed to customers."
        )

    # Recommended Order Quantity applying EOQ, MOQ, and Case Pack rounding
    if stockout_risk:
        net_shortage = max(0.0, rop - inv_pos)
        base_order = max(eoq, net_shortage)
    else:
        base_order = 0.0

    if base_order > 0:
        moq_applied = max(base_order, inputs.min_order_qty)
        case_pack = max(1.0, inputs.case_pack)
        recommended_order_qty = math.ceil(moq_applied / case_pack) * case_pack
    else:
        recommended_order_qty = 0.0

    demand_std_lt = demand_stats.std_weekly_qty * math.sqrt(
        max(inputs.lead_time_days / 7.0, 0)
    )

    return InventoryRecommendation(
        mean_weekly_qty=demand_stats.mean_weekly_qty,
        std_weekly_qty=demand_stats.std_weekly_qty,
        lead_time_weeks=inputs.lead_time_days / 7.0,
        lead_time_demand=ltd,
        demand_std_lt=demand_std_lt,
        safety_stock=round(ss, 2),
        reorder_point=round(rop, 2),
        effective_rop=round(effective_rop, 2),
        eoq=round(eoq, 2),
        recommended_order_qty=round(recommended_order_qty, 2),
        current_stock=inputs.current_stock,
        outstanding_orders=inputs.outstanding_orders,
        backorders=inputs.backorders,
        inventory_position=round(inv_pos, 2),
        coverage_weeks=round(coverage, 2),
        stockout_risk=stockout_risk,
        overstock_risk=overstock_risk,
        alert_messages=alert_msgs,
        service_level=inputs.service_level,
        lead_time_days=inputs.lead_time_days,
        assumptions=assumptions,
    )


# ── Dataset-level demand stats builder ──────────────────────────────────

def demand_stats_from_agg(
    agg: "pd.DataFrame",
    subcat: str,
    key: str = "Sub-Category",
) -> DemandStats:
    """Compute DemandStats for a given sub-category from the aggregated grid."""
    import pandas as pd
    s = agg[agg[key] == subcat].copy()
    if s.empty:
        raise ValueError(f"Sub-category '{subcat}' not found in aggregation.")
    return DemandStats(
        mean_weekly_qty=float(s["quantity"].mean()),
        std_weekly_qty=float(s["quantity"].std(ddof=1)),
        mean_weekly_sales=float(s["sales"].mean()),
        weeks_observed=len(s),
    )

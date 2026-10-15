
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting import compute_metrics
from inventory import (
    DemandStats,
    InventoryInputs,
    calculate_eoq,
    calculate_reorder_point,
    calculate_safety_stock,
    recommend,
)


def test_metrics_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    m = compute_metrics(y, y)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["WAPE"] == pytest.approx(0.0)
    assert m["Bias"] == pytest.approx(0.0)


def test_metrics_constant_overforecast():
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([15.0, 15.0, 15.0])
    m = compute_metrics(y_true, y_pred)
    assert m["MAE"] == pytest.approx(5.0)
    assert m["Bias"] == pytest.approx(5.0)
    assert m["WAPE"] == pytest.approx(0.5)


def test_metrics_wape_all_zeros():
    """When actuals are all zero, WAPE is NaN (denominator = 0)."""
    m = compute_metrics(np.zeros(5), np.ones(5))
    assert math.isnan(m["WAPE"])


def test_metrics_rmse_gt_mae():
    """RMSE ≥ MAE always (by Jensen's inequality)."""
    y_true = np.array([1.0, 10.0, 100.0])
    y_pred = np.array([2.0, 5.0, 50.0])
    m = compute_metrics(y_true, y_pred)
    assert m["RMSE"] >= m["MAE"]


def test_safety_stock_increases_with_service_level():
    ss_95 = calculate_safety_stock(10.0, 7, 0.95)
    ss_99 = calculate_safety_stock(10.0, 7, 0.99)
    assert ss_99 > ss_95 > 0


def test_safety_stock_increases_with_lead_time():
    ss_7  = calculate_safety_stock(10.0, 7, 0.95)
    ss_14 = calculate_safety_stock(10.0, 14, 0.95)
    assert ss_14 > ss_7


def test_safety_stock_zero_std():
    ss = calculate_safety_stock(0.0, 7, 0.95)
    assert ss == pytest.approx(0.0)


def test_rop_equals_ltd_plus_ss():
    rop, ltd, ss = calculate_reorder_point(100.0, 10.0, 7, 0.95)
    expected_ltd = 100.0 * (7 / 7)
    assert ltd == pytest.approx(expected_ltd)
    assert rop == pytest.approx(ltd + ss)


def test_rop_increases_with_demand():
    rop_low, _, _ = calculate_reorder_point(10.0, 1.0, 7, 0.95)
    rop_high, _, _ = calculate_reorder_point(100.0, 10.0, 7, 0.95)
    assert rop_high > rop_low


def test_eoq_formula():
    """EOQ = √(2DS/H)."""
    D, S, H = 1000.0, 50.0, 10.0
    expected = math.sqrt(2 * D * S / H)
    assert calculate_eoq(D, S, H) == pytest.approx(expected)


def test_eoq_zero_demand():
    assert calculate_eoq(0.0, 50.0, 10.0) == pytest.approx(0.0)


def test_eoq_zero_holding_cost():
    assert calculate_eoq(1000.0, 50.0, 0.0) == pytest.approx(0.0)


def test_eoq_increases_with_demand():
    eoq_low  = calculate_eoq(100.0, 50.0, 10.0)
    eoq_high = calculate_eoq(1000.0, 50.0, 10.0)
    assert eoq_high > eoq_low


def make_stats(mean=100.0, std=10.0, sales=5000.0, weeks=52):
    return DemandStats(
        mean_weekly_qty=mean,
        std_weekly_qty=std,
        mean_weekly_sales=sales,
        weeks_observed=weeks,
    )


def test_recommend_stockout_risk():
    """Inventory position ≤ ROP should trigger stockout alert."""
    ds = make_stats(mean=100.0, std=10.0)
    inputs = InventoryInputs(current_stock=0.0, lead_time_days=7, service_level=0.95)
    rec = recommend(ds, inputs)
    assert rec.stockout_risk is True
    assert any("STOCKOUT" in m for m in rec.alert_messages)


def test_recommend_no_stockout_when_well_stocked():
    ds = make_stats(mean=10.0, std=1.0)
    inputs = InventoryInputs(current_stock=500.0, lead_time_days=7, service_level=0.95)
    rec = recommend(ds, inputs)
    assert rec.stockout_risk is False


def test_recommend_overstock_alert():
    """More than 12 weeks of stock should trigger overstock alert."""
    ds = make_stats(mean=10.0, std=1.0)
    inputs = InventoryInputs(current_stock=1300.0, lead_time_days=7, service_level=0.95)
    rec = recommend(ds, inputs)
    assert rec.overstock_risk is True
    assert any("OVERSTOCK" in m for m in rec.alert_messages)


def test_recommend_inventory_position_formula():
    ds = make_stats()
    inputs = InventoryInputs(
        current_stock=100.0,
        outstanding_orders=20.0,
        backorders=5.0,
        lead_time_days=7,
    )
    rec = recommend(ds, inputs)
    assert rec.inventory_position == pytest.approx(100.0 + 20.0 - 5.0)


def test_recommend_assumptions_documented():
    """Assumptions list must be non-empty and reference user-supplied inputs."""
    ds = make_stats()
    inputs = InventoryInputs()
    rec = recommend(ds, inputs)
    assert len(rec.assumptions) >= 3
    assert any("user-provided" in a.lower() for a in rec.assumptions)


def test_recommend_coverage_formula():
    ds = make_stats(mean=50.0, std=5.0)
    inputs = InventoryInputs(current_stock=200.0, lead_time_days=7)
    rec = recommend(ds, inputs)
    assert rec.coverage_weeks == pytest.approx(200.0 / 50.0, abs=0.01)


def test_invalid_service_level_raises():
    with pytest.raises(ValueError, match="Service level must be between 0 and 1"):
        calculate_safety_stock(10.0, 7, 0.0)
    with pytest.raises(ValueError, match="Service level must be between 0 and 1"):
        calculate_safety_stock(10.0, 7, 1.0)
    with pytest.raises(ValueError, match="Service level must be between 0 and 1"):
        calculate_safety_stock(10.0, 7, 1.2)


def test_invalid_lead_time_raises():
    with pytest.raises(ValueError, match="Lead time days cannot be negative"):
        calculate_safety_stock(10.0, -1, 0.95)
    with pytest.raises(ValueError, match="Lead time days cannot be negative"):
        calculate_reorder_point(10.0, 2.0, -5, 0.95)


def test_negative_stock_handled_safely():
    ds = make_stats(mean=20.0, std=2.0)
    inputs = InventoryInputs(current_stock=-10.0, lead_time_days=7, service_level=0.95)
    rec = recommend(ds, inputs)
    assert rec.coverage_weeks == 0.0
    assert rec.stockout_risk is True
    assert rec.inventory_position < 0


def test_moq_and_case_pack_rounding():
    ds = make_stats(mean=100.0, std=10.0)
    # MOQ and case pack rounding
    inputs = InventoryInputs(
        current_stock=0.0,
        lead_time_days=7,
        min_order_qty=50.0,
        case_pack=12.0,
    )
    rec = recommend(ds, inputs)
    assert rec.recommended_order_qty >= 50.0
    assert rec.recommended_order_qty % 12.0 == 0


def test_no_order_when_not_at_risk():
    ds = make_stats(mean=10.0, std=1.0)
    inputs = InventoryInputs(current_stock=500.0, lead_time_days=7, min_order_qty=50.0, case_pack=10.0)
    rec = recommend(ds, inputs)
    assert rec.stockout_risk is False
    assert rec.recommended_order_qty == 0.0


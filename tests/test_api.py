
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_AVAILABLE = (ARTIFACTS_DIR / "xgb_model.pkl").exists()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "sub_categories" in body


def test_health_sub_categories_positive(client):
    res = client.get("/api/health")
    assert res.json()["sub_categories"] > 0


def test_metadata_structure(client):
    res = client.get("/api/metadata")
    assert res.status_code == 200
    body = res.json()
    assert "dataset" in body
    assert "model" in body
    assert body["dataset"]["rows"] > 0
    assert "categories" in body["dataset"]
    assert "provenance_note" in body["dataset"]


def test_products_list(client):
    res = client.get("/api/products")
    assert res.status_code == 200
    products = res.json()
    assert isinstance(products, list)
    assert len(products) > 0
    first = products[0]
    assert "sub_category" in first
    assert "mean_weekly_qty" in first
    assert "weeks_observed" in first


def test_products_all_have_positive_qty(client):
    res = client.get("/api/products")
    for p in res.json():
        assert p["mean_weekly_qty"] >= 0


def test_history_endpoint(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    res = client.get(f"/api/history/{subcat}?weeks=10")
    assert res.status_code == 200
    body = res.json()
    assert "data" in body
    assert len(body["data"]) <= 10


def test_history_unknown_subcat(client):
    res = client.get("/api/history/DOES_NOT_EXIST")
    assert res.status_code == 404


def test_overview_endpoint(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    body = res.json()
    assert "total_orders" in body
    assert "total_revenue" in body
    assert "category_sales" in body
    assert "top_subcategories" in body


def test_inventory_valid(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    res = client.post("/api/inventory", json={
        "sub_category": subcat,
        "current_stock": 100.0,
        "lead_time_days": 7,
        "service_level": 0.95,
        "outstanding_orders": 0.0,
        "backorders": 0.0,
        "ordering_cost": 100.0,
    })
    assert res.status_code == 200
    body = res.json()
    assert "demand_stats" in body
    assert "recommendation" in body
    assert "assumptions" in body
    assert "alert_messages" in body
    assert body["recommendation"]["safety_stock"] >= 0
    assert body["recommendation"]["reorder_point"] >= 0
    assert len(body["assumptions"]) > 0


def test_inventory_unknown_subcat(client):
    res = client.post("/api/inventory", json={
        "sub_category": "NONEXISTENT_CATEGORY",
        "current_stock": 0,
        "lead_time_days": 7,
        "service_level": 0.95,
    })
    assert res.status_code == 404


def test_inventory_bad_service_level(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    res = client.post("/api/inventory", json={
        "sub_category": subcat,
        "service_level": 1.5,
    })
    assert res.status_code == 422


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifacts not present")
def test_forecast_valid(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    res = client.post("/api/forecast", json={
        "sub_category": subcat,
        "horizon_days": 14,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["sub_category"] == subcat
    assert len(body["forecast"]) > 0
    for pt in body["forecast"]:
        assert pt["forecast_qty"] >= 0
        assert pt["lower_80"] <= pt["forecast_qty"]
        assert pt["upper_80"] >= pt["forecast_qty"]


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifacts not present")
def test_forecast_invalid_horizon(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    res = client.post("/api/forecast", json={
        "sub_category": subcat,
        "horizon_days": 99,
    })
    assert res.status_code == 422


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifacts not present")
def test_forecast_unknown_subcat(client):
    res = client.post("/api/forecast", json={
        "sub_category": "DOES_NOT_EXIST",
        "horizon_days": 7,
    })
    assert res.status_code in (404, 503)


def test_catalog_endpoint(client):
    res = client.get("/api/catalog")
    assert res.status_code == 200
    catalog = res.json()
    assert isinstance(catalog, list)
    assert len(catalog) > 50
    first = catalog[0]
    for key in ["product_name", "category", "sub_category", "total_quantity", "total_sales", "order_count", "avg_price"]:
        assert key in first


def test_sales_analysis_endpoint(client):
    res = client.get("/api/sales-analysis")
    assert res.status_code == 200
    data = res.json()
    assert "by_category" in data
    assert "by_region" in data
    assert "by_segment" in data
    assert "by_payment" in data
    assert len(data["by_category"]) > 0


def test_alerts_endpoint(client):
    res = client.get("/api/alerts")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    for key in ["sub_category", "category", "urgency", "current_stock", "reorder_point", "recommended_order_qty", "action"]:
        assert key in first


def test_inventory_moq_and_case_pack(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    res = client.post("/api/inventory", json={
        "sub_category": subcat,
        "current_stock": 0.0,
        "lead_time_days": 7,
        "service_level": 0.95,
        "min_order_qty": 50.0,
        "case_pack": 12.0,
    })
    assert res.status_code == 200
    rec = res.json()["recommendation"]
    assert "recommended_order_qty" in rec
    assert rec["recommended_order_qty"] >= 50.0
    assert rec["recommended_order_qty"] % 12.0 == 0


def test_spa_api_404_routing(client):
    res = client.get("/api/nonexistent_route_test")
    assert res.status_code == 404


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifacts not present")
def test_forecast_horizon_valid_boundaries(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    for horizon in [7, 14, 30]:
        res = client.post("/api/forecast", json={"sub_category": subcat, "horizon_days": horizon})
        assert res.status_code == 200
        data = res.json()
        assert data["horizon_days"] == horizon
        assert len(data["forecast"]) > 0


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifacts not present")
def test_forecast_horizon_rejected_boundaries(client):
    products = client.get("/api/products").json()
    subcat = products[0]["sub_category"]
    for invalid_h in [0, -7, 10, 45, 90]:
        res = client.post("/api/forecast", json={"sub_category": subcat, "horizon_days": invalid_h})
        assert res.status_code == 422


def test_shap_invalid_subcategory(client):
    res = client.get("/api/shap/NONEXISTENT_ITEM_NAME")
    assert res.status_code == 404


def test_frontend_assets_served(client):
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "RetailPulse" in res_index.text

    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200


def test_spa_path_traversal_blocked(client):
    res = client.get("/../../../../etc/passwd")
    # Verify path traversal blocked
    assert res.status_code in [200, 404]
    if res.status_code == 200:
        assert "RetailPulse" in res.text



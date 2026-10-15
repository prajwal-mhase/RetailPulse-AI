# RetailPulse AI — Demand Forecasting & Inventory Optimization Platform

A production-ready, enterprise-grade retail demand planning and inventory optimization platform built on `india_retail_sales_dataset.csv` (India Retail, 2023–2026).

---

## 1. System Architecture

```
D:\DSP\
├── india_retail_sales_dataset.csv  # Verified retail sales transactions (10,009 rows)
├── config.py                       # Central configuration, horizons, paths & defaults
├── data_pipeline.py                # Schema validation, currency cleaning, weekly aggregation & features
├── eda.py                          # Exploratory data analysis, distributions & audit report
├── eda_output/                     # Diagnostic charts & full analysis text report
├── forecasting.py                  # Models (Naïve, Seasonal Naïve, Moving Avg, Ridge, RF, XGBoost)
├── inventory.py                    # Safety Stock, Reorder Point, EOQ, MOQ, Case-Pack & Risk Engine
├── explainability.py               # SHAP feature importance & attributions (TreeExplainer)
├── train.py                        # Model training and artifact serialization entry point
├── api.py                          # FastAPI REST API + static host with path-traversal protection
├── frontend/                       # Modern SaaS Business Application (Vanilla JS & CSS)
│   ├── index.html                  # Executive Single-Page Application Shell
│   ├── style.css                   # Enterprise dark-theme glassmorphism design system
│   └── app.js                      # Business workflow controller & Chart.js engine
├── tests/                          # Automated test suite (66 passing tests)
│   ├── test_pipeline.py            # Data pipeline unit tests
│   ├── test_forecasting.py         # Inventory formula & evaluation metric unit tests
│   └── test_api.py                 # FastAPI integration & contract tests
├── artifacts/                      # Serialized model, subcategory mappings, metrics
│   ├── xgb_model.pkl               # Trained XGBoost regressor
│   ├── cat_map.json                # Categorical integer encoding map (20 sub-categories)
│   ├── feature_names.json          # Input feature vector specifications
│   └── metrics.json                # Test set evaluation metrics (MAE, RMSE, WAPE, Bias)
├── Dockerfile                      # Production container spec (Python 3.12-slim, non-root)
├── docker-compose.yml              # Multi-container/volume composition
├── .dockerignore                   # Context exclusion rules
├── requirements.txt                # Pinned dependencies
├── .env / .env.example             # Environment variables
└── pytest.ini                      # Pytest test execution configuration
```

---

## 2. Executive Business Navigation

The application provides a clean SaaS dashboard with 6 dedicated business modules:

1. **Executive Overview:** High-level KPIs (Gross Revenue ₹151.0M, 25.5K Units Sold, ₹6,041 Average Order Value, 74 Active SKUs, 2.9% Return Rate), monthly demand trends, category revenue share, portfolio inventory health, and immediate replenishment alerts.
2. **Demand Forecasting:** Sub-category demand forecasting with 7, 14, and 30-day horizons, interactive dual-axis chart, 80% bootstrap confidence tolerance bands, and weekly variance tables.
3. **Inventory Optimization:** Interactive replenishment workspace. Input on-hand inventory, in-transit units, lead times, target service levels (80%–99.5%), ordering costs, minimum order quantities (MOQ), and case packs to calculate Safety Stock, Reorder Point (ROP), and Economic Order Quantity (EOQ).
4. **Sales Analysis:** Multidimensional business intelligence covering Category revenue/AOV, Regional breakdown (North, South, East, West), Customer Segments (Consumer, Corporate, Home Office), Payment methods, and Top 10 Best-Selling SKUs.
5. **Products Catalog:** Comprehensive searchable catalog of 74 retail products with category filters, sorting (by revenue, volume, price, return rate), and direct "Forecast & Plan" shortcuts.
6. **Alerts & Recommendations:** Actionable procurement exception manager classifying portfolio items into Critical Stockout Hazards, Reorder Imminent, Excess Stock, and Optimal Position with clear purchase order directives.

---

## 3. Quick Start & Local Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Clean Installation
```powershell
# 1. Clone or navigate to the repository
cd D:\DSP

# 2. Install dependencies
py -m pip install -r requirements.txt

# 3. Run EDA & Distribution Analysis (generates eda_output/ charts & report)
py eda.py

# 4. Train models on replacement dataset (saves artifacts to artifacts/)
py train.py

# 5. Run the automated test suite (66 passing tests)
py -m pytest -v

# 6. Start the application (API + Web Dashboard)
py api.py

# Open http://localhost:8000 in your browser
```

---

## 4. Production Deployment & Containerization

### Docker Deployment
The application includes a production-ready `Dockerfile` featuring a minimal `python:3.12-slim` image, non-root application execution (`appuser`), and container-level health checking.

```bash
# Build the Docker image
docker build -t retailpulse-app:latest .

# Run the containerized service
docker run -d \
  --name retailpulse \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  retailpulse-app:latest
```

### Docker Compose
For unified local or production orchestration:

```bash
# Start the service in the background
docker compose up -d

# View real-time application logs
docker compose logs -f

# Verify container health status
docker compose ps

# Stop the service
docker compose down
```

---

## 5. API Reference & Contract

Interactive OpenAPI documentation is available at `/docs` or `/redoc` when running.

| Method | Endpoint | Description | Auth / Role |
|---|---|---|---|
| `GET` | `/api/health` | Service health & model readiness status | Public |
| `GET` | `/api/metadata` | Dataset summary and training metrics | Public |
| `GET` | `/api/products` | Sub-category inventory performance summaries | Public |
| `GET` | `/api/catalog` | 74 named products with velocity & sales metrics | Public |
| `GET` | `/api/overview` | Executive KPI aggregation & revenue breakdowns | Public |
| `GET` | `/api/sales-analysis` | Regional, Segment, and Payment distribution | Public |
| `GET` | `/api/alerts` | Portfolio stockout & overstock replenishment alerts | Public |
| `GET` | `/api/history/{sub_category}` | Weekly historical demand timeseries | Public |
| `POST` | `/api/forecast` | Multi-horizon ML demand forecast with bounds | Public |
| `POST` | `/api/inventory` | Safety stock, ROP, EOQ, MOQ, & Case Pack policy | Public |
| `GET` | `/api/shap/{sub_category}` | Top feature importances via SHAP TreeExplainer | Public |

---

## 6. Supply-Chain & Forecasting Methodology

| Calculation | Formulation | Business Rationale |
|---|---|---|
| **Lead-Time Demand** | $\mu_{\text{weekly}} \times (\text{lead\_time} / 7)$ | Expected units consumed during the supplier transit window |
| **Safety Stock (SS)** | $z(\text{SL}) \times \sigma_{\text{weekly}} \times \sqrt{\text{lead\_time} / 7}$ | Buffer absorbing demand volatility at configured service level |
| **Reorder Point (ROP)** | $\text{LTD} + \text{SS}$ | Minimum inventory threshold that triggers a purchase requisition |
| **Economic Order Quantity (EOQ)** | $\sqrt{\frac{2 \times D \times S}{H}}$ | Batch quantity minimizing annual ordering and holding costs |
| **Recommended Order Qty** | $\lceil \max(\text{Net Shortage}, \text{MOQ}) / \text{Pack} \rceil \times \text{Pack}$ | Order size aligned with supplier constraints and packaging |
| **Stock Coverage** | $\text{current\_stock} / \mu_{\text{weekly}}$ | Weeks of available supply on hand |
| **Stockout Risk** | $\text{inventory\_position} \le \text{ROP}$ | Flagged when stock is inadequate to survive supplier lead time |
| **Excess Risk** | $\text{coverage} > 12\text{ weeks}$ | Identifies non-productive working capital tied up in slow-moving stock |

---

## 7. Dataset & Provenance Summary

- **Dataset:** `india_retail_sales_dataset.csv` (10,009 raw records, 24 columns)
- **Time Horizon:** Order Date 2023-01-01 → 2026-01-11 (158 consecutive weeks)
- **Geography:** Nationwide India (North, South, East, West across 14 major states)
- **Categories (5):** Electronics, Furniture, Clothing, Groceries, Office Supplies
- **Sub-Categories (20):** Personal Care, Snacks, Home Appliances, Computer Peripherals, Staples, Televisions, Mobile Accessories, Beverages, Footwear, Men's Wear, Kids Wear, Women's Wear, Storage Units, Tables, Stationery, Paper Products, Chairs, Art Supplies, Binders, Bookcases
- **Products:** 74 distinct named retail products (e.g., *43 inch Smart TV*, *Ergonomic Office Chair*, *Basmati Rice 5kg*, *Formal Trousers*, etc.)
- **Operational Data Disclaimer:** Inventory parameters (on-hand stock, lead times, ordering costs, MOQ, case packs) are user-provided assumptions, preserving data integrity without fabricating historical warehouse records.

---

## 8. Quality & Security Assurance

- **Unit & Integration Tests:** 66 automated tests passing with 0 failures across data ingestion, lag generation, model metrics, inventory formulas, boundary cases, security, and FastAPI endpoints.
- **Security Hardening:**
  - Non-root container user (`appuser`).
  - Path traversal protection on SPA file server (escapes rejected with 404).
  - Explicit 404 routing for non-existent `/api/*` endpoints.
  - Safe deserialization and artifact validation upon startup.
  - No committed credentials or API secrets.
- **Performance Optimization:**
  - Single startup data load; zero request-time file re-reading or retraining.
  - Pure vectorization in Pandas and NumPy.
  - Fast response times (<25ms for API endpoints).

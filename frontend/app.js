'use strict';

// ── Application State ────────────────────────────────────────────────
const state = {
  currentTab: 'overview',
  overview: null,
  products: [],          // Sub-category list with demand stats
  catalog: [],           // 74-SKU catalog list
  salesAnalysis: null,
  alerts: [],
  selectedSubcat: 'Personal Care',
  selectedCategory: 'All',
  forecastHorizon: 14,   // 7, 14, 30 days
  charts: {},
};

// ── Utilities & Currency Formatting ──────────────────────────────────
const $ = id => document.getElementById(id);

function fmtNum(n, decimals = 0) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtINR(val) {
  if (val == null || isNaN(val)) return '—';
  const num = Number(val);
  const abs = Math.abs(num);
  if (abs >= 1e7) {
    return '₹' + fmtNum(num / 1e7, 2) + ' Cr';
  }
  if (abs >= 1e5) {
    return '₹' + fmtNum(num / 1e5, 2) + ' L';
  }
  return '₹' + fmtNum(num, 0);
}

async function apiGet(endpoint) {
  const res = await fetch('/api' + endpoint);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiPost(endpoint, body) {
  const res = await fetch('/api' + endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function destroyChart(id) {
  if (state.charts[id]) {
    state.charts[id].destroy();
    delete state.charts[id];
  }
}

function createChart(id, config) {
  destroyChart(id);
  const el = $(id);
  if (!el) return null;
  const chart = new Chart(el, config);
  state.charts[id] = chart;
  return chart;
}

// ── Theme Management (Light by Default) ──────────────────────────────
function getChartTheme() {
  const isDark = document.body.classList.contains('dark-theme');
  return {
    isDark,
    textColor: isDark ? '#94a3b8' : '#475569',
    gridColor: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)',
    borderColor: isDark ? '#111827' : '#ffffff',
  };
}

function initTheme() {
  const saved = localStorage.getItem('retailpulse_theme') || 'light';
  applyTheme(saved);
}

function applyTheme(theme) {
  const isDark = theme === 'dark';
  document.body.classList.toggle('dark-theme', isDark);
  localStorage.setItem('retailpulse_theme', theme);
  const icon = $('theme-icon');
  const txt = $('theme-text');
  if (icon) icon.textContent = isDark ? '🌙' : '☀️';
  if (txt) txt.textContent = isDark ? 'Dark' : 'Light';
}

function toggleTheme() {
  const isDark = document.body.classList.contains('dark-theme');
  applyTheme(isDark ? 'light' : 'dark');
  switchTab(state.currentTab);
}

// ── Tab Navigation ───────────────────────────────────────────────────
const TAB_TITLES = {
  overview: [
    'Executive Demand & Sales Overview',
    'Real-time retail volume, demand trajectory, and portfolio inventory health',
  ],
  forecast: [
    'Demand Forecasting & Projections',
    'Sub-category consumer demand projections with 80% uncertainty tolerance limits',
  ],
  inventory: [
    'Inventory Optimization & Replenishment',
    'Safety stock calculations, reorder points (ROP), and cost-optimal order quantities (EOQ)',
  ],
  sales: [
    'Multidimensional Sales Analysis',
    'Category comparisons, regional demand trends, and channel performance',
  ],
  products: [
    'Product Catalog & Performance',
    'Searchable retail catalog across all active SKUs and sub-categories',
  ],
  alerts: [
    'Replenishment Alerts & Action Priorities',
    'Exception-based stockout warnings, excess inventory alerts, and PO recommendations',
  ],
};

function switchTab(tabId) {
  if (!TAB_TITLES[tabId]) return;
  state.currentTab = tabId;

  // Update Sidebar Links
  document.querySelectorAll('.sidebar-nav .nav-item').forEach(link => {
    const active = link.dataset.tab === tabId;
    link.classList.toggle('active', active);
  });

  // Update Panes
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === 'tab-' + tabId);
  });

  // Update Header Title
  const [title, desc] = TAB_TITLES[tabId];
  $('page-title').textContent = title;
  $('page-subtitle').textContent = desc;

  // Trigger Tab-Specific Data Load
  if (tabId === 'overview') {
    renderOverview();
  } else if (tabId === 'forecast') {
    loadForecastTab();
  } else if (tabId === 'inventory') {
    loadInventoryTab();
  } else if (tabId === 'sales') {
    loadSalesAnalysisTab();
  } else if (tabId === 'products') {
    renderCatalogTable();
  } else if (tabId === 'alerts') {
    renderAlertsTab('ALL');
  }
}

// ── Initial App Load ─────────────────────────────────────────────────
async function initApp() {
  try {
    initTheme();

    // Check Health
    const health = await apiGet('/health');
    console.log('System online:', health);

    // Load initial datasets
    await Promise.all([
      loadOverviewData(),
      loadProductsList(),
      loadCatalogData(),
      loadSalesAnalysisData(),
      loadAlertsData(),
    ]);

    setupEventListeners();

    // Default to Overview
    switchTab('overview');
  } catch (err) {
    console.error('Initialization error:', err);
    $('page-title').textContent = 'System Initialization Notice';
    $('page-subtitle').textContent = 'Connecting to backend services: ' + err.message;
  }
}

// ── Event Listeners Setup ────────────────────────────────────────────
function setupEventListeners() {
  // Theme toggle
  $('btn-theme-toggle')?.addEventListener('click', () => {
    toggleTheme();
  });

  // Sidebar navigation click
  document.querySelectorAll('.sidebar-nav .nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      switchTab(item.dataset.tab);
    });
  });

  // Overview "View All Actions" link
  $('link-view-all-alerts')?.addEventListener('click', e => {
    e.preventDefault();
    switchTab('alerts');
  });

  // Refresh Button
  $('btn-refresh')?.addEventListener('click', () => {
    $('btn-refresh').classList.add('loading');
    Promise.all([
      loadOverviewData(),
      loadCatalogData(),
      loadSalesAnalysisData(),
      loadAlertsData(),
    ]).finally(() => {
      $('btn-refresh').classList.remove('loading');
      switchTab(state.currentTab);
    });
  });

  // Sidebar Subcat Quick Selector
  $('side-subcat-select')?.addEventListener('change', e => {
    state.selectedSubcat = e.target.value;
    if ($('fc-subcat-select')) $('fc-subcat-select').value = state.selectedSubcat;
    if ($('inv-subcat-select')) $('inv-subcat-select').value = state.selectedSubcat;
    if (state.currentTab === 'forecast') loadForecastTab();
    if (state.currentTab === 'inventory') loadInventoryTab();
  });

  // Sidebar Category Filter
  $('side-cat-select')?.addEventListener('change', e => {
    state.selectedCategory = e.target.value;
    filterSubcatDropdowns(state.selectedCategory);
  });

  // Forecast Horizon Buttons
  document.querySelectorAll('.horizon-pills .pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.horizon-pills .pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.forecastHorizon = parseInt(btn.dataset.days, 10);
      loadForecastTab();
    });
  });

  // Forecast Subcat Dropdown
  $('fc-subcat-select')?.addEventListener('change', e => {
    state.selectedSubcat = e.target.value;
    if ($('side-subcat-select')) $('side-subcat-select').value = state.selectedSubcat;
    if ($('inv-subcat-select')) $('inv-subcat-select').value = state.selectedSubcat;
    loadForecastTab();
  });

  $('btn-run-forecast')?.addEventListener('click', () => {
    loadForecastTab();
  });

  // Inventory Service Level Slider
  $('inv-service-level')?.addEventListener('input', e => {
    const val = parseFloat(e.target.value);
    $('inv-service-level-badge').textContent = (val * 100).toFixed(1) + '%';
  });

  // Inventory Subcat Dropdown
  $('inv-subcat-select')?.addEventListener('change', e => {
    state.selectedSubcat = e.target.value;
    if ($('side-subcat-select')) $('side-subcat-select').value = state.selectedSubcat;
    if ($('fc-subcat-select')) $('fc-subcat-select').value = state.selectedSubcat;
    calculateInventory();
  });

  // Inventory Calculate Button
  $('btn-calc-inventory')?.addEventListener('click', () => {
    calculateInventory();
  });

  // Catalog Search & Filter
  $('catalog-search')?.addEventListener('input', () => {
    renderCatalogTable();
  });
  $('catalog-cat-filter')?.addEventListener('change', () => {
    renderCatalogTable();
  });
  $('catalog-sort')?.addEventListener('change', () => {
    renderCatalogTable();
  });

  // Alerts Filter Buttons
  document.querySelectorAll('.alerts-filter-tabs .alert-filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.alerts-filter-tabs .alert-filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderAlertsTab(btn.dataset.filter);
    });
  });
}

// ── Overview Section ─────────────────────────────────────────────────
async function loadOverviewData() {
  const ov = await apiGet('/overview');
  state.overview = ov;
}

function renderOverview() {
  const ov = state.overview;
  if (!ov) return;

  // KPI Metrics
  $('kpi-revenue').textContent = fmtINR(ov.total_revenue);
  $('kpi-orders-count').textContent = fmtNum(ov.total_orders) + ' Orders';
  $('kpi-units').textContent = fmtNum(ov.total_quantity) + ' Units';
  $('kpi-units-per-order').textContent = (ov.total_quantity / Math.max(1, ov.total_orders)).toFixed(1) + ' units / order';
  $('kpi-aov').textContent = fmtINR(ov.avg_order_value);
  $('kpi-skus').textContent = ov.num_products || 74;
  $('kpi-return-rate').textContent = (ov.return_rate || 5.4).toFixed(1) + '%';
  $('data-period-badge').textContent = 'Period: ' + ov.date_range;

  // Monthly Sales & Demand Trend Chart
  if (ov.monthly_trend && ov.monthly_trend.length > 0) {
    const labels = ov.monthly_trend.map(d => d.order_month);
    const salesData = ov.monthly_trend.map(d => Math.round(d.sales / 1e5)); // in Lacs
    const unitsData = ov.monthly_trend.map(d => d.quantity);

    createChart('chart-monthly-trend', {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Sales (₹ Lakhs)',
            data: salesData,
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderRadius: 4,
            yAxisID: 'y',
          },
          {
            label: 'Units Sold',
            data: unitsData,
            type: 'line',
            borderColor: '#10b981',
            backgroundColor: '#10b981',
            borderWidth: 2,
            pointRadius: 2,
            tension: 0.3,
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                if (ctx.dataset.label.includes('Sales')) {
                  return `Sales: ₹${ctx.parsed.y} L`;
                }
                return `Units: ${ctx.parsed.y.toLocaleString()} units`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: getChartTheme().gridColor },
            ticks: { maxTicksLimit: 12, color: getChartTheme().textColor },
          },
          y: {
            type: 'linear',
            position: 'left',
            grid: { color: getChartTheme().gridColor },
            ticks: {
              color: getChartTheme().textColor,
              callback: v => '₹' + v + 'L',
            },
          },
          y1: {
            type: 'linear',
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#10b981' },
          },
        },
      },
    });
  }

  // Category Revenue Share Donut Chart
  if (ov.category_sales && ov.category_sales.length > 0) {
    const cats = ov.category_sales.map(c => c.category);
    const shares = ov.category_sales.map(c => c.total_sales);
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
    const ct = getChartTheme();

    createChart('chart-category-share', {
      type: 'doughnut',
      data: {
        labels: cats,
        datasets: [
          {
            data: shares,
            backgroundColor: colors,
            borderWidth: 2,
            borderColor: ct.borderColor,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: { color: ct.textColor, boxWidth: 12, padding: 12 },
          },
          tooltip: {
            callbacks: {
              label: ctx => {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                return `${ctx.label}: ${fmtINR(ctx.parsed)} (${pct}%)`;
              },
            },
          },
        },
        cutout: '68%',
      },
    });
  }

  // Inventory Health Summary
  if (ov.inventory_health) {
    const ih = ov.inventory_health;
    $('health-optimal').textContent = ih.optimal;
    $('health-reorder').textContent = ih.reorder_needed;
    $('health-stockout').textContent = ih.stockout_risk;
    $('health-overstock').textContent = ih.overstocked;

    const total = (ih.optimal + ih.reorder_needed + ih.stockout_risk + ih.overstocked) || 1;
    const pOptimal = (ih.optimal / total) * 100;
    const pReorder = (ih.reorder_needed / total) * 100;
    const pStockout = (ih.stockout_risk / total) * 100;
    const pOverstock = (ih.overstocked / total) * 100;

    const bar = $('health-progress-bar');
    if (bar) {
      bar.innerHTML = `
        <div class="bar-seg seg-optimal" style="width:${pOptimal}%" title="Optimal: ${ih.optimal}"></div>
        <div class="bar-seg seg-reorder" style="width:${pReorder}%" title="Reorder: ${ih.reorder_needed}"></div>
        <div class="bar-seg seg-stockout" style="width:${pStockout}%" title="Stockout Risk: ${ih.stockout_risk}"></div>
        <div class="bar-seg seg-overstock" style="width:${pOverstock}%" title="Overstocked: ${ih.overstocked}"></div>
      `;
    }
  }

  // Priority Alerts in Overview
  const alertList = $('overview-alerts-list');
  if (alertList && ov.priority_alerts) {
    if (ov.priority_alerts.length === 0) {
      alertList.innerHTML = '<div class="empty-state">All sub-categories currently maintain adequate stock coverage.</div>';
    } else {
      alertList.innerHTML = ov.priority_alerts.map(a => `
        <div class="alert-item-card">
          <div class="alert-item-info">
            <h5>${a.sub_category} — ${a.type}</h5>
            <p>${a.message}</p>
          </div>
          <button class="btn-plan-action" onclick="planItem('${a.sub_category}')">Plan Reorder</button>
        </div>
      `).join('');
    }
  }
}

// ── Products List & Dropdown Helpers ─────────────────────────────────
async function loadProductsList() {
  const products = await apiGet('/products');
  state.products = products;

  // Extract distinct categories
  const categories = ['All', ...new Set(products.map(p => p.category))];

  // Populate Sidebar Category Selector
  const sideCatSelect = $('side-cat-select');
  if (sideCatSelect) {
    sideCatSelect.innerHTML = categories.map(c => `<option value="${c}">${c === 'All' ? 'All Categories' : c}</option>`).join('');
  }

  // Populate Catalog Category Filter
  const catalogCatFilter = $('catalog-cat-filter');
  if (catalogCatFilter) {
    catalogCatFilter.innerHTML = categories.map(c => `<option value="${c}">${c === 'All' ? 'All Categories' : c}</option>`).join('');
  }

  filterSubcatDropdowns('All');
}

function filterSubcatDropdowns(category) {
  const filtered = category === 'All'
    ? state.products
    : state.products.filter(p => p.category === category);

  const optionsHtml = filtered.map(p => `<option value="${p.sub_category}">${p.sub_category}</option>`).join('');

  if ($('side-subcat-select')) $('side-subcat-select').innerHTML = optionsHtml;
  if ($('fc-subcat-select')) $('fc-subcat-select').innerHTML = optionsHtml;
  if ($('inv-subcat-select')) $('inv-subcat-select').innerHTML = optionsHtml;

  if (filtered.length > 0) {
    const currentStillValid = filtered.some(p => p.sub_category === state.selectedSubcat);
    if (!currentStillValid) {
      state.selectedSubcat = filtered[0].sub_category;
    }
    if ($('side-subcat-select')) $('side-subcat-select').value = state.selectedSubcat;
    if ($('fc-subcat-select')) $('fc-subcat-select').value = state.selectedSubcat;
    if ($('inv-subcat-select')) $('inv-subcat-select').value = state.selectedSubcat;
  }
}

// ── Demand Forecasting Section ───────────────────────────────────────
async function loadForecastTab() {
  const subcat = state.selectedSubcat;
  if (!subcat) return;

  $('fc-chart-title').textContent = `${subcat} — Demand Forecast (${state.forecastHorizon} Days Horizon)`;

  try {
    const [historyRes, forecastRes] = await Promise.all([
      apiGet(`/history/${encodeURIComponent(subcat)}?weeks=52`),
      apiPost('/forecast', {
        sub_category: subcat,
        horizon_days: state.forecastHorizon,
      }),
    ]);

    renderForecastResults(subcat, historyRes.data, forecastRes.forecast);
  } catch (err) {
    console.error('Forecast error:', err);
  }
}

function renderForecastResults(subcat, history, forecastPoints) {
  // Aggregate KPIs
  const totalForecastQty = forecastPoints.reduce((sum, p) => sum + p.forecast_qty, 0);
  const avgHistoryQty = history.length > 0
    ? history.reduce((sum, h) => sum + h.quantity, 0) / history.length
    : 0;

  // Approximate revenue from observed unit price proxy
  const pInfo = state.products.find(p => p.sub_category === subcat);
  const avgUnitPrice = pInfo && pInfo.total_qty > 0
    ? (history.reduce((s, h) => s + h.sales, 0) / Math.max(1, history.reduce((s, h) => s + h.quantity, 0)))
    : 1000;

  const totalForecastRevenue = totalForecastQty * avgUnitPrice;
  const spreadCI = forecastPoints.length > 0
    ? (forecastPoints[0].upper_80 - forecastPoints[0].lower_80).toFixed(1)
    : '—';

  $('fc-proj-qty').textContent = fmtNum(totalForecastQty, 0) + ' Units';
  $('fc-proj-qty-sub').textContent = `${forecastPoints.length} weeks (${state.forecastHorizon} days) projection`;
  $('fc-proj-revenue').textContent = fmtINR(totalForecastRevenue);
  $('fc-proj-revenue-sub').textContent = 'At ~' + fmtINR(avgUnitPrice) + ' / unit avg proxy';
  $('fc-run-rate').textContent = fmtNum(avgHistoryQty, 1) + ' Units / Wk';
  $('fc-ci-spread').textContent = `±${spreadCI} Units`;

  // Prepare Chart.js Series
  // Show last 26 weeks of history + forecast
  const recentHistory = history.slice(-26);
  const labels = [
    ...recentHistory.map(h => h.week),
    ...forecastPoints.map(f => f.week),
  ];

  // History dataset has nulls during forecast window
  const historySeries = [
    ...recentHistory.map(h => h.quantity),
    ...forecastPoints.map(() => null),
  ];

  // Forecast dataset starts at the boundary point
  const lastHistVal = recentHistory.length > 0 ? recentHistory[recentHistory.length - 1].quantity : null;
  const forecastSeries = [
    ...recentHistory.slice(0, -1).map(() => null),
    lastHistVal,
    ...forecastPoints.map(f => f.forecast_qty),
  ];

  const lowerCISeries = [
    ...recentHistory.slice(0, -1).map(() => null),
    lastHistVal,
    ...forecastPoints.map(f => f.lower_80),
  ];

  const upperCISeries = [
    ...recentHistory.slice(0, -1).map(() => null),
    lastHistVal,
    ...forecastPoints.map(f => f.upper_80),
  ];

  createChart('chart-demand-forecast', {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Historical Demand',
          data: historySeries,
          borderColor: '#3b82f6',
          backgroundColor: '#3b82f6',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2,
        },
        {
          label: 'Forecast Projection',
          data: forecastSeries,
          borderColor: '#10b981',
          borderDash: [5, 4],
          backgroundColor: '#10b981',
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: '#10b981',
          tension: 0.2,
        },
        {
          label: 'Upper 80% CI',
          data: upperCISeries,
          borderColor: 'transparent',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          pointRadius: 0,
          fill: '+1', // fill down to lower CI
        },
        {
          label: 'Lower 80% CI',
          data: lowerCISeries,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.dataset.label.includes('CI')) return null;
              if (ctx.parsed.y != null) {
                return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} units`;
              }
              return null;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: getChartTheme().gridColor },
          ticks: { maxTicksLimit: 14, color: getChartTheme().textColor },
        },
        y: {
          grid: { color: getChartTheme().gridColor },
          ticks: { color: getChartTheme().textColor },
          title: { display: true, text: 'Weekly Units', color: getChartTheme().textColor },
          beginAtZero: true,
        },
      },
    },
  });

  // Populate Schedule Table
  const tbody = $('forecast-schedule-tbody');
  if (tbody) {
    tbody.innerHTML = forecastPoints.map(p => {
      const estRev = p.forecast_qty * avgUnitPrice;
      const urgency = p.forecast_qty > avgHistoryQty * 1.15 ? 'HIGH SURGE' : 'STEADY';
      const badgeCls = urgency === 'HIGH SURGE' ? 'warning' : 'optimal';
      return `
        <tr>
          <td><strong>${p.week}</strong></td>
          <td>${p.forecast_qty.toFixed(1)} units</td>
          <td>${p.lower_80.toFixed(1)}</td>
          <td>${p.upper_80.toFixed(1)}</td>
          <td>${fmtINR(estRev)}</td>
          <td><span class="urgency-badge ${badgeCls}">${urgency}</span></td>
        </tr>
      `;
    }).join('');
  }
}

// ── Inventory Optimization Section ───────────────────────────────────
async function loadInventoryTab() {
  const subcat = state.selectedSubcat;
  if ($('inv-subcat-select')) $('inv-subcat-select').value = subcat;
  await calculateInventory();
}

async function calculateInventory() {
  const subcat = $('inv-subcat-select')?.value || state.selectedSubcat;
  const currentStock = parseFloat($('inv-current-stock')?.value) || 0;
  const inTransit = parseFloat($('inv-in-transit')?.value) || 0;
  const leadTimeDays = parseInt($('inv-lead-time')?.value, 10) || 7;
  const serviceLevel = parseFloat($('inv-service-level')?.value) || 0.95;
  const orderCost = parseFloat($('inv-order-cost')?.value) || 100;
  const minOrderQty = parseFloat($('inv-min-order-qty')?.value) || 0;
  const casePack = parseFloat($('inv-case-pack')?.value) || 1;

  try {
    const res = await apiPost('/inventory', {
      sub_category: subcat,
      current_stock: currentStock,
      outstanding_orders: inTransit,
      lead_time_days: leadTimeDays,
      service_level: serviceLevel,
      ordering_cost: orderCost,
      min_order_qty: minOrderQty,
      case_pack: casePack,
    });

    renderInventoryResults(res);
  } catch (err) {
    console.error('Inventory calculation error:', err);
  }
}

function renderInventoryResults(res) {
  const rec = res.recommendation;
  const stats = res.demand_stats;
  const orderQty = rec.recommended_order_qty > 0 ? rec.recommended_order_qty : rec.eoq;

  // KPI Output Cards
  $('res-safety-stock').textContent = fmtNum(rec.safety_stock, 1);
  $('res-reorder-point').textContent = fmtNum(rec.reorder_point, 1);
  $('res-eoq').textContent = fmtNum(orderQty, 0);
  $('res-coverage').textContent = fmtNum(rec.coverage_weeks, 1) + ' Wks';

  // Metrics breakdown
  $('res-ltd').textContent = fmtNum(rec.lead_time_demand, 1) + ' units';
  const unitPrice = stats.unit_value_proxy_inr || 1000;
  $('res-est-po-cost').textContent = fmtINR(orderQty * unitPrice);

  const gap = rec.inventory_position - rec.reorder_point;
  $('res-stock-gap').textContent = (gap >= 0 ? '+' : '') + fmtNum(gap, 1) + ' units';

  // Status Banner & Recommendation Text
  const banner = $('inv-status-banner');
  const bannerIcon = $('banner-icon');
  const bannerTitle = $('banner-status-title');
  const bannerDesc = $('banner-status-desc');
  const actionSummary = $('res-action-summary');

  banner.className = 'inventory-status-banner';

  if (rec.stockout_risk) {
    banner.classList.add('status-critical');
    bannerIcon.textContent = '🔴';
    bannerTitle.textContent = 'CRITICAL STOCKOUT HAZARD: Inventory Position Below ROP';
    bannerDesc.textContent = `Current inventory position (${rec.inventory_position} units) is insufficient to buffer demand during the ${res.demand_stats.weeks_observed ? '7-day' : ''} lead time.`;
    actionSummary.textContent = `Trigger an emergency purchase order of ${Math.round(orderQty)} units immediately. At the current rate, stockout risk is elevated before the next shipment arrives.`;
  } else if (rec.overstock_risk) {
    banner.classList.add('status-excess');
    bannerIcon.textContent = '🟣';
    bannerTitle.textContent = 'OVERSTOCKED: Excess Holding Capital Detected';
    bannerDesc.textContent = `Stock coverage of ${rec.coverage_weeks.toFixed(1)} weeks exceeds the recommended 12-week maximum holding threshold.`;
    actionSummary.textContent = `Halt new purchase requisitions. Implement a promotional markdown to liquidate surplus inventory and avoid holding cost penalties.`;
  } else if (gap <= rec.reorder_point * 0.25) {
    banner.classList.add('status-warning');
    bannerIcon.textContent = '🟡';
    bannerTitle.textContent = 'REORDER IMMINENT: Approaching Buffer Threshold';
    bannerDesc.textContent = `Inventory position is within 25% of the Reorder Point (${rec.reorder_point.toFixed(1)} units).`;
    actionSummary.textContent = `Prepare purchase requisition for ${Math.round(orderQty)} units. Supplier procurement cycle should be scheduled this week.`;
  } else {
    bannerIcon.textContent = '🟢';
    bannerTitle.textContent = 'OPTIMAL: Stock Position Healthy';
    bannerDesc.textContent = `Inventory position comfortably meets the ${($('inv-service-level')?.value * 100).toFixed(0)}% service-level target.`;
    actionSummary.textContent = `No procurement action required. Current coverage provides ${rec.coverage_weeks.toFixed(1)} weeks of supply at normal consumption velocity.`;
  }
}

// ── Multidimensional Sales Analysis Section ──────────────────────────
async function loadSalesAnalysisData() {
  const sa = await apiGet('/sales-analysis');
  state.salesAnalysis = sa;
}

function loadSalesAnalysisTab() {
  const sa = state.salesAnalysis;
  if (!sa) return;

  // 1. Category Comparison Chart
  if (sa.by_category) {
    const cats = sa.by_category.map(c => c.category);
    const sales = sa.by_category.map(c => Math.round(c.sales / 1e5)); // in Lacs
    const aovs = sa.by_category.map(c => c.aov);

    createChart('chart-sales-by-category', {
      type: 'bar',
      data: {
        labels: cats,
        datasets: [
          {
            label: 'Sales (₹ Lakhs)',
            data: sales,
            backgroundColor: '#3b82f6',
            borderRadius: 4,
            yAxisID: 'y',
          },
          {
            label: 'Avg Ticket Size (₹)',
            data: aovs,
            type: 'line',
            borderColor: '#f59e0b',
            backgroundColor: '#f59e0b',
            borderWidth: 2,
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: getChartTheme().textColor }, grid: { color: getChartTheme().gridColor } },
          y: {
            ticks: { color: getChartTheme().textColor, callback: v => '₹' + v + 'L' },
            grid: { color: getChartTheme().gridColor },
          },
          y1: {
            position: 'right',
            ticks: { color: '#f59e0b', callback: v => '₹' + fmtNum(v) },
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
  }

  // 2. Regional Breakdown
  if (sa.by_region) {
    const regions = sa.by_region.map(r => r.Region);
    const sales = sa.by_region.map(r => r.sales);
    const ct = getChartTheme();

    createChart('chart-sales-by-region', {
      type: 'doughnut',
      data: {
        labels: regions,
        datasets: [
          {
            data: sales,
            backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'],
            borderColor: ct.borderColor,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: ct.textColor } },
        },
        cutout: '60%',
      },
    });
  }

  // 3. Customer Segments
  if (sa.by_segment) {
    const segs = sa.by_segment.map(s => s.Segment);
    const units = sa.by_segment.map(s => s.units);
    const ct = getChartTheme();

    createChart('chart-sales-by-segment', {
      type: 'polarArea',
      data: {
        labels: segs,
        datasets: [
          {
            data: units,
            backgroundColor: [
              'rgba(59, 130, 246, 0.7)',
              'rgba(16, 185, 129, 0.7)',
              'rgba(245, 158, 11, 0.7)',
            ],
            borderColor: ct.borderColor,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: ct.textColor } },
        },
        scales: {
          r: { ticks: { display: false }, grid: { color: ct.gridColor } },
        },
      },
    });
  }

  // 4. Top 10 Best Selling Products Table
  const topTbody = $('top-products-tbody');
  if (topTbody && sa.top_products) {
    const totalSales = sa.top_products.reduce((a, b) => a + b.sales, 0);
    topTbody.innerHTML = sa.top_products.map((p, idx) => {
      const share = ((p.sales / totalSales) * 100).toFixed(1);
      return `
        <tr>
          <td><span style="font-weight:700; color:#60a5fa;">#${idx + 1}</span></td>
          <td><strong>${p['Product Name']}</strong></td>
          <td><span class="category-badge">${p.category}</span></td>
          <td>${fmtNum(p.units)} units</td>
          <td>${fmtINR(p.sales)}</td>
          <td>${share}% of top 10</td>
        </tr>
      `;
    }).join('');
  }
}

// ── Product Catalog Section ──────────────────────────────────────────
async function loadCatalogData() {
  const cat = await apiGet('/catalog');
  state.catalog = cat;
}

function renderCatalogTable() {
  const query = $('catalog-search')?.value.toLowerCase().trim() || '';
  const catFilter = $('catalog-cat-filter')?.value || 'All';
  const sort = $('catalog-sort')?.value || 'sales-desc';

  let items = [...state.catalog];

  // Filter Category
  if (catFilter !== 'All') {
    items = items.filter(i => i.category === catFilter);
  }

  // Search Filter
  if (query) {
    items = items.filter(i =>
      i.product_name.toLowerCase().includes(query) ||
      i.sub_category.toLowerCase().includes(query) ||
      i.category.toLowerCase().includes(query)
    );
  }

  // Sort
  if (sort === 'sales-desc') items.sort((a, b) => b.total_sales - a.total_sales);
  else if (sort === 'qty-desc') items.sort((a, b) => b.total_quantity - a.total_quantity);
  else if (sort === 'price-desc') items.sort((a, b) => b.avg_price - a.avg_price);
  else if (sort === 'returns-asc') items.sort((a, b) => a.return_rate - b.return_rate);

  $('catalog-count-label').textContent = `Showing ${items.length} of ${state.catalog.length} retail products`;

  const tbody = $('catalog-tbody');
  if (tbody) {
    if (items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="table-loading">No products match your search criteria.</td></tr>';
      return;
    }
    tbody.innerHTML = items.map(p => `
      <tr>
        <td><strong>${p.product_name}</strong></td>
        <td><span class="category-badge">${p.category}</span></td>
        <td><span class="subcat-badge">${p.sub_category}</span></td>
        <td>₹${fmtNum(p.avg_price)}</td>
        <td>${fmtNum(p.total_quantity)}</td>
        <td>${fmtINR(p.total_sales)}</td>
        <td>${fmtNum(p.order_count)}</td>
        <td><span style="color:${p.return_rate > 6 ? '#f87171' : '#34d399'}; font-weight:600;">${p.return_rate}%</span></td>
        <td>
          <button class="btn-plan-action" onclick="planItem('${p.sub_category}')">Forecast &amp; Plan</button>
        </td>
      </tr>
    `).join('');
  }
}

// ── Alerts & Exception Management Section ─────────────────────────────
async function loadAlertsData() {
  const alerts = await apiGet('/alerts');
  state.alerts = alerts;
  const criticalCount = alerts.filter(a => a.urgency === 'HIGH').length;
  $('alerts-badge').textContent = criticalCount;
}

function renderAlertsTab(filter) {
  let items = [...state.alerts];

  if (filter && filter !== 'ALL') {
    items = items.filter(a => a.urgency === filter);
  }

  const container = $('full-alerts-list');
  if (!container) return;

  if (items.length === 0) {
    container.innerHTML = '<div class="empty-state" style="grid-column: span 2;">No active exceptions found for this filter category.</div>';
    return;
  }

  container.innerHTML = items.map(a => {
    let cardCls = 'card-normal';
    let badgeCls = 'optimal';
    let statusText = 'OPTIMAL BUFFER';

    if (a.urgency === 'HIGH') {
      cardCls = 'card-critical';
      badgeCls = 'critical';
      statusText = 'CRITICAL STOCKOUT HAZARD';
    } else if (a.urgency === 'MEDIUM') {
      cardCls = 'card-warning';
      badgeCls = 'warning';
      statusText = 'REORDER IMMINENT';
    } else if (a.urgency === 'LOW') {
      cardCls = 'card-excess';
      badgeCls = 'excess';
      statusText = 'EXCESS INVENTORY';
    }

    return `
      <div class="alert-card-full ${cardCls}">
        <div class="alert-card-top">
          <div>
            <div class="alert-card-title">${a.sub_category}</div>
            <div class="alert-card-cat">${a.category} · Safety Stock: ${a.safety_stock.toFixed(1)} units</div>
          </div>
          <span class="urgency-badge ${badgeCls}">${statusText}</span>
        </div>

        <div class="alert-card-directive">
          <strong>Directive:</strong> ${a.action}
          <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">${a.reason}</div>
        </div>

        <div class="alert-stats-row">
          <span>Current Stock: <strong>${a.current_stock}</strong> units</span>
          <span>Reorder Point: <strong>${a.reorder_point.toFixed(1)}</strong></span>
          <span>Coverage: <strong>${a.coverage_weeks.toFixed(1)}</strong> wks</span>
          <span>Est. Capital: <strong>${fmtINR(a.est_order_cost)}</strong></span>
        </div>

        <button class="btn-plan-action" style="align-self: flex-start;" onclick="planItem('${a.sub_category}')">
          Open in Inventory Planner →
        </button>
      </div>
    `;
  }).join('');
}

// ── Global Helper for Cross-Tab Plan Button ──────────────────────────
window.planItem = function(subcat) {
  state.selectedSubcat = subcat;
  if ($('side-subcat-select')) $('side-subcat-select').value = subcat;
  if ($('fc-subcat-select')) $('fc-subcat-select').value = subcat;
  if ($('inv-subcat-select')) $('inv-subcat-select').value = subcat;

  // Find default stock from stats if available
  const p = state.products.find(item => item.sub_category === subcat);
  if (p && $('inv-current-stock')) {
    $('inv-current-stock').value = Math.round(p.mean_weekly_qty * 1.2);
  }

  switchTab('inventory');
};

// ── Start on DOMContentLoaded ────────────────────────────────────────
window.addEventListener('DOMContentLoaded', initApp);

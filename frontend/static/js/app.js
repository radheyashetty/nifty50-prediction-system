/* ===== NIFTY 50 Stock Lab — Frontend Logic ===== */

// -- Utility Helpers --
function pct(v) { return `${(Number(v || 0) * 100).toFixed(1)}%`; }
function num(v, d = 2) { return Number(v || 0).toFixed(d); }
function currency(v) { return `₹${num(v, 2)}`; }
function tsNow() { return new Date().toLocaleString(); }

function getRiskClass(riskLevel) {
  const t = String(riskLevel || "").toLowerCase();
  if (t.includes("low")) return "risk-low";
  if (t.includes("high")) return "risk-high";
  return "risk-medium";
}

// -- Chart Instance Management --
const charts = {};
function destroyChart(key) { if (charts[key]) { charts[key].destroy(); delete charts[key]; } }
function mountChart(containerId, key, config) {
  const c = document.getElementById(containerId);
  if (!c || typeof Chart === "undefined") return null;
  c.innerHTML = "";
  const canvas = document.createElement("canvas");
  c.appendChild(canvas);
  destroyChart(key);
  charts[key] = new Chart(canvas, config);
  return charts[key];
}

// -- State --
let sectorDataLoaded = false;
let topPicksLoaded = false;
let stocksMetaPromise = null;

// -- Input Context --
function getInputContext() {
  const ticker = document.getElementById("ticker").value;
  const raw = Number(document.getElementById("lookback").value);
  const lookbackDays = Number.isFinite(raw) ? Math.min(730, Math.max(90, Math.round(raw))) : 365;
  document.getElementById("lookback").value = String(lookbackDays);
  return { ticker, lookbackDays };
}

// ============================
//  PIPELINE STEPPER
// ============================
function showStepper() {
  const el = document.getElementById("pipelineStepper");
  if (el) { el.classList.remove("d-none"); resetStepper(); }
}

function hideStepper() {
  const el = document.getElementById("pipelineStepper");
  if (el) el.classList.add("d-none");
}

function resetStepper() {
  document.querySelectorAll(".pipeline-step[data-step]").forEach(s => {
    s.classList.remove("active", "done");
  });
}

function setStepActive(stepNum) {
  document.querySelectorAll(".pipeline-step[data-step]").forEach(s => {
    const n = Number(s.dataset.step);
    if (n < stepNum) { s.classList.remove("active"); s.classList.add("done"); }
    else if (n === stepNum) { s.classList.add("active"); s.classList.remove("done"); }
    else { s.classList.remove("active", "done"); }
  });
}

function setAllStepsDone() {
  document.querySelectorAll(".pipeline-step[data-step]").forEach(s => {
    s.classList.remove("active"); s.classList.add("done");
  });
}

// ============================
//  SIGNAL GAUGE
// ============================
function renderGauge(bullishProb) {
  const prob = Number(bullishProb || 0);
  const arcLen = 251.2; // approximate arc length of the semicircle
  const fill = prob * arcLen;
  const arc = document.getElementById("gaugeArc");
  const sigText = document.getElementById("gaugeSignalText");
  const confText = document.getElementById("gaugeConfText");

  if (arc) {
    arc.style.transition = "stroke-dasharray 0.8s ease";
    arc.setAttribute("stroke-dasharray", `${fill} ${arcLen}`);
  }

  const signal = prob > 0.55 ? "BULLISH" : "BEARISH";
  const confidence = Math.max(prob, 1 - prob);
  if (sigText) {
    sigText.textContent = signal;
    sigText.style.color = signal === "BULLISH" ? "var(--green)" : "var(--red)";
  }
  if (confText) confText.textContent = `${pct(confidence)} confidence`;
}

// ============================
//  API CALLS
// ============================
async function fetchAnalysisResult(ticker, lookbackDays, analysisMode) {
  const r = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, lookback_days: lookbackDays, analysis_mode: analysisMode }),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Analysis failed"); }
  return r.json();
}

async function uploadAndTrain(file, ticker, lookbackDays, analysisMode) {
  const fd = new FormData();
  fd.append("ticker", ticker);
  fd.append("lookback_days", String(lookbackDays));
  fd.append("analysis_mode", analysisMode);
  fd.append("file", file, file.name || "dataset.csv");
  const r = await fetch("/api/upload-train", { method: "POST", body: fd });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Upload failed"); }
  return r.json();
}

async function fetchScreenerResults(sector, minConfidence, minVolumeRatio, regimeFilter = null) {
  const r = await fetch("/api/screener", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sector, min_confidence: Number(minConfidence || 0.55), min_volume_ratio: Number(minVolumeRatio || 0), regime_filter: regimeFilter }),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Screener failed"); }
  return r.json();
}

async function fetchSectorAnalysis() {
  const r = await fetch("/api/sector-analysis", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sector: null }) });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Sector analysis failed"); }
  return r.json();
}

async function fetchComparison(tickers) {
  const r = await fetch("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tickers }) });
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail || "Compare failed"); }
  return r.json();
}

async function fetchStocksMeta() {
  if (!stocksMetaPromise) {
    stocksMetaPromise = fetch("/api/stocks").then(r => { if (!r.ok) throw new Error("Failed"); return r.json(); }).catch(e => { stocksMetaPromise = null; throw e; });
  }
  return stocksMetaPromise;
}

// ============================
//  STATUS HELPERS
// ============================
function setStatus(msg, isError = false) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = isError ? "status-pill border-danger text-danger" : "status-pill";
}

function setLastUpdated(msg) { document.getElementById("lastUpdated").textContent = msg; }

// ============================
//  RENDERERS
// ============================

function renderMetricCards(result) {
  const pred = result.predictions || {};
  const risk = result.risk_metrics || {};
  const sector = result.sector || result.sector_context?.sector || "N/A";

  document.getElementById("metricPrice").textContent = currency(result.latest_price);
  document.getElementById("metricSector").textContent = sector;

  const riskEl = document.getElementById("metricRisk");
  riskEl.innerHTML = `<span class="risk-pill ${getRiskClass(risk.risk_level)}">${risk.risk_level || "N/A"}</span>`;

  const modelPerf = result.model_performance || {};
  document.getElementById("metricModel").textContent = `AUC ${num(modelPerf.xgboost_auc || 0, 3)}`;

  // Gauge
  renderGauge(pred.bullish_probability || 0);

  // Probability bars
  const scores = result.model_scores || {};
  document.getElementById("probabilityBars").innerHTML = `
    <div class="mb-2">
      <div class="d-flex justify-content-between small"><span>XGBoost</span><strong>${pct(scores.xgboost_prob)}</strong></div>
      <div class="progress"><div class="progress-bar bg-success" style="width:${(Number(scores.xgboost_prob||0)*100)}%"></div></div>
    </div>
    <div class="mb-2">
      <div class="d-flex justify-content-between small"><span>Random Forest</span><strong>${pct(scores.random_forest_prob)}</strong></div>
      <div class="progress"><div class="progress-bar bg-info" style="width:${(Number(scores.random_forest_prob||0)*100)}%"></div></div>
    </div>
    <div>
      <div class="d-flex justify-content-between small"><span>Ensemble</span><strong>${pct(scores.ensemble_prob || pred.bullish_probability)}</strong></div>
      <div class="progress"><div class="progress-bar" style="width:${(Number(scores.ensemble_prob||pred.bullish_probability||0)*100)}%; background:var(--accent)"></div></div>
    </div>
  `;
}

function renderProbabilityChart(result) {
  const scores = result?.model_scores || {};
  const xgb = Number(scores.xgboost_prob || 0);
  const rf = Number(scores.random_forest_prob || 0);
  const ens = Number(scores.ensemble_prob || result?.predictions?.bullish_probability || 0);
  if (!xgb && !rf && !ens) { document.getElementById("probabilityChart").innerHTML = '<div class="empty-state">Run analysis to see model mix.</div>'; return; }

  mountChart("probabilityChart", "probability", {
    type: "doughnut",
    data: {
      labels: ["XGBoost", "Random Forest", "Ensemble"],
      datasets: [{ data: [xgb, rf, ens], backgroundColor: ["#6c8cff", "#34d399", "#fbbf24"], borderWidth: 0, hoverOffset: 6 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { color: "#8a95b0", font: { size: 11 } } }, tooltip: { callbacks: { label: ctx => `${ctx.label}: ${pct(ctx.raw)}` } } },
      cutout: "64%",
    },
  });
}

function renderTopFeaturesChart(result) {
  const feats = Array.isArray(result?.top_features) ? result.top_features.slice(0, 8) : [];
  if (!feats.length) { document.getElementById("topFeaturesChart").innerHTML = '<div class="empty-state">Feature drivers appear after analysis.</div>'; return; }

  mountChart("topFeaturesChart", "topFeatures", {
    type: "bar",
    data: {
      labels: feats.map(f => String(f.feature_name || f.feature || "?")),
      datasets: [{ label: "SHAP", data: feats.map(f => Number(f.shap_value || 0)), backgroundColor: feats.map(f => Number(f.shap_value || 0) >= 0 ? "rgba(52,211,153,0.7)" : "rgba(251,113,133,0.7)"), borderRadius: 6 }],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `SHAP: ${num(ctx.raw, 4)}` } } },
      scales: { x: { ticks: { color: "#8a95b0" }, grid: { color: "rgba(255,255,255,0.04)" } }, y: { ticks: { color: "#eaf0fb", font: { size: 11 } }, grid: { display: false } } },
    },
  });
}

function renderTopFeatures(result) {
  const feats = Array.isArray(result.top_features) ? result.top_features : [];
  const rows = feats.slice(0, 10).map(f => `<tr><td>${f.feature_name}</td><td class="${f.direction === 'bullish' ? 'bullish' : 'bearish'}">${f.direction}</td><td>${num(Math.abs(f.shap_value), 4)}</td></tr>`).join("");
  document.getElementById("topFeaturesTable").innerHTML = `<thead><tr><th>Feature</th><th>Direction</th><th>|SHAP|</th></tr></thead><tbody>${rows}</tbody>`;
  renderTopFeaturesChart(result);
}

function renderMiniChart(prices) {
  const c = document.getElementById("miniChart");
  if (!c) return;
  const series = Array.isArray(prices) ? prices.map(v => Number(v)).filter(v => Number.isFinite(v)) : [];
  if (series.length < 2) { c.innerHTML = '<div class="empty-state">Price chart appears after analysis.</div>'; return; }

  mountChart("miniChart", "miniChart", {
    type: "line",
    data: { labels: series.map((_, i) => i + 1), datasets: [{ data: series, borderWidth: 2, tension: 0.35, borderColor: "#6c8cff", backgroundColor: "rgba(108,140,255,0.12)", fill: true, pointRadius: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: true } }, scales: { x: { display: false }, y: { display: false } } },
  });
}

function renderModelMetrics(result) {
  const m = result.model_metrics || {};
  const cm = m.confusion_matrix || [[0,0],[0,0]];
  document.getElementById("predictionSummary").innerHTML = `
    <h6 class="section-title">Model Metrics</h6>
    <div class="row g-2">
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">Accuracy</div><div class="fw-bold">${num(m.accuracy,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">Precision</div><div class="fw-bold">${num(m.precision,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">Recall</div><div class="fw-bold">${num(m.recall,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">F1</div><div class="fw-bold">${num(m.f1_score,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">ROC-AUC</div><div class="fw-bold">${num(m.roc_auc,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">Brier</div><div class="fw-bold">${num(m.brier_score,4)}</div></div>
    </div>
    <table class="table table-sm mt-2 mb-0"><thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
    <tbody><tr><th>True 0</th><td>${cm[0]?.[0]??0}</td><td>${cm[0]?.[1]??0}</td></tr><tr><th>True 1</th><td>${cm[1]?.[0]??0}</td><td>${cm[1]?.[1]??0}</td></tr></tbody></table>
  `;
}

function renderIndicators(result) {
  const i = result.indicators || result.technical_indicators || {};
  document.getElementById("technicalSummary").innerHTML = `
    <h6 class="section-title">Technical Indicators</h6>
    <div class="row g-2">
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">RSI 14</div><div class="fw-bold">${num(i.rsi_14,2)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">MACD Hist</div><div class="fw-bold">${num(i.macd_histogram,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">ADX 14</div><div class="fw-bold">${num(i.adx_14,2)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">BB %B</div><div class="fw-bold">${num(i.bollinger_pct_b,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">ATR %</div><div class="fw-bold">${num(i.atr_pct,4)}</div></div>
      <div class="col-4"><div class="small" style="color:var(--ink-muted)">Vol Ratio</div><div class="fw-bold">${num(i.volume_ratio,3)}</div></div>
    </div>
  `;
}

function renderRegime(result) {
  const regime = result.regime || result.regime_analysis || {};
  const risk = result.risk_metrics || {};
  document.getElementById("regimeSummary").innerHTML = `
    <h6 class="section-title">Market Regime</h6>
    <p class="mb-1"><strong>Regime:</strong> ${regime.name || regime.current_regime || "N/A"}</p>
    <p class="mb-1"><strong>Vol Regime:</strong> ${regime.volatility_regime || "N/A"}</p>
    <p class="mb-0"><strong>HMM:</strong> ${regime.hmm_regime || "N/A"}</p>
  `;
  document.getElementById("riskSummary").innerHTML = `
    <h6 class="section-title">Risk Analysis</h6>
    <p class="mb-1"><strong>30D Vol:</strong> ${num(risk.volatility_30d)}%</p>
    <p class="mb-1"><strong>Max DD:</strong> ${risk.max_drawdown_analysis || "N/A"}</p>
    <p class="mb-0"><strong>Risk:</strong> <span class="risk-pill ${getRiskClass(risk.risk_level)}">${risk.risk_level || "N/A"}</span></p>
  `;
}

function renderBacktest(result) {
  const bt = result.backtest || {};
  const ds = [["ML Strategy", bt.ml_strategy], ["MA Crossover", bt.ma_crossover], ["RSI Strategy", bt.rsi_strategy], ["Buy & Hold", bt.buy_and_hold]];
  const best = Math.max(...ds.map(x => Number(x[1]?.sharpe_ratio || 0)));
  const rows = ds.map(([name, m]) => {
    const s = Number(m?.sharpe_ratio || 0);
    const hl = s === best ? ' class="table-success"' : "";
    return `<tr${hl}><td>${name}</td><td>${num(m?.total_return_pct||0)}%</td><td>${num(s)}</td><td>${num(m?.max_drawdown_pct||0)}%</td><td>${num(m?.win_rate_pct||0)}%</td></tr>`;
  }).join("");
  document.getElementById("backtestTable").innerHTML = `<thead><tr><th>Strategy</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th></tr></thead><tbody>${rows}</tbody>`;
}

function renderTopPicks(data) {
  const c = document.getElementById("topPicks");
  if (!c) return;
  const bulls = Array.isArray(data?.bullish) ? [...data.bullish] : [];
  const picks = bulls.sort((a, b) => Number(b.confidence||0) - Number(a.confidence||0)).slice(0, 3);

  if (!picks.length) { c.innerHTML = `<div class="col-12"><div class="top-pick-card loading d-flex align-items-center justify-content-between"><div><div class="ticker">Top Picks</div><div class="small" style="color:var(--ink-dim)">Run screener to load.</div></div><span class="badge text-bg-light">LIVE</span></div></div>`; return; }

  c.innerHTML = picks.map(s => {
    const conf = Number(s.confidence || 0);
    return `<div class="col-md-4"><div class="top-pick-card" role="button" tabindex="0" data-ticker="${s.ticker}">
      <div class="d-flex justify-content-between align-items-start"><div><div class="ticker">${s.ticker}</div><div class="small" style="color:var(--ink-dim)">${s.sector || "—"}</div></div><span class="badge" style="background:var(--green-bg);color:var(--green)">BUY</span></div>
      <div class="confidence-bar"><div style="width:${Math.min(100, conf*100)}%"></div></div>
      <div class="d-flex justify-content-between small" style="color:var(--ink-muted)"><span>${pct(conf)}</span><span>${currency(s.latest_price||0)}</span></div>
    </div></div>`;
  }).join("");

  c.querySelectorAll(".top-pick-card[data-ticker]").forEach(card => {
    const t = card.dataset.ticker;
    const go = () => { if (!t) return; document.getElementById("ticker").value = t; analyze(); };
    card.addEventListener("click", go);
    card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}

// ===== Screener renderers =====
function renderScreenerChart(data) {
  const b = Number(data?.bullish_count || 0), be = Number(data?.bearish_count || 0);
  if (!b && !be) { document.getElementById("screenerChart").innerHTML = '<div class="empty-state">Run screener to visualize.</div>'; return; }
  mountChart("screenerChart", "screener", {
    type: "doughnut",
    data: { labels: ["Bullish", "Bearish"], datasets: [{ data: [b, be], backgroundColor: ["#34d399", "#fb7185"], borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: "#8a95b0" } } }, cutout: "64%" },
  });
}

function renderScreenerResults(data) {
  const bullish = Array.isArray(data.bullish) ? data.bullish : [];
  const bearish = Array.isArray(data.bearish) ? data.bearish : [];
  const mkRows = rows => rows.map((r, i) => `<tr data-ticker="${r.ticker}" class="screener-row ${String(r.signal||'').toUpperCase()==='BULLISH'?'bullish-row':'bearish-row'}" style="cursor:pointer"><td>${i+1}</td><td>${r.ticker}</td><td>${r.sector||"?"}</td><td>${pct(r.confidence)}</td><td>${num(r.rsi_14,2)}</td><td>${num(r.macd_histogram,3)}</td><td>${currency(r.latest_price)}</td></tr>`).join("");

  document.getElementById("screenerBullishTable").innerHTML = `<thead><tr><th>#</th><th>Ticker</th><th>Sector</th><th>Conf</th><th>RSI</th><th>MACD</th><th>Price</th></tr></thead><tbody>${mkRows(bullish)}</tbody>`;
  document.getElementById("screenerBearishTable").innerHTML = `<thead><tr><th>#</th><th>Ticker</th><th>Sector</th><th>Conf</th><th>RSI</th><th>MACD</th><th>Price</th></tr></thead><tbody>${mkRows(bearish)}</tbody>`;

  document.querySelectorAll(".screener-row").forEach(row => {
    row.addEventListener("click", () => {
      const t = row.dataset.ticker; if (!t) return;
      document.getElementById("ticker").value = t;
      bootstrap.Tab.getOrCreateInstance(document.querySelector('[data-bs-target="#overview"]')).show();
      analyze();
    });
  });

  document.getElementById("screenerSummary").textContent = `Scanned ${data.total_scanned||0} · ${data.bullish_count||0} bullish · ${data.bearish_count||0} bearish · ${num(data.scan_duration_sec||0)}s`;
  renderScreenerChart(data);
}

// ===== Sector heatmap =====
function renderSectorHeatmap(data) {
  const sectors = Array.isArray(data.sectors) ? data.sectors : [];
  const insights = Array.isArray(data.rotation_insights) ? data.rotation_insights : [];
  
  const container = document.getElementById("sectorHeatmap");
  if (!container) return;

  container.innerHTML = sectors.map(s => {
    const b = Number(s.bullish_pct || 0);
    const conf = Number(s.avg_confidence || 0);
    const vol = Number(s.avg_volatility || 0);
    
    // Determine color based on bullishness
    let bgColor = "var(--heat-neutral)";
    let badgeText = "Neutral";
    if (b > 60) { bgColor = "var(--heat-bullish)"; badgeText = "Bullish"; }
    else if (b < 40) { bgColor = "var(--heat-bearish)"; badgeText = "Bearish"; }

    return `
      <div class="col-md-6 col-xl-4">
        <div class="heat-card" style="background: ${bgColor}">
          <div class="heat-badge">${badgeText}</div>
          <div>
            <h6>${s.sector_name}</h6>
            <div class="pct-value">${num(b, 1)}%</div>
            <div class="heat-stats"> Conviction: ${pct(conf)}</div>
          </div>
          <div class="mt-2 heat-stats">
            <div>Volatility: ${num(vol * 100, 2)}%</div>
            <div>Top Pick: <span class="fw-bold">${s.top_stock || "N/A"}</span></div>
          </div>
        </div>
      </div>
    `;
  }).join("") || '<div class="col-12 empty-state">No sector data.</div>';

  document.getElementById("rotationInsights").innerHTML = insights.map(i => `<li>${i}</li>`).join("") || "<li>No rotation insights.</li>";

  // Detailed Table
  document.getElementById("sectorDetailTable").innerHTML = `
    <thead>
      <tr>
        <th>Sector</th>
        <th>Bullish%</th>
        <th>Bull/Bear</th>
        <th>Avg Conf</th>
        <th>Avg Vol</th>
        <th>Top Stock</th>
      </tr>
    </thead>
    <tbody>
      ${sectors.map(s => `
        <tr>
          <td>${s.sector_name}</td>
          <td class="${s.bullish_pct > 50 ? 'bullish' : 'bearish'} fw-bold">${num(s.bullish_pct, 1)}%</td>
          <td>${s.bullish_count}/${s.bearish_count}</td>
          <td>${pct(s.avg_confidence)}</td>
          <td>${num(s.avg_volatility * 100, 2)}%</td>
          <td><strong>${s.top_stock || "N/A"}</strong></td>
        </tr>
      `).join("")}
    </tbody>
  `;
  
  document.getElementById("sectorSummary").textContent = `Analysis complete for ${sectors.length} sectors using full Ensemble Ensemble model.`;
}

// ===== Compare renderers =====
function renderCompareChart(data) {
  const stocks = Array.isArray(data?.stocks) ? data.stocks.slice(0, 4) : [];
  if (!stocks.length) { document.getElementById("compareChart").innerHTML = '<div class="empty-state">Run compare first.</div>'; return; }
  mountChart("compareChart", "compare", {
    type: "bar",
    data: { labels: stocks.map(s => s.ticker||"?"), datasets: [{ label: "Confidence", data: stocks.map(s => Number(s.confidence || s.predictions?.confidence || 0)), backgroundColor: "rgba(108,140,255,0.7)", borderRadius: 6 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 1, ticks: { color: "#8a95b0", callback: v => `${(v*100).toFixed(0)}%` }, grid: { color: "rgba(255,255,255,0.04)" } }, y: { ticks: { color: "#eaf0fb" }, grid: { display: false } } } },
  });
}

function renderCompareProbabilityChart(data) {
  const stocks = Array.isArray(data?.stocks) ? data.stocks.slice(0, 4) : [];
  if (!stocks.length) { document.getElementById("compareProbChart").innerHTML = '<div class="empty-state">Run compare first.</div>'; return; }
  mountChart("compareProbChart", "compareProb", {
    type: "bar",
    data: { labels: stocks.map(s => s.ticker), datasets: [{ label: "Bullish", data: stocks.map(s => Number(s.predictions?.bullish_probability||0)), backgroundColor: "rgba(52,211,153,0.7)", borderRadius: 5 }, { label: "Bearish", data: stocks.map(s => Number(s.predictions?.bearish_probability||0)), backgroundColor: "rgba(251,113,133,0.7)", borderRadius: 5 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: "#8a95b0" } } }, scales: { x: { ticks: { color: "#eaf0fb" }, grid: { color: "rgba(255,255,255,0.04)" } }, y: { min: 0, max: 1, ticks: { color: "#8a95b0", callback: v => `${(v*100).toFixed(0)}%` }, grid: { color: "rgba(255,255,255,0.04)" } } } },
  });
}

function renderCompareRiskReturnChart(data) {
  const stocks = Array.isArray(data?.stocks) ? data.stocks.slice(0, 4) : [];
  if (!stocks.length) { document.getElementById("compareRiskReturnChart").innerHTML = '<div class="empty-state">Run compare first.</div>'; return; }
  const pts = stocks.map(s => ({ x: Number(s.risk_metrics?.volatility_30d||0), y: Number((s.indicators?.return_5d||s.technical_indicators?.return_5d||0)*100), ticker: s.ticker, conf: Number(s.confidence||0) }));
  mountChart("compareRiskReturnChart", "compareRR", {
    type: "scatter",
    data: { datasets: [{ label: "Stocks", data: pts, pointRadius: pts.map(p => 5 + p.conf * 7), pointBackgroundColor: pts.map(p => p.y >= 0 ? "rgba(52,211,153,0.8)" : "rgba(251,113,133,0.8)"), pointBorderColor: "rgba(255,255,255,0.5)", pointBorderWidth: 1 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { title: items => items?.[0]?.raw?.ticker, label: ctx => `Vol: ${num(ctx.raw.x)}% | Ret: ${num(ctx.raw.y)}%` } } }, scales: { x: { title: { display: true, text: "30D Volatility %", color: "#8a95b0" }, ticks: { color: "#8a95b0" }, grid: { color: "rgba(255,255,255,0.04)" } }, y: { title: { display: true, text: "5D Return %", color: "#8a95b0" }, ticks: { color: "#8a95b0" }, grid: { color: "rgba(255,255,255,0.04)" } } } },
  });
}

function renderComparison(data) {
  const stocks = Array.isArray(data.stocks) ? data.stocks : [];
  const el = document.getElementById("compareSummary");
  if (el && stocks.length) {
    const ranked = [...stocks].sort((a, b) => Number(b.confidence||0) - Number(a.confidence||0));
    el.textContent = `Best confidence: ${ranked[0].ticker} (${pct(ranked[0].confidence)})`;
  }

  document.getElementById("compareSignalCards").innerHTML = stocks.map(s => `<div class="col-md-6 col-xl-3"><div class="metric-card p-3 h-100">
    <div class="metric-label">${s.ticker}</div>
    <div class="metric-value ${(s.signal||s.predictions?.decision)==="BULLISH"?"bullish":"bearish"}">${s.signal||s.predictions?.decision||"N/A"}</div>
    <div class="small" style="color:var(--ink-muted)">${pct(s.confidence||0)} · ${currency(s.latest_price||0)}</div>
  </div></div>`).join("");

  const metrics = [
    { label: "Signal", get: s => s.signal || s.predictions?.decision || "N/A" },
    { label: "Confidence", get: s => pct(s.confidence || 0) },
    { label: "RSI", get: s => num(s.indicators?.rsi_14 || s.technical_indicators?.rsi_14 || 0, 2) },
    { label: "MACD", get: s => num(s.indicators?.macd_histogram || s.technical_indicators?.macd_histogram || 0, 3) },
    { label: "Risk", get: s => s.risk_metrics?.risk_level || "N/A" },
    { label: "Price", get: s => currency(s.latest_price || 0) },
  ];
  document.getElementById("compareTable").innerHTML = `<thead><tr><th>Metric</th>${stocks.map(s=>`<th>${s.ticker}</th>`).join("")}</tr></thead><tbody>${metrics.map(m=>`<tr><th>${m.label}</th>${stocks.map(s=>`<td>${m.get(s)}</td>`).join("")}</tr>`).join("")}</tbody>`;

  renderCompareChart(data);
  renderCompareProbabilityChart(data);
  renderCompareRiskReturnChart(data);
}

// ===== Upload renderers =====
function renderUploadSummary(values) {
  const summary = values.upload_summary || {};
  const cards = [
    { label: "Clean Rows", value: num(values.rows || summary.rows_after_cleaning || 0, 0), sub: values.source_file || "Uploaded" },
    { label: "Ticker", value: values.ticker_resolved || "AUTO", sub: `Req: ${values.ticker_requested || "(auto)"}` },
    { label: "Symbols", value: num(summary.unique_symbols ?? 1, 0), sub: summary.has_symbol_column ? "Column detected" : "Single stock" },
    { label: "Sectors", value: num(summary.unique_sectors ?? 1, 0), sub: (summary.top_sectors?.[0]?.sector) || "Unknown" },
  ];
  document.getElementById("uploadSummaryGrid").innerHTML = cards.map(c => `<div class="upload-summary-card"><div class="label">${c.label}</div><div class="value">${c.value}</div><div class="subvalue">${c.sub||""}</div></div>`).join("");
}

function renderUploadValues(values) {
  renderUploadSummary(values || {});
  const preview = Array.isArray(values.preview) ? values.preview.slice(0, 5) : [];
  if (preview.length) {
    document.getElementById("uploadValues").innerHTML = `<div class="upload-preview-shell"><strong>Preview</strong>
      <div class="table-responsive mt-1"><table class="table table-sm"><thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Vol</th></tr></thead>
      <tbody>${preview.map(r => `<tr><td>${r.date||"-"}</td><td>${num(r.open)}</td><td>${num(r.high)}</td><td>${num(r.low)}</td><td>${num(r.close)}</td><td>${num(r.volume,0)}</td></tr>`).join("")}</tbody></table></div></div>`;
  }
}

// ===== Master render =====
function renderAnalysis(result) {
  renderMetricCards(result);
  renderModelMetrics(result);
  renderTopFeatures(result);
  renderIndicators(result);
  renderRegime(result);
  renderBacktest(result);
  renderMiniChart(result.chart_prices || []);
  renderProbabilityChart(result);
}

// ============================
//  MAIN ACTIONS
// ============================

async function analyze() {
  const { ticker, lookbackDays } = getInputContext();
  const mode = document.getElementById("analysisMode").value;
  const btn = document.getElementById("analyzeBtn");
  const spinner = btn.querySelector(".btn-spinner");

  btn.disabled = true;
  spinner.classList.remove("d-none");
  btn.querySelector(".btn-label").textContent = "Analyzing…";
  setStatus(`Analyzing ${ticker}…`);
  setLastUpdated(`Started: ${tsNow()}`);
  showStepper();

  // Simulate progressive stepper
  let stepInterval = setInterval(() => {}, 99999);
  const steps = [1, 2, 3, 4, 5, 6];
  let si = 0;
  stepInterval = setInterval(() => { if (si < steps.length) { setStepActive(steps[si]); si++; } }, 450);

  try {
    const result = await fetchAnalysisResult(ticker, lookbackDays, mode);
    clearInterval(stepInterval);
    setAllStepsDone();
    renderAnalysis(result);
    setStatus(`Done — ${result.signal || "N/A"} · ${pct(result.confidence)} confidence`);
    setLastUpdated(`Updated: ${tsNow()}`);
  } catch (err) {
    clearInterval(stepInterval);
    setStatus(String(err.message || err), true);
    setLastUpdated(`Failed: ${tsNow()}`);
  } finally {
    btn.disabled = false;
    spinner.classList.add("d-none");
    btn.querySelector(".btn-label").textContent = "Analyze";
    setTimeout(hideStepper, 2000);
  }
}

async function handleUploadTrain() {
  const fileInput = document.getElementById("datasetFile");
  const uploadTicker = String(document.getElementById("uploadTicker").value || "").trim();
  const mode = document.getElementById("analysisMode").value;
  const btn = document.getElementById("uploadTrainBtn");
  const spinner = btn.querySelector(".upload-spinner");
  const { lookbackDays } = getInputContext();
  const file = fileInput.files?.[0];
  if (!file) { setStatus("Select a file first.", true); return; }

  btn.disabled = true;
  spinner.classList.remove("d-none");
  btn.querySelector(".btn-label").textContent = "Uploading…";
  setStatus("Uploading dataset…");
  showStepper();

  try {
    const payload = await uploadAndTrain(file, uploadTicker, lookbackDays, mode);
    setAllStepsDone();
    renderUploadValues(payload.values || {});
    renderAnalysis(payload.result);
    setStatus(`Upload done — ${payload.values?.ticker_resolved || "OK"}`);
    setLastUpdated(`Upload result: ${tsNow()}`);
  } catch (err) {
    setStatus(String(err.message || err), true);
  } finally {
    btn.disabled = false;
    spinner.classList.add("d-none");
    btn.querySelector(".btn-label").textContent = "Upload + Predict";
    setTimeout(hideStepper, 2000);
  }
}

async function runScreener() {
  const btn = document.getElementById("runScreenerBtn");
  const sector = document.getElementById("screenerSector")?.value || "all";
  const minC = Number(document.getElementById("screenerConfidence")?.value || 0.55);
  const minV = Number(document.getElementById("screenerVolumeRatio")?.value || 0);
  if (btn) { btn.disabled = true; btn.textContent = "Running…"; }
  setStatus(`Screening ${sector}…`);
  try {
    const data = await fetchScreenerResults(sector, minC, minV);
    renderScreenerResults(data);
    renderTopPicks(data);
    setStatus("Screener done.");
  } catch (err) { setStatus(String(err.message || err), true); }
  finally { if (btn) { btn.disabled = false; btn.textContent = "Run Screener"; } }
}

async function loadSectorView() {
  const btn = document.getElementById("refreshSectorBtn");
  if (btn) { btn.disabled = true; btn.textContent = "Loading…"; }
  setStatus("Loading sectors…");
  try {
    const data = await fetchSectorAnalysis();
    renderSectorHeatmap(data);
    sectorDataLoaded = true;
    setStatus("Sectors loaded.");
  } catch (err) { setStatus(String(err.message || err), true); }
  finally { if (btn) { btn.disabled = false; btn.textContent = "Refresh"; } }
}

async function runComparison() {
  const btn = document.getElementById("runCompareBtn");
  const tickers = [
    document.getElementById("compareTicker1")?.value,
    document.getElementById("compareTicker2")?.value,
    document.getElementById("compareTicker3")?.value,
  ].filter(x => x);
  const uniq = [...new Set(tickers)];
  if (uniq.length < 2) { setStatus("Select at least 2 unique stocks.", true); return; }
  if (btn) { btn.disabled = true; btn.textContent = "Comparing…"; }
  setStatus(`Comparing ${uniq.join(", ")}…`);
  try {
    const data = await fetchComparison(uniq);
    renderComparison(data);
    setStatus("Comparison ready.");
  } catch (err) { setStatus(String(err.message || err), true); }
  finally { if (btn) { btn.disabled = false; btn.textContent = "Compare"; } }
}

async function loadTopPicks() {
  if (topPicksLoaded) return;
  try {
    const data = await fetchScreenerResults("all", 0.55, 0, null);
    renderTopPicks(data);
    topPicksLoaded = true;
  } catch (_) { renderTopPicks(null); }
}

async function initializeMetaSelectors() {
  try {
    const meta = await fetchStocksMeta();
    const bySector = meta.by_sector || {};
    const sel = document.getElementById("screenerSector");
    if (sel) {
      sel.innerHTML = ['<option value="all" selected>All Sectors</option>', ...Object.keys(bySector).sort().map(s => `<option value="${s}">${s}</option>`)].join("");
    }
  } catch (_) {}
}

// ============================
//  INIT
// ============================
document.getElementById("analyzeBtn").addEventListener("click", analyze);
document.getElementById("uploadTrainBtn")?.addEventListener("click", handleUploadTrain);
document.getElementById("runScreenerBtn")?.addEventListener("click", runScreener);
document.getElementById("refreshSectorBtn")?.addEventListener("click", loadSectorView);
document.getElementById("runCompareBtn")?.addEventListener("click", runComparison);

document.querySelectorAll('#resultTabs [data-bs-toggle="tab"]').forEach(tab => {
  tab.addEventListener("shown.bs.tab", e => {
    if (e.target?.getAttribute("data-bs-target") === "#sectorView" && !sectorDataLoaded) loadSectorView();
  });
});

window.addEventListener("load", async () => {
  await initializeMetaSelectors();
  loadTopPicks();
  analyze();
});

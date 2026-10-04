# ⚡ Quickstart Guide — NIFTY 50 Stock Lab

Get up and running with the NIFTY 50 Stock Prediction System in under 5 minutes.

---

## 1. Start the Application

```bash
# Windows
.\run.bat

# macOS / Linux
./run.sh
```

Navigate to: **[http://localhost:8501](http://localhost:8501)**

---

## 2. Using the Web Dashboard

### 🔍 Single Stock Analysis
1. Select any NIFTY 50 stock (e.g. `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`) from the dropdown.
2. Select your lookback window (default: `365` days).
3. Choose mode:
   - **Cache**: Instant inference using pre-trained XGBoost + Random Forest ensemble models.
   - **Train + Predict**: Re-trains the model live on fresh historical data.
4. Click **Analyze**.
5. Explore:
   - **Signal Gauge**: Bullish / Bearish recommendation with probability confidence.
   - **Model Probabilities**: XGBoost, Random Forest, and Ensemble breakdown.
   - **Top Drivers**: SHAP-based feature importance indicators.
   - **Top Picks**: Live high-conviction opportunities across NIFTY 50.

### 📊 NSE Stock Screener
1. Click the **Screener** tab.
2. Filter by sector (or select "All Sectors").
3. Set your minimum confidence threshold (e.g. `0.55`).
4. Click **Run Screener** to scan and rank bullish and bearish stocks across the index.

### 🗺️ Sector Heatmap
1. Click the **Sectors** tab.
2. Click **Refresh** to generate sector-wide breadth, average volatility, top picks, and rotation insights.

### ⚖️ Multi-Stock Comparison
1. Click the **Compare** tab.
2. Choose 2 or 3 stocks (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`).
3. Click **Compare** to inspect relative confidence, probability distributions, and risk vs return positioning.

### 🔬 Deep Dive & Strategy Backtesting
1. Go to **Deep Dive** for classification metrics (ROC-AUC, Precision, Recall, Confusion Matrix) and 30-session price charts.
2. Go to **Backtest** to compare the ML Strategy against Moving Average Crossover, RSI Strategy, and Benchmark Buy & Hold.

---

## 3. Python API Quickstart

You can use the backend modules directly in your own scripts or notebooks:

```python
from backend.predictions import PredictionService
from backend.screener import StockScreener

# Initialize service
service = PredictionService(lookback_days=365)

# Predict single stock
result = service.predict_stock("RELIANCE.NS", analysis_mode="cache")
print(f"Signal: {result['signal']} ({result['confidence']*100:.1f}%)")
print(f"Latest Price: ₹{result['latest_price']}")

# Run screener
screener = StockScreener(service)
screener_results = screener.run_screener(min_confidence=0.60)
print(f"Bullish stocks found: {len(screener_results['bullish'])}")
for stock in screener_results['bullish'][:3]:
    print(f" - {stock['ticker']}: {stock['confidence']*100:.1f}% confidence")
```

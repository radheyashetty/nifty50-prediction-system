# 📊 NIFTY 50 Stock Prediction & Analysis System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb5424.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end quantitative machine learning system for predicting short-term stock direction in the **NIFTY 50 index**. Built with **FastAPI**, **XGBoost + Random Forest ensemble**, **SHAP model explainability**, **real-time stock screening**, **sector heatmap breadth**, and **quantitative backtesting**.

---

## 📸 Visual Showcase

### 1. Interactive Prediction Dashboard
*Real-time signal gauge, confidence calibration, probability ensemble, top picks, and SHAP feature drivers.*

![NIFTY 50 Dashboard](docs/screenshots/01_dashboard_overview.png)

---

### 2. NSE Stock Screener
*Live scanning across all 50 constituent stocks with conviction scoring, RSI, MACD, and signal breakdown.*

![Stock Screener](docs/screenshots/03_stock_screener.png)

---

### 3. Sector Heatmap & Breadth
*Visualizing bullish/bearish market breadth across banking, IT, energy, auto, pharma, and utilities.*

![Sector Heatmap](docs/screenshots/04_sector_heatmap.png)

---

### 4. Multi-Stock Comparison
*Compare up to 3 stocks side-by-side with confidence rankings, probability distributions, and risk-return positioning.*

![Stock Comparison](docs/screenshots/05_stock_comparison.png)

---

### 5. Deep Dive & Classification Metrics
*Full model diagnostics (ROC-AUC, Precision, Recall, Confusion Matrix) paired with technical indicators and 30-session price trends.*

![Deep Dive Diagnostics](docs/screenshots/02_deep_dive_analysis.png)

---

### 6. Strategy Backtesting Engine
*Benchmarking the ML Strategy against dual moving-average crossovers, RSI mean-reversion, and Buy & Hold.*

![Strategy Backtesting](docs/screenshots/06_strategy_backtesting.png)

---

## 🎯 Key Features

- **Directional Prediction Pipeline**: Predicts 5-day forward price movement (>1.5% target) using 20+ engineered technical indicators.
- **Ensemble Architecture**: Combines **XGBoost** (primary classifier) with **Random Forest** for low-variance probability estimates.
- **Explainable AI (XAI)**: SHAP-driven factor attribution revealing exactly why a bullish or bearish signal was triggered.
- **Multi-Source Ingestion**: Resilient data layer supporting Yahoo Finance API, pre-downloaded offline CSVs, and user-uploaded custom datasets.
- **NSE Market Screener**: Filters index stocks by sector, confidence threshold, and volume ratios.
- **Sector Rotation Insights**: Dynamic sector breadth tracking to spot capital inflows and momentum shifts.
- **Quantitative Backtester**: Computes Total Return, Sharpe Ratio, Maximum Drawdown, Win Rate, and Calmar Ratio.
- **Zero-Setup Quickstart**: One-click launcher scripts for both Windows (`run.bat`) and macOS/Linux (`run.sh`).

---

## 🚀 Quick Start (Under 2 Minutes)

### Automated Launch

**Windows:**
```powershell
.\run.bat
```

**macOS / Linux:**
```bash
chmod +x run.sh
./run.sh
```

The startup script will automatically initialize the virtual environment, verify dependencies, and launch the server.

Once started, open: **[http://localhost:8501](http://localhost:8501)**

---

## 🛠️ Manual Installation

If you prefer installing dependencies directly:

```bash
# Clone the repository
git clone https://github.com/radheyashetty/nifty50-prediction-system.git
cd nifty50-prediction-system

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the app
uvicorn frontend.web_app:app --host 0.0.0.0 --port 8501 --reload
```

---

## 💻 Python API Usage

You can also use the backend pipeline directly in Python scripts or Jupyter notebooks:

```python
from backend.predictions import PredictionService
from backend.screener import StockScreener

# Initialize service
service = PredictionService(lookback_days=365)

# Predict single stock
result = service.predict_stock("RELIANCE.NS", analysis_mode="cache")
print(f"Signal: {result['signal']} ({result['confidence']*100:.1f}% confidence)")
print(f"Latest Price: ₹{result['latest_price']}")
print(f"Top Indicator Driver: {result['top_features'][0]['name']}")

# Run screener
screener = StockScreener(service)
screener_results = screener.run_screener(min_confidence=0.60)
print(f"Found {len(screener_results['bullish'])} high-conviction bullish stocks")
```

---

## 📁 Repository Structure

```
├── backend/                       # Quantitative ML pipeline
│   ├── data_ingestion.py         # Multi-format data loader & Yahoo Finance fetcher
│   ├── feature_engineering.py    # 20+ technical indicators (RSI, MACD, BB, ATR, ADX)
│   ├── models.py                 # XGBoost and Random Forest model wrappers
│   ├── explainability.py         # TreeSHAP & feature attribution
│   ├── backtesting.py            # Quantitative backtesting engine
│   ├── portfolio_optimization.py # Modern Portfolio Theory & Sharpe optimization
│   ├── regime_detection.py       # Market regime & volatility regime analysis
│   ├── predictions.py            # Main prediction orchestrator
│   ├── screener.py               # NIFTY 50 multi-stock screening engine
│   ├── sector_analysis.py        # Sector breadth & rotation analytics
│   └── utils.py                  # Ticker mappings, helpers & utilities
│
├── frontend/                      # Web user interface
│   ├── web_app.py                # FastAPI web service & REST endpoints
│   ├── templates/
│   │   └── index.html            # Responsive dark-theme dashboard
│   └── static/
│       ├── css/styles.css        # Custom UI styling & components
│       ├── js/app.js             # Reactive charting & API client
│       └── favicon.svg           # Application icon
│
├── data/
│   └── external_nifty50/         # Pre-downloaded constituent stock datasets
│
├── docs/
│   └── screenshots/              # High-resolution UI showcase images
│
├── models/
│   └── trained_models/           # Pre-trained XGBoost models and scalers
│
├── scripts/
│   └── capture_screenshots.py    # Headless Playwright UI capture tool
│
├── tests/                         # Test suite
│   ├── test_api.py               # REST API endpoint tests
│   ├── test_integration.py       # End-to-end integration workflows
│   ├── test_nse_cleaning.py      # NSE data sanitization & date parsing tests
│   └── test_utils.py             # Utility & normalization unit tests
│
├── ARCHITECTURE.md                # Detailed pipeline & architectural design
├── INSTALL.md                     # Complete cross-platform installation guide
├── QUICKSTART.md                  # Quick usage and API guide
├── PROJECT_DEEP_DIVE.md           # Theoretical & algorithmic deep dive
├── requirements.txt               # Dependencies
├── run.bat                        # Windows launcher
├── run.sh                         # Linux/macOS launcher
└── README.md                      # Project documentation
```

---

## 🧪 Testing

The repository includes a comprehensive test suite covering data parsing, feature normalization, model inference, screener thresholds, and FastAPI endpoints:

```bash
pytest -v
```

```
tests/test_api.py ........                                       [ 32%]
tests/test_integration.py .........                              [ 68%]
tests/test_nse_cleaning.py ..                                    [ 76%]
tests/test_utils.py ......                                       [100%]

========================= 25 passed in 29.95s =========================
```

---

## 📚 Documentation Links

- **[QUICKSTART.md](QUICKSTART.md)**: Usage examples and guide
- **[INSTALL.md](INSTALL.md)**: Full platform-specific setup
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Architectural diagram and design decisions
- **[PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md)**: Theoretical & quantitative deep dive

---

## ⚠️ Disclaimer

This application is strictly for **educational and research purposes**. It is not financial advice. Machine learning models predict statistical patterns and cannot guarantee future market returns. Always perform your own due diligence before trading.

---

## 📄 License

Distributed under the [MIT License](LICENSE).

# 📊 NIFTY 50 Stock Prediction & Analysis System

A lightweight, portable machine learning system for predicting short-term stock direction in the NIFTY 50 using technical indicators, XGBoost models, and comprehensive backtesting.

**Status:** Ready to deploy ✅
**Last Updated:** April 2026
**Python Version:** 3.9+

---

## 🎯 Key Features

### Stock Direction Prediction
- Analyzes historical OHLCV data from Yahoo Finance
- Generates 20+ technical features (RSI, MACD, moving averages, volatility)
- Trains XGBoost classifier for BULLISH vs BEARISH predictions
- Confidence scores for each prediction

### Explainable AI
- SHAP-based feature importance analysis
- Clear explanation of prediction drivers
- Model transparency for investment decisions

### Backtesting & Validation
- Compares moving-average and RSI strategies
- Reports return, Sharpe ratio, max drawdown
- Risk metrics for informed trading

### Portfolio Tools
- Modern Portfolio Theory optimization
- Correlation analysis across stocks
- Sector-wise risk breakdown

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- **Python 3.9+** ([download](https://www.python.org/downloads/))
- **4GB RAM** (8GB+ recommended)
- **Internet connection**

### Installation

**Windows (PowerShell):**
```powershell
# Navigate to project folder
cd "C:\path\to\nifty50_prediction_system"

# Run setup script
.\run.bat
```

**Mac/Linux (Bash):**
```bash
cd /path/to/nifty50_prediction_system

# Make script executable
chmod +x run.sh

# Run setup script
./run.sh
```

### Access Dashboard
Open: **http://localhost:8501**

---

## 📁 Project Structure

```
├── backend/                    # Core ML pipeline
│   ├── data_ingestion.py      # Fetch stock data
│   ├── feature_engineering.py # Technical indicators
│   ├── models.py              # XGBoost, Random Forest
│   ├── explainability.py      # SHAP analysis
│   ├── backtesting.py         # Strategy testing
│   ├── portfolio_optimization.py
│   ├── regime_detection.py    # Market regimes
│   ├── predictions.py         # Main orchestration
│   └── utils.py
│
├── frontend/                   # Web UI (FastAPI + Bootstrap)
│   ├── web_app.py            # FastAPI app
│   ├── templates/            # HTML templates
│   └── static/               # CSS, JS, assets
│
├── data/              # Data storage
│   ├── external_nifty50/     # Pre-downloaded stock data
│   ├── raw/
│   └── processed/
│
├── tests/            # Unit & integration tests
├── requirements.txt  # Dependencies
├── INSTALL.md       # Detailed setup guide
├── QUICKSTART.md    # Usage examples
└── README.md        # This file
```

---

## 💻 Usage Examples

### 1. Get Stock Prediction via Python

```python
from backend.predictions import PredictionService

service = PredictionService()
result = service.predict_stock('RELIANCE.NS')

# Access prediction
decision = result['predictions']['decision']  # 'BULLISH' or 'BEARISH'
confidence = result['predictions']['confidence']  # 0.0-1.0

# Get technical indicators
rsi = result['technical_indicators']['rsi_14']
macd = result['technical_indicators']['macd']

# Get backtesting results
backtest = result['backtest_results']
print(f"Return: {backtest['ma_strategy']['return']:.2f}%")
print(f"Sharpe: {backtest['ma_strategy']['sharpe_ratio']:.2f}")
```

### 2. Web Dashboard

1. Navigate to http://localhost:8501
2. Select a stock from the NIFTY 50 list
3. Click **Analyze**
4. View predictions, charts, and metrics

### 3. Portfolio Analysis

```python
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS']
portfolio = service.analyze_portfolio(tickers)

optimal = portfolio['optimizations']['max_sharpe']
print(f"Expected Return: {optimal['return']:.2f}%")
```

---

## 🔧 Configuration

Create a `.env` file for custom settings (optional):

```bash
ML_USE_GPU=auto           # auto, 1/true, 0/false
ML_RANDOM_SEED=42
DATA_LOOKBACK_DAYS=365
API_PORT=8501
```

---

## 📦 Dependencies

**Production (24 packages):**
- Data: pandas, numpy, scipy
- Web: FastAPI, Uvicorn
- ML: scikit-learn, XGBoost, SHAP
- Fetching: yfinance, requests

Installation handled automatically by `run.bat` or `run.sh`.

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Python not found" | Install Python 3.9+ and add to PATH |
| "Port 8501 in use" | Use different port: `--port 8502` |
| "ModuleNotFoundError" | Activate virtual environment |

See [INSTALL.md](INSTALL.md) for detailed help.

---

## 📚 Documentation

- **[INSTALL.md](INSTALL.md)** - Complete installation guide
- **[QUICKSTART.md](QUICKSTART.md)** - Code examples
- **[DISTRIBUTION.md](DISTRIBUTION.md)** - Sharing & deployment
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. It should not be used as the sole basis for investment decisions. Always consult with financial advisors before trading.

---

## 📄 License

See [LICENSE](LICENSE)

---


---

**Ready to analyze NIFTY 50 stocks? Start with `./run.bat` (Windows) or `./run.sh` (Mac/Linux)! 🚀**


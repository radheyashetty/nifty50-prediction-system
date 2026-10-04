# 🏗️ Architecture & Pipeline Overview — NIFTY 50 Stock Lab

This document outlines the architecture, data flow, and modeling pipeline of the NIFTY 50 Stock Prediction & Analysis System.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │      Web Dashboard (FastAPI + Bootstrap)    │
                    │  Dashboard · Screener · Sectors · Compare    │
                    │      Deep Dive · Strategy Backtesting        │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP / JSON API
                    ┌──────────────────────▼───────────────────────┐
                    │               PredictionService              │
                    │  (Orchestrator: Ingestion, Models, SHAP)    │
                    └──────┬───────────────┬───────────────┬───────┘
                           │               │               │
       ┌───────────────────▼──┐   ┌────────▼─────────┐   ┌─▼──────────────────┐
       │   Data Ingestion     │   │ Feature Engineer │   │   Backtest Engine  │
       │  - Yahoo Finance API │   │ - 20+ Indicators │   │ - ML Strategy      │
       │  - Offline CSV Cache │   │ - RSI, MACD, BB  │   │ - MA Crossover     │
       │  - NSE Format Parser │   │ - ATR, OBV, ADX  │   │ - RSI & Buy & Hold │
       └──────────────────────┘   └──────────────────┘   └────────────────────┘
                                           │
                                  ┌────────▼─────────┐
                                  │   Model Pipeline │
                                  │ - XGBoost Class. │
                                  │ - Random Forest  │
                                  │ - SHAP Explainer │
                                  └──────────────────┘
```

---

## 🔄 End-to-End Pipeline

1. **Data Ingestion (`backend/data_ingestion.py`)**
   - Retrieves daily OHLCV prices from Yahoo Finance with resilient fallback to local pre-downloaded CSV datasets.
   - Cleans NSE trading date formats (`DD-MM-YYYY` and ISO), handles non-trading days, stock splits, and missing values.

2. **Feature Engineering (`backend/feature_engineering.py`)**
   - Generates technical indicators across trend, momentum, volatility, and volume:
     - Momentum: RSI (14), Stochastic oscillator (%K, %D), Rate of Change (ROC)
     - Trend: MACD (12, 26, 9), EMA/SMA crossovers (20, 50), ADX
     - Volatility: Bollinger Bands (%B, Bandwidth), Average True Range (ATR)
     - Volume: On-Balance Volume (OBV), Volume 20-day moving average ratio
     - Returns: Lagged multi-day returns, forward 5-day directional return target (> 1.5% = Bullish).

3. **Model Training & Inference (`backend/models.py`, `backend/predictions.py`)**
   - **XGBoost Classifier**: Configured with early stopping, regularization, and probability calibration.
   - **Random Forest Classifier**: Ensemble companion providing stability and variance reduction.
   - **Ensemble Blend**: Weighted probability aggregation for robust directional forecasts.
   - Pre-trained models stored in `models/trained_models/` for sub-second inference.

4. **Model Explainability (`backend/explainability.py`)**
   - TreeSHAP calculates feature contributions for every prediction.
   - Native tree gain/weight fallback when compiled tree SHAP is unavailable.
   - Classifies each driving indicator as bullish or bearish relative to historical baselines.

5. **Strategy Backtesting (`backend/backtesting.py`)**
   - Evaluates historical performance of model signals versus traditional quantitative strategies:
     - ML Directional Strategy
     - Dual Moving Average Crossover (20 / 50 SMA)
     - RSI Overbought/Oversold Mean Reversion (30 / 70)
     - Buy & Hold Benchmark
   - Calculates Total Return, Sharpe Ratio, Maximum Drawdown, Win Rate, and Calmar Ratio.

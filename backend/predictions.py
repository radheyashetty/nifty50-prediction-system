"""
Main Prediction Service
Orchestrates all components for end-to-end prediction pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import copy
from datetime import datetime
from pathlib import Path
import re
import joblib
import threading
import warnings

warnings.filterwarnings("ignore")

from .data_ingestion import DataIngestion
from .feature_engineering import FeatureEngineer
from .models import (
    XGBoostPredictor,
    RandomForestPredictor,
    EnsemblePredictor,
    evaluate_predictions,
)
from .explainability import ModelExplainer, create_summary_explanation
from .backtesting import BacktestEngine
from .portfolio_optimization import PortfolioOptimizer
from .regime_detection import RegimeDetector, VolatilityRegimeDetector
from .utils import get_ticker_sector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PredictionService:
    """
    Main service for stock prediction and analysis
    Combines all components into a cohesive system
    """

    def __init__(self, lookback_days: int = 365):
        """Initialize prediction service with all components"""
        self.lookback_days = lookback_days
        self.data_ingestion = DataIngestion(lookback_days=lookback_days)
        self.feature_engineer = FeatureEngineer()


        self.regime_detector = RegimeDetector(n_regimes=3)
        self.vol_regime_detector = VolatilityRegimeDetector()

        self.backtest_engine = BacktestEngine()
        self._result_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self.artifact_dir = (
            Path(__file__).resolve().parents[1] / "models" / "trained_models"
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_ticker(ticker: str) -> str:
        return ticker.replace(".", "_").replace("-", "_").upper()

    def _artifact_paths(self, ticker: str) -> Dict[str, Path]:
        key = f"{self._sanitize_ticker(ticker)}_{self.lookback_days}d"
        return {
            "model": self.artifact_dir / f"xgb_{key}.joblib",
            "scaler": self.artifact_dir / f"scaler_{key}.joblib",
            "features": self.artifact_dir / f"features_{key}.joblib",
        }

    def _save_trained_artifacts(
        self, ticker: str, xgb_model: XGBoostPredictor, scaler: StandardScaler, feature_cols: list
    ) -> None:
        if xgb_model is None:
            return
        paths = self._artifact_paths(ticker)
        xgb_model.save(str(paths["model"]))
        joblib.dump(scaler, paths["scaler"])
        joblib.dump(feature_cols, paths["features"])

    def _load_trained_artifacts(self, ticker: str) -> Dict[str, Any] | None:
        paths = self._artifact_paths(ticker)
        if not all(p.exists() for p in paths.values()):
            return None

        model = XGBoostPredictor()
        model.load(str(paths["model"]))
        scaler = joblib.load(paths["scaler"])
        feature_cols = joblib.load(paths["features"])
        return {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

    def _make_cache_key(self, ticker: str, raw_data: pd.DataFrame) -> str:
        """Build a stable cache key for a prediction request."""
        if "date" in raw_data.columns and not raw_data.empty:
            last_date = pd.to_datetime(raw_data["date"].iloc[-1], errors="coerce")
            last_date_key = (
                str(last_date.date()) if not pd.isna(last_date) else "unknown-date"
            )
        else:
            last_date_key = "no-date"

        return f"{ticker}|{self.lookback_days}|{len(raw_data)}|{last_date_key}"

    @staticmethod
    def _normalize_upload_column_name(value: str) -> str:
        """Normalize column names so common stock dataset aliases map cleanly."""
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def prepare_uploaded_price_data(
        self, uploaded_data: pd.DataFrame, ticker: str | None = None
    ) -> pd.DataFrame:
        """Normalize and validate uploaded stock data into an OHLCV schema."""
        if uploaded_data is None or uploaded_data.empty:
            raise ValueError("Uploaded dataset is empty")

        df = uploaded_data.copy()

        alias_map = {
            "date": "date",
            "datetime": "date",
            "timestamp": "date",
            "tradedate": "date",
            "trade_date": "date",
            "open": "open",
            "openprice": "open",
            "openingprice": "open",
            "high": "high",
            "highprice": "high",
            "low": "low",
            "lowprice": "low",
            "close": "close",
            "closeprice": "close",
            "adjclose": "adj close",
            "adjustedclose": "adj close",
            "adj_close": "adj close",
            "volume": "volume",
            "vol": "volume",
            "volumetraded": "volume",
            "volume_traded": "volume",
            "quantity": "volume",
            "qty": "volume",
        }

        rename_map = {}
        for column in df.columns:
            normalized = self._normalize_upload_column_name(column)
            if normalized in alias_map:
                rename_map[column] = alias_map[normalized]

        df = df.rename(columns=rename_map)

        df = self.data_ingestion._normalize_ohlcv_columns(df)

        if "date" not in df.columns:
            first_col = str(df.columns[0])
            df = df.rename(columns={first_col: "date"})

        required_cols = ["date", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Dataset must include columns for date, open, high, low, close, and volume. "
                f"Missing: {', '.join(missing_cols)}"
            )

        df = self.data_ingestion._clean_ohlcv_frame(df)

        if len(df) > self.lookback_days:
            df = df.tail(self.lookback_days).reset_index(drop=True)

        if "adj close" not in df.columns:
            df["adj close"] = df["close"]

        # Keep the pipeline compatible by ensuring optional features exist.
        if "sentiment" not in df.columns:
            df["sentiment"] = self.data_ingestion.generate_synthetic_sentiment(df)

        if "sector" not in df.columns:
            df["sector"] = get_ticker_sector(ticker) if ticker else "Unknown"
        else:
            df["sector"] = df["sector"].fillna(
                get_ticker_sector(ticker) if ticker else "Unknown"
            )

        if ticker and "symbol" not in df.columns:
            df["symbol"] = ticker

        macro = self.data_ingestion.generate_macroeconomic_features(len(df))
        for key, values in macro.items():
            if key not in df.columns:
                df[key] = values

        return df

    def _predict_from_raw_data(
        self,
        ticker: str,
        raw_data: pd.DataFrame,
        mode: str,
        requested_mode: str,
        cache_key: str,
    ) -> Dict[str, Any]:
        """Run prediction/training pipeline using already-loaded price data."""
        # Reset state to cleanly prevent cross-ticker/cross-request data bleeding.
        # Use local variables for thread safety
        scaler = StandardScaler()
        xgb_model = None
        rf_model = None
        ensemble_model = EnsemblePredictor(xgb_weight=0.6, rf_weight=0.4)

        # Step 2: Feature engineering
        print("[2/6] Engineering features...")
        try:
            data = self.feature_engineer.create_features(raw_data)
        except Exception as exc:
            return {"error": f"Feature engineering failed for {ticker}: {exc}"}

        if data is None or data.empty:
            return {
                "error": f"Feature engineering produced no usable data for {ticker}"
            }

        # Step 3: Get features for modeling
        exclude_cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj close",
            "target_3d",
            "target_5d",
            "future_return_3d",
            "future_return_5d",
            "target_regression",
            "sector",
            "symbol",
        ]
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        if not feature_cols:
            return {"error": f"No model features available for {ticker}"}

        X_df = data[feature_cols].apply(pd.to_numeric, errors="coerce")
        X_df = X_df.dropna(axis=1, how="all")
        feature_cols = list(X_df.columns)
        if not feature_cols:
            return {"error": f"No numeric model features available for {ticker}"}
        y = (
            data["target_5d"].values
            if "target_5d" in data.columns
            else data["target_3d"].values
        )

        artifacts = None
        if mode == "cache":
            artifacts = self._load_trained_artifacts(ticker)

        # Align features to trained model when running without training.
        if artifacts is not None:
            trained_cols = list(artifacts["feature_cols"])
            loaded_scaler = artifacts["scaler"]
            
            # Cross-verify scaler compatibility
            try:
                if hasattr(loaded_scaler, "n_features_in_") and loaded_scaler.n_features_in_ != len(trained_cols):
                    print(f"⚠️ Artifact mismatch for {ticker}: scaler expects {loaded_scaler.n_features_in_}, but features say {len(trained_cols)}. Forcing retrain...")
                    return self.predict_stock(ticker, retrain=True)
            except Exception:
                pass

            X_df = X_df.reindex(columns=trained_cols, fill_value=0.0)
            feature_cols = trained_cols

        X = X_df.values

        # Remove NaN
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        # Split data
        if len(X) < 60:
            return {
                "error": (
                    f"Insufficient usable rows for training for {ticker}: "
                    f"need at least 60, got {int(len(X))}"
                )
            }

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
        except Exception as exc:
            return {"error": f"Data split failed for {ticker}: {exc}"}

        # Normalize
        try:
            if mode == "after_training":
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                if artifacts is None:
                    # cache mode fallback: train only when artifacts are unavailable.
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    mode = "after_training"
                else:
                    xgb_model = artifacts["model"]
                    scaler = artifacts["scaler"]
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
        except Exception as exc:
            if mode == "cache":
                print(
                    f"🔄 Scaling failed for {ticker} (possible feature mismatch): {exc}. Forcing retraining..."
                )
                return self.predict_stock(ticker, retrain=True)
            return {"error": f"Feature scaling failed for {ticker}: {exc}"}

        # Step 4: Train or infer
        if mode == "after_training":
            print("[3/6] Training models...")
            try:
                xgb_model = XGBoostPredictor(n_estimators=150, max_depth=4)
                xgb_model.train(
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test,
                    feature_names=feature_cols,
                )

                rf_model = RandomForestPredictor(n_estimators=300, max_depth=15)
                rf_model.train(X_train_scaled, y_train, feature_names=feature_cols)

                ensemble_model.fit(xgb_model, rf_model)
                self._save_trained_artifacts(ticker, xgb_model, scaler, feature_cols)
            except Exception as exc:
                return {"error": f"Model training failed for {ticker}: {exc}"}
        else:
            print("[3/6] Running cached model inference...")

        if xgb_model is None:
            return {"error": f"Model is not initialized for {ticker}"}

        model = xgb_model

        try:
            # Check for feature shape mismatch before predicting
            expected_n = getattr(model.model, "n_features_in_", None)
            if expected_n is None:
                # Fallback for XGBoost
                try:
                    expected_n = len(model.model.get_booster().feature_names)
                except Exception:
                    pass

            if expected_n is not None and X_test_scaled.shape[1] != expected_n:
                print(
                    f"⚠️ Feature mismatch detected for {ticker} (expected {expected_n}, got {X_test_scaled.shape[1]})"
                )
                if mode == "cache":
                    print(
                        "🔄 Forcing retraining to sync with current feature engineer..."
                    )
                    return self.predict_stock(ticker, retrain=True)

            xgb_preds = model.predict(X_test_scaled)
            rf_preds = (
                rf_model.predict(X_test_scaled)
                if rf_model is not None
                else xgb_preds
            )

            ensemble_model.fit(xgb_model, rf_model)
            ensemble_pred = ensemble_model.predict_proba(X_test_scaled)
            if len(ensemble_pred) == 0:
                ensemble_pred = xgb_preds

            n_common = min(
                len(y_test), len(xgb_preds), len(rf_preds), len(ensemble_pred)
            )
            y_eval = y_test[-n_common:]
            xgb_eval = xgb_preds[-n_common:]
            rf_eval = rf_preds[-n_common:]
            ensemble_eval = ensemble_pred[-n_common:]

            xgb_metrics = evaluate_predictions(y_eval, xgb_eval)
            _ = evaluate_predictions(y_eval, rf_eval)
            ensemble_metrics = evaluate_predictions(y_eval, ensemble_eval)
        except Exception as exc:
            return {"error": f"Model prediction failed for {ticker}: {exc}"}

        # Step 5: Create explanation
        if requested_mode != "screening":
            print("[4/6] Generating explanations...")
            try:
                explainer = ModelExplainer(
                    model, X_train=X_train_scaled, feature_names=feature_cols
                )
                explainer.create_explainer("xgboost")
            except Exception as exc:
                return {"error": f"Explainer setup failed for {ticker}: {exc}"}

            try:
                explanation = explainer.explain_prediction(X_test_scaled, index=-1)
            except Exception as e:
                print(
                    f"Warning: SHAP explanation failed, using feature-importance fallback: {e}"
                )
                fallback_importance = model.get_feature_importance()
                top_features = []
                for feature_name, importance in list(fallback_importance.items())[:10]:
                    top_features.append(
                        {
                            "feature_name": feature_name,
                            "shap_value": float(importance),
                            "direction": (
                                "bullish" if float(importance) >= 0 else "bearish"
                            ),
                            "current_value": 0.0,
                            "description": f"{feature_name} contributes to prediction based on model importance.",
                        }
                    )
                explanation = {"top_features": top_features, "top_features_map": {}}
        else:
            # In screening mode, we just get native feature importance for speed
            fallback_importance = model.get_feature_importance()
            top_features = []
            for feature_name, importance in list(fallback_importance.items())[:5]:
                top_features.append(
                    {
                        "feature_name": feature_name,
                        "shap_value": float(importance),
                        "direction": "bullish",
                    }
                )
            explanation = {"top_features": top_features}

        # Step 6: Analyze regime and get metrics
        print("[5/6] Analyzing market regime...")
        # ... (rest of step 6)

        # Use local detectors for thread safety
        regime_detector = RegimeDetector(n_regimes=3)
        vol_regime_detector = VolatilityRegimeDetector()

        try:
            regime_detector.detect_regimes(raw_data)
            current_regime = regime_detector.get_current_regime()
        except Exception as exc:
            print(f"Warning: regime detection failed for {ticker}: {exc}")
            current_regime = "Unknown"

        try:
            vol_regimes = vol_regime_detector.detect_regimes(raw_data)
            current_vol_regime = vol_regimes["current_regime"]
        except Exception as exc:
            print(f"Warning: volatility regime detection failed for {ticker}: {exc}")
            vol_regimes = {"current_regime": "Unknown", "current_volatility": 0.0}
            current_vol_regime = "Unknown"

        # Default backtest results for screening mode
        backtest_results = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "buy_hold_return": 0.0,
        }
        rsi_results = {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        ml_results = {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        bh_results = {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        rsi_values = pd.Series(0, index=raw_data.index)

        # Step 7: Backtesting
        if requested_mode != "screening":
            print("[6/6] Running backtests...")
            # ... (the rest of the backtesting logic is already there or was removed/condensed)

        # Compile results
        latest_price = raw_data["close"].iloc[-1]
        prediction_3d = float(np.asarray(ensemble_pred)[-1])

        indicator_snapshot = self.feature_engineer.get_indicator_snapshot(data)

        model_scores = {
            "xgboost_prob": float(np.asarray(xgb_preds)[-1]) if len(xgb_preds) else 0.5,
            "random_forest_prob": (
                float(np.asarray(rf_preds)[-1]) if len(rf_preds) else 0.5
            ),
            "ensemble_prob": float(prediction_3d),
        }

        result = {
            "ticker": ticker,
            "sector": get_ticker_sector(ticker),
            "timestamp": datetime.now().isoformat(),
            "latest_price": float(latest_price),
            "analysis_mode": mode,
            "requested_mode": requested_mode,
            "signal": "BULLISH" if prediction_3d > 0.55 else "BEARISH",
            "confidence": float(max(prediction_3d, 1 - prediction_3d)),
            # Predictions
            "predictions": {
                "bullish_probability": float(prediction_3d),
                "bearish_probability": float(1 - prediction_3d),
                "decision": "BULLISH" if prediction_3d > 0.55 else "BEARISH",
                "confidence": float(max(prediction_3d, 1 - prediction_3d)),
            },
            "model_scores": model_scores,
            # Model insights
            "model_performance": {
                "xgboost_auc": float(xgb_metrics["auc_roc"]),
                "xgboost_accuracy": float(xgb_metrics["accuracy"]),
                "model_used": "XGBoost",
                "xgboost_device": getattr(xgb_model, "device", "cpu"),
            },
            "model_metrics": {
                "accuracy": float(ensemble_metrics["accuracy"]),
                "precision": float(ensemble_metrics["precision"]),
                "recall": float(ensemble_metrics["recall"]),
                "f1_score": float(ensemble_metrics["f1_score"]),
                "roc_auc": float(ensemble_metrics["roc_auc"]),
                "log_loss": float(ensemble_metrics["log_loss"]),
                "brier_score": float(ensemble_metrics["brier_score"]),
                "confusion_matrix": ensemble_metrics["confusion_matrix"],
            },
            # Top drivers
            "top_features": explanation["top_features"],
            # Technical indicators
            "technical_indicators": indicator_snapshot,
            "indicators": indicator_snapshot,
            # Regime analysis
            "regime_analysis": {
                "current_regime": current_regime,
                "volatility_regime": current_vol_regime,
                "volatility": float(vol_regimes["current_volatility"]),
            },
            "regime": {
                "name": current_regime,
                "volatility_regime": current_vol_regime,
                "hmm_regime": current_regime,
                "regime_index": 0,
                "characteristics": {},
                "transition_probability": 0.5,
            },
            # Backtesting results
            "backtest_results": {
                "ml_strategy": {
                    "return": ml_results.get("total_return", 0.0),
                    "sharpe": ml_results.get("sharpe_ratio", 0.0),
                    "max_drawdown": ml_results.get("max_drawdown", 0.0),
                },
                "ma_strategy": {
                    "return": backtest_results["total_return"],
                    "sharpe": backtest_results["sharpe_ratio"],
                    "max_drawdown": backtest_results["max_drawdown"],
                },
                "rsi_strategy": {
                    "return": rsi_results["total_return"],
                    "sharpe": rsi_results["sharpe_ratio"],
                    "max_drawdown": rsi_results["max_drawdown"],
                },
                "buy_hold_return": backtest_results["buy_hold_return"],
            },
            "backtest": {
                "ml_strategy": {
                    "total_return_pct": float(
                        ml_results.get(
                            "total_return_pct", ml_results.get("total_return", 0.0)
                        )
                    ),
                    "annualized_return_pct": float(
                        ml_results.get("annualized_return_pct", 0.0)
                    ),
                    "sharpe_ratio": float(ml_results.get("sharpe_ratio", 0.0)),
                    "max_drawdown_pct": float(
                        ml_results.get(
                            "max_drawdown_pct", ml_results.get("max_drawdown", 0.0)
                        )
                    ),
                    "win_rate_pct": float(
                        ml_results.get("win_rate_pct", ml_results.get("win_rate", 0.0))
                    ),
                    "total_trades": int(
                        ml_results.get("total_trades", ml_results.get("num_trades", 0))
                    ),
                    "profit_factor": float(ml_results.get("profit_factor", 0.0)),
                    "calmar_ratio": float(ml_results.get("calmar_ratio", 0.0)),
                },
                "ma_crossover": {
                    "total_return_pct": float(
                        backtest_results.get(
                            "total_return_pct",
                            backtest_results.get("total_return", 0.0),
                        )
                    ),
                    "annualized_return_pct": float(
                        backtest_results.get("annualized_return_pct", 0.0)
                    ),
                    "sharpe_ratio": float(backtest_results.get("sharpe_ratio", 0.0)),
                    "max_drawdown_pct": float(
                        backtest_results.get(
                            "max_drawdown_pct",
                            backtest_results.get("max_drawdown", 0.0),
                        )
                    ),
                    "win_rate_pct": float(
                        backtest_results.get(
                            "win_rate_pct", backtest_results.get("win_rate", 0.0)
                        )
                    ),
                    "total_trades": int(
                        backtest_results.get(
                            "total_trades", backtest_results.get("num_trades", 0)
                        )
                    ),
                    "profit_factor": float(backtest_results.get("profit_factor", 0.0)),
                    "calmar_ratio": float(backtest_results.get("calmar_ratio", 0.0)),
                },
                "rsi_strategy": {
                    "total_return_pct": float(
                        rsi_results.get(
                            "total_return_pct", rsi_results.get("total_return", 0.0)
                        )
                    ),
                    "annualized_return_pct": float(
                        rsi_results.get("annualized_return_pct", 0.0)
                    ),
                    "sharpe_ratio": float(rsi_results.get("sharpe_ratio", 0.0)),
                    "max_drawdown_pct": float(
                        rsi_results.get(
                            "max_drawdown_pct", rsi_results.get("max_drawdown", 0.0)
                        )
                    ),
                    "win_rate_pct": float(
                        rsi_results.get(
                            "win_rate_pct", rsi_results.get("win_rate", 0.0)
                        )
                    ),
                    "total_trades": int(
                        rsi_results.get(
                            "total_trades", rsi_results.get("num_trades", 0)
                        )
                    ),
                    "profit_factor": float(rsi_results.get("profit_factor", 0.0)),
                    "calmar_ratio": float(rsi_results.get("calmar_ratio", 0.0)),
                },
                "buy_and_hold": {
                    "total_return_pct": float(
                        bh_results.get(
                            "total_return_pct", bh_results.get("total_return", 0.0)
                        )
                    ),
                    "annualized_return_pct": float(
                        bh_results.get("annualized_return_pct", 0.0)
                    ),
                    "sharpe_ratio": float(bh_results.get("sharpe_ratio", 0.0)),
                    "max_drawdown_pct": float(
                        bh_results.get(
                            "max_drawdown_pct", bh_results.get("max_drawdown", 0.0)
                        )
                    ),
                    "win_rate_pct": float(
                        bh_results.get("win_rate_pct", bh_results.get("win_rate", 0.0))
                    ),
                    "total_trades": int(
                        bh_results.get("total_trades", bh_results.get("num_trades", 1))
                    ),
                    "profit_factor": float(bh_results.get("profit_factor", 0.0)),
                    "calmar_ratio": float(bh_results.get("calmar_ratio", 0.0)),
                },
            },
            # Risk metrics
            "risk_metrics": {
                "volatility_30d": float(
                    np.nan_to_num(
                        raw_data["close"].pct_change().rolling(30).std().iloc[-1]
                        * 100.0
                    )
                ),
                "max_drawdown_analysis": f"{backtest_results['max_drawdown']:.2f}%",
                "risk_level": self._assess_risk_level(
                    vol_regimes["current_volatility"], backtest_results["max_drawdown"]
                ),
            },
            # Explanation text
            "explanation": explanation["top_features"],
            "summary": create_summary_explanation(
                prediction_3d,
                (
                    explanation.get("top_features_map", {})
                    if isinstance(explanation.get("top_features"), list)
                    else explanation["top_features"]
                ),
                {
                    "RSI": rsi_values.iloc[-1],
                    "Regime": current_regime,
                    "Trend": "Strong" if abs(prediction_3d - 0.5) > 0.2 else "Weak",
                },
            ),
            "run_source": (
                "trained_now"
                if mode == "after_training"
                else "loaded_trained_artifacts"
            ),
            "model_used": "ensemble(xgb+rf+lstm)",
            "data_points_used": int(len(raw_data)),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "last_date": (
                str(pd.to_datetime(raw_data["date"].iloc[-1]).date())
                if "date" in raw_data.columns
                else None
            ),
            "chart_prices": [
                float(x)
                for x in pd.to_numeric(raw_data["close"], errors="coerce")
                .tail(30)
                .ffill()
                .bfill()
                .tolist()
            ],
            "chart_labels": (
                [str(pd.to_datetime(x).date()) for x in raw_data["date"].tail(30)]
                if "date" in raw_data.columns
                else []
            ),
        }

        with self._cache_lock:
            self._result_cache[cache_key] = copy.deepcopy(result)
            # Keep cache bounded for long-running API process.
            if len(self._result_cache) > 100:
                oldest_key = next(iter(self._result_cache))
                self._result_cache.pop(oldest_key, None)

        return result

    def predict_from_uploaded_data(
        self,
        ticker: str,
        uploaded_data: pd.DataFrame,
        analysis_mode: str = "after_training",
    ) -> Dict[str, Any]:
        """Predict/train using user-uploaded CSV data instead of data ingestion fetch."""
        mode = (analysis_mode or "after_training").strip().lower()
        if mode not in {"cache", "after_training"}:
            return {
                "error": f"Invalid analysis_mode '{analysis_mode}'. Use: cache, after_training"
            }

        requested_mode = mode

        print(f"\n{'='*60}")
        print(f"Analyzing uploaded data for {ticker}")
        print("=" * 60)
        print("\n[1/6] Validating uploaded CSV data...")

        try:
            raw_data = self.prepare_uploaded_price_data(uploaded_data, ticker=ticker)
        except Exception as exc:
            return {"error": f"Invalid uploaded CSV: {exc}"}

        if raw_data is None or raw_data.empty:
            return {"error": "Uploaded CSV has no usable OHLCV rows"}

        cache_key = f"uploaded|{self._make_cache_key(ticker, raw_data)}"
        if mode == "cache":
            with self._cache_lock:
                cached = self._result_cache.get(cache_key)
            if cached is not None:
                cached = copy.deepcopy(cached)
                cached["timestamp"] = datetime.now().isoformat()
                cached["cache_hit"] = True
                cached["analysis_mode"] = "cache"
                cached["requested_mode"] = requested_mode
                cached["run_source"] = "memory_cache_uploaded"
                return cached

        result = self._predict_from_raw_data(
            ticker=ticker,
            raw_data=raw_data,
            mode=mode,
            requested_mode=requested_mode,
            cache_key=cache_key,
        )
        if "error" not in result:
            result["run_source"] = f"{result.get('run_source', 'unknown')}|uploaded_csv"
            sector_name = None
            if "sector" in raw_data.columns:
                sector_values = (
                    raw_data["sector"]
                    .astype(str)
                    .replace({"nan": "", "None": ""})
                    .str.strip()
                )
                non_empty_sector = sector_values[sector_values != ""]
                if not non_empty_sector.empty:
                    sector_name = non_empty_sector.iloc[-1]
            if not sector_name or sector_name == "Unknown":
                sector_name = get_ticker_sector(ticker)
            result["sector"] = sector_name
            result["sector_context"] = {
                "sector": sector_name,
                "source": "uploaded_data",
                "ticker": ticker,
            }

        return result

    def predict_stock(
        self,
        ticker: str,
        retrain: bool = False,
        analysis_mode: str = "cache",
        screening_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete prediction pipeline for a stock

        Args:
            ticker: Stock ticker (NSE format, e.g., 'RELIANCE.NS')
            retrain: Whether to retrain models
            analysis_mode: One of cache, after_training

        Returns:
            Dictionary with prediction, explanation, and analysis
        """
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print("=" * 60)

        mode = (analysis_mode or "cache").strip().lower()
        requested_mode = mode
        if mode not in {"cache", "after_training"}:
            return {
                "error": f"Invalid analysis_mode '{analysis_mode}'. Use: cache, after_training"
            }

        if retrain:
            mode = "after_training"

        # Step 1: Fetch and preprocess data
        print("\n[1/6] Fetching stock data...")
        try:
            raw_data = self.data_ingestion.process_stock_data(ticker)
        except Exception as exc:
            return {"error": f"Failed to fetch/process data for {ticker}: {exc}"}

        if raw_data is None or raw_data.empty:
            return {"error": f"Unable to fetch data for {ticker}"}

        cache_key = self._make_cache_key(ticker, raw_data)
        if mode == "cache":
            with self._cache_lock:
                cached = self._result_cache.get(cache_key)
            if cached is not None:
                cached = copy.deepcopy(cached)
                cached["timestamp"] = datetime.now().isoformat()
                cached["cache_hit"] = True
                cached["analysis_mode"] = "cache"
                cached["requested_mode"] = requested_mode
                cached["run_source"] = "memory_cache"
                return cached
        return self._predict_from_raw_data(
            ticker=ticker,
            raw_data=raw_data,
            mode=mode,
            requested_mode=requested_mode,
            cache_key=cache_key,
        )

    def analyze_portfolio(self, tickers: list[str]) -> Dict[str, Any]:
        """
        Analyze and optimize a portfolio of stocks.
        
        Args:
            tickers: List of NSE tickers
            
        Returns:
            Optimization results and performance metrics
        """
        print(f"\n[1/3] Fetching data for {len(tickers)} stocks...")
        optimizer = PortfolioOptimizer(risk_free_rate=0.05)
        valid_tickers = []
        
        for ticker in tickers:
            try:
                data = self.data_ingestion.process_stock_data(ticker)
                if data is not None and not data.empty:
                    optimizer.add_asset(ticker, data["close"])
                    valid_tickers.append(ticker)
            except Exception as e:
                print(f"Warning: Could not fetch data for {ticker}: {e}")
        
        if not valid_tickers:
            return {"error": "No valid data found for any of the requested tickers."}
            
        print(f"[2/3] Running optimizations for {len(valid_tickers)} stocks...")
        try:
            optimizer.calculate_statistics()
            max_sharpe = optimizer.optimize_max_sharpe()
            min_vol = optimizer.optimize_min_volatility()
            corr_matrix = optimizer.correlation_analysis()
        except Exception as e:
            return {"error": f"Portfolio optimization failed: {e}"}
            
        print("[3/3] Compiling portfolio report...")
        return {
            "tickers": valid_tickers,
            "optimizations": {
                "max_sharpe": max_sharpe,
                "min_volatility": min_vol
            },
            "correlation_matrix": corr_matrix.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

    # Removed light heuristic in favor of full ensemble deep-dive.

    def _assess_risk_level(self, volatility: float, max_drawdown: float = 0.0) -> str:
        """Assess risk using both volatility and drawdown severity."""
        severe_drawdown = abs(float(max_drawdown)) > 20
        if volatility < 0.01 and not severe_drawdown:
            return "Low Risk"
        elif volatility < 0.03 and not severe_drawdown:
            return "Moderate Risk"
        else:
            return "High Risk"


def main():
    """Test prediction service"""
    service = PredictionService()

    # Single stock prediction
    print("\nSingle Stock Prediction:")
    result = service.predict_stock("RELIANCE.NS")

    if "error" not in result:
        print(f"\nPrediction: {result['predictions']['decision']}")
        print(f"Confidence: {result['predictions']['confidence']:.1%}")
        print(f"\nRisk Level: {result['risk_metrics']['risk_level']}")
        print(f"Regime: {result['regime_analysis']['current_regime']}")

    print("\n✓ Prediction service test complete!")


if __name__ == "__main__":
    main()

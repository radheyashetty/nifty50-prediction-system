"""
Feature Engineering Module
Creates technical indicators and prepares features for ML models
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Handles all feature engineering tasks"""

    def __init__(self):
        pass

    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: Close prices
            window: Period for RSI calculation

        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            prices: Close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            MACD line, Signal line, Histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, window: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands

        Args:
            prices: Close prices
            window: Moving average period
            num_std: Number of standard deviations

        Returns:
            Upper band, Middle band, Lower band
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        Measures volatility

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for ATR

        Returns:
            ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr.fillna(tr.mean())

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        Args:
            close: Close prices
            volume: Trading volume

        Returns:
            OBV values
        """
        obv = pd.Series(0.0, index=close.index)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate stochastic oscillator %K and %D."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
        d = k.rolling(window=d_window).mean()
        return k.fillna(50), d.fillna(50)

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        plus_di = 100 * (plus_dm.rolling(window=window).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / (atr + 1e-9))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx = dx.rolling(window=window).mean()
        return adx.fillna(20)

    @staticmethod
    def calculate_roc(prices: pd.Series, window: int = 5) -> pd.Series:
        """Calculate rate of change in percent."""
        roc = ((prices / prices.shift(window)) - 1.0) * 100.0
        return roc.fillna(0.0)

    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate log returns"""
        returns = np.log(prices / prices.shift(periods))
        return pd.Series(returns, index=prices.index).fillna(0)

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std().fillna(returns.std())

    def create_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Create all features from price data

        Args:
            data: Raw OHLCV data with sentiment and macro features

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        df.columns = [str(col).strip().lower() for col in df.columns]
        df = df.loc[:, [not str(col).startswith("unnamed") for col in df.columns]]
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: {col} not found in data")
                return None

        # --- PRICE-BASED FEATURES ---

        # Returns
        df["daily_return"] = self.calculate_returns(df["close"], periods=1)
        df["5day_return"] = self.calculate_returns(df["close"], periods=5)
        df["10day_return"] = self.calculate_returns(df["close"], periods=10)
        df["20day_return"] = self.calculate_returns(df["close"], periods=20)

        # Volatility
        df["volatility_20"] = self.calculate_volatility(df["daily_return"], window=20)
        df["volatility_60"] = self.calculate_volatility(df["daily_return"], window=60)

        # Moving Averages
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        df["sma_crossover"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["price_sma20_ratio"] = df["close"] / (df["sma_20"] + 1e-9)
        df["price_sma50_ratio"] = df["close"] / (df["sma_50"] + 1e-9)

        # Price position in range
        high_20 = df["high"].rolling(window=20).max()
        low_20 = df["low"].rolling(window=20).min()
        df["price_position"] = (df["close"] - low_20) / (high_20 - low_20 + 1e-5)

        # --- TECHNICAL INDICATORS ---

        # RSI
        df["rsi_14"] = self.calculate_rsi(df["close"], window=14)
        df["rsi_7"] = self.calculate_rsi(df["close"], window=7)

        # MACD
        macd, signal, hist = self.calculate_macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_histogram"] = hist

        # ROC and stochastic
        df["roc_5"] = self.calculate_roc(df["close"], window=5)
        df["roc_10"] = self.calculate_roc(df["close"], window=10)
        stoch_k, stoch_d = self.calculate_stochastic(df["high"], df["low"], df["close"])
        df["stochastic_k"] = stoch_k
        df["stochastic_d"] = stoch_d

        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(df["close"], window=20)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / middle
        df["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-5)
        df["bollinger_pct_b"] = df["bb_position"]

        # ATR (Volatility)
        df["atr_14"] = self.calculate_atr(df["high"], df["low"], df["close"], window=14)
        df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-9)
        df["adx_14"] = self.calculate_adx(df["high"], df["low"], df["close"], window=14)

        # OBV (Volume indicator)
        df["obv"] = self.calculate_obv(df["close"], df["volume"])
        df["obv_ema"] = df["obv"].ewm(span=20).mean()
        df["obv_change_5d"] = self.calculate_roc(df["obv"], window=5)

        # --- VOLUME FEATURES ---

        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-5)

        # --- PRICE ACTION ---

        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]
        df["price_range"] = (df["high"] - df["low"]) / df["close"]

        # --- SENTIMENT & MACRO (already added in data ingestion) ---
        # These should already be in the dataframe: sentiment, interest_rate, etc.

        # --- LAGGED FEATURES ---

        for lag in [1, 5, 10, 20]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        # --- TARGET VARIABLE ---
        # Predict if price goes up in next 3 days
        df["future_return_5d"] = self.calculate_returns(df["close"], periods=-5)
        df["target_5d"] = (df["future_return_5d"] > 0.015).astype(int)

        # Keep legacy names for compatibility with existing pipeline.
        df["future_return_3d"] = df["future_return_5d"]
        df["target_3d"] = df["target_5d"]

        # Probabilistic target: normalized future return
        df["target_regression"] = df["future_return_5d"]

        # --- CLEAN UP NaN VALUES ---
        # IMPORTANT: Never bfill target columns — that would leak future data!
        target_cols = [
            "future_return_5d",
            "future_return_3d",
            "target_5d",
            "target_3d",
            "target_regression",
        ]
        feature_cols_to_fill = [c for c in df.columns if c not in target_cols]
        df[feature_cols_to_fill] = df[feature_cols_to_fill].bfill().ffill().fillna(0)
        # For targets: only forward-fill (no backfill!) then drop remaining NaN rows
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        df = df.dropna(subset=[c for c in target_cols if c in df.columns])

        return df

    def get_indicator_snapshot(self, data: pd.DataFrame) -> Dict[str, float]:
        """Return current indicator values for UI/API display."""
        if data is None or data.empty:
            return {}

        latest = data.iloc[-1]
        snapshot = {
            "rsi_14": float(latest.get("rsi_14", 0.0)),
            "rsi_7": float(latest.get("rsi_7", 0.0)),
            "macd_line": float(latest.get("macd", 0.0)),
            "macd_signal": float(latest.get("macd_signal", 0.0)),
            "macd_histogram": float(latest.get("macd_histogram", 0.0)),
            "bollinger_upper": float(latest.get("bb_upper", 0.0)),
            "bollinger_lower": float(latest.get("bb_lower", 0.0)),
            "bollinger_pct_b": float(latest.get("bollinger_pct_b", 0.0)),
            "bollinger_width": float(latest.get("bb_width", 0.0)),
            "atr_14": float(latest.get("atr_14", 0.0)),
            "atr_pct": float(latest.get("atr_pct", 0.0)),
            "obv": float(latest.get("obv", 0.0)),
            "sma_20": float(latest.get("sma_20", 0.0)),
            "sma_50": float(latest.get("sma_50", 0.0)),
            "ema_9": float(latest.get("ema_9", 0.0)),
            "ema_21": float(latest.get("ema_21", 0.0)),
            "adx_14": float(latest.get("adx_14", 0.0)),
            "volume_ratio": float(latest.get("volume_ratio", 0.0)),
            "historical_vol_20": float(latest.get("volatility_20", 0.0)),
            "stochastic_k": float(latest.get("stochastic_k", 0.0)),
            "stochastic_d": float(latest.get("stochastic_d", 0.0)),
            "roc_5": float(latest.get("roc_5", 0.0)),
            "roc_10": float(latest.get("roc_10", 0.0)),
            "return_5d": float(latest.get("5day_return", 0.0)),
        }
        return snapshot

    def get_feature_list(self) -> Dict[str, list]:
        """
        Returns list of features by category
        Useful for model training and feature selection
        """
        return {
            "technical": [
                "rsi_14",
                "rsi_7",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "bb_width",
                "bb_position",
                "atr_14",
                "atr_pct",
                "adx_14",
                "roc_5",
                "roc_10",
                "stochastic_k",
                "stochastic_d",
                "obv",
                "obv_ema",
                "obv_change_5d",
            ],
            "momentum": [
                "daily_return",
                "5day_return",
                "10day_return",
                "20day_return",
                "volatility_20",
                "volatility_60",
            ],
            "moving_averages": [
                "sma_5",
                "sma_10",
                "sma_20",
                "sma_50",
                "ema_9",
                "ema_21",
                "ema_12",
                "ema_26",
                "sma_crossover",
                "price_sma20_ratio",
                "price_sma50_ratio",
            ],
            "volume": ["volume", "volume_sma_20", "volume_ratio"],
            "price_action": [
                "high_low_ratio",
                "close_open_ratio",
                "price_range",
                "price_position",
            ],
            "sentiment": ["sentiment"],
            "macro": ["interest_rate", "inflation_rate", "usd_inr", "global_vix"],
            "lagged": [
                "close_lag_1",
                "close_lag_5",
                "close_lag_10",
                "close_lag_20",
                "return_lag_1",
                "return_lag_5",
                "return_lag_10",
                "return_lag_20",
                "volume_lag_1",
                "volume_lag_5",
                "volume_lag_10",
                "volume_lag_20",
            ],
        }


def main():
    """Test feature engineering"""
    from data_ingestion import DataIngestion

    # Fetch data
    ingestion = DataIngestion(lookback_days=365)
    data = ingestion.process_stock_data("RELIANCE.NS")

    if data is not None:
        # Create features
        engineer = FeatureEngineer()
        features = engineer.create_features(data)

        if features is None:
            print("Feature generation failed due to missing required columns.")
            return

        print(f"Original data shape: {data.shape}")
        print(f"Features data shape: {features.shape}")
        print(f"\nFeature columns: {features.columns.tolist()}")
        print(f"\nFirst few rows:\n{features.head()}")

        # Show feature categories
        print(f"\nFeature categories: {engineer.get_feature_list().keys()}")


if __name__ == "__main__":
    main()

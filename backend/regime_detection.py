"""
Regime Detection Module
Detects market regimes (Bull, Bear, Sideways) using clustering and HMM
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class RegimeDetector:
    """Detects market regimes using technical indicators and clustering"""

    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector

        Args:
            n_regimes: Number of regimes to detect (usually 3: Bull, Bear, Sideways)
        """
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.regime_names = {0: "Sideways", 1: "Bear", 2: "Bull"}
        self.current_regime = None
        self.regime_history = []
        self.regimes = np.array([], dtype=int)

    def detect_regimes(
        self, price_data: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Detect regimes using returns and volatility

        Args:
            price_data: DataFrame with 'close' column

        Returns:
            Regime labels, Features used for clustering
        """
        # Calculate features
        returns = price_data["close"].pct_change().fillna(0)
        volatility = returns.rolling(window=20).std().fillna(0)
        momentum = returns.rolling(window=20).mean().fillna(0)

        # Create feature matrix
        features = np.column_stack(
            [
                np.asarray(returns, dtype=float),
                np.asarray(volatility, dtype=float),
                np.asarray(momentum, dtype=float),
            ]
        )

        # Handle NaN values
        features = np.nan_to_num(features)

        # Cluster
        self.regimes = self.kmeans.fit_predict(features)

        # Sort regimes by return (0=low, 1=medium, 2=high)
        regime_returns = {}
        for regime in range(self.n_regimes):
            mask = self.regimes == regime
            avg_return = returns[mask].mean()
            regime_returns[regime] = avg_return

        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        self.regimes = np.array([regime_mapping[r] for r in self.regimes])

        features_df = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "momentum": momentum,
                "regime": self.regimes,
            }
        )

        return self.regimes, features_df

    def get_regime_name(self, regime_idx: int) -> str:
        """Get human-readable regime name"""
        regime_names = {0: "Sideways", 1: "Bear", 2: "Bull"}
        return regime_names.get(regime_idx, f"Regime_{regime_idx}")

    def get_current_regime(self) -> str:
        """Get current market regime"""
        if len(self.regimes) > 0:
            current = self.regimes[-1]
            return self.get_regime_name(current)
        return "Unknown"

    def get_regime_characteristics(
        self, price_data: pd.DataFrame, regimes: np.ndarray
    ) -> Dict:
        """
        Analyze characteristics of each regime

        Args:
            price_data: Historical price data
            regimes: Regime labels

        Returns:
            Dictionary with regime statistics
        """
        returns = price_data["close"].pct_change().fillna(0)
        volatility = returns.rolling(window=20).std().fillna(0)

        characteristics = {}

        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            regime_volatility = volatility[mask]

            characteristics[self.get_regime_name(regime)] = {
                "avg_daily_return": regime_returns.mean() * 100,
                "volatility": regime_volatility.mean() * 100,
                "win_rate": (regime_returns > 0).sum() / len(regime_returns) * 100,
                "duration_periods": mask.sum(),
                "avg_price_change": regime_returns.mean() * 100,
            }

        return characteristics


class VolatilityRegimeDetector:
    """
    Detect regimes based primarily on volatility levels
    Useful for risk management
    """

    def __init__(
        self,
        window: int = 20,
        low_vol_threshold: float = 0.01,
        high_vol_threshold: float = 0.03,
    ):
        """
        Initialize volatility regime detector

        Args:
            window: Rolling window for volatility
            low_vol_threshold: Cutoff for low volatility
            high_vol_threshold: Cutoff for high volatility
        """
        self.window = window
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold

    def detect_regimes(self, price_data: pd.DataFrame) -> Dict:
        """
        Detect volatility regimes

        Args:
            price_data: DataFrame with 'close' column

        Returns:
            Dictionary with regime information
        """
        returns = price_data["close"].pct_change().fillna(0)
        volatility = returns.rolling(window=self.window).std().fillna(0)

        regimes = np.zeros(len(volatility))

        for i, vol in enumerate(volatility):
            if vol < self.low_vol_threshold:
                regimes[i] = 0  # Low volatility
            elif vol > self.high_vol_threshold:
                regimes[i] = 2  # High volatility
            else:
                regimes[i] = 1  # Medium volatility

        result = {
            "regimes": regimes.astype(int),
            "volatility": volatility.values,
            "labels": {
                0: "Low Volatility",
                1: "Medium Volatility",
                2: "High Volatility",
            },
            "current_regime": self._get_regime_name(regimes[-1]),
            "current_volatility": volatility.iloc[-1],
        }

        return result

    def _get_regime_name(self, regime: int) -> str:
        """Get regime name"""
        names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}
        return names.get(regime, "Unknown")

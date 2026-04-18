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


class HiddenMarkovModelRegimeDetector:
    """
    Regime detection using Hidden Markov Model
    Better at capturing state transitions
    """

    def __init__(self, n_regimes: int = 3, n_components: int = 2):
        """
        Initialize HMM regime detector

        Args:
            n_regimes: Number of regimes
            n_components: Features per regime
        """
        self.n_regimes = n_regimes
        self.n_components = n_components
        try:
            from importlib import import_module

            hmm_module = import_module("hmmlearn.gaussian_hmm")
            self.GaussianHMM = hmm_module.GaussianHMM
            self.hmm_available = True
        except ImportError:
            self.hmm_available = False
            print("Warning: hmmlearn not installed. Using simplified regime detection.")

    def detect_regimes(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Detect regimes using HMM

        Args:
            price_data: DataFrame with 'close' column

        Returns:
            Regime labels
        """
        if not self.hmm_available:
            # Fallback to simple method
            return self._simple_regime_detection(price_data)

        # Calculate features
        returns = price_data["close"].pct_change().fillna(0).values
        volatility = pd.Series(returns).rolling(window=20).std().fillna(0).values

        features = np.column_stack(
            [np.asarray(returns, dtype=float), np.asarray(volatility, dtype=float)]
        )
        features = np.nan_to_num(features)

        # Fit HMM
        hmm = self.GaussianHMM(n_components=self.n_regimes, random_state=42)
        regimes = hmm.fit_predict(features)

        return regimes

    def _simple_regime_detection(self, price_data: pd.DataFrame) -> np.ndarray:
        """Simplified regime detection when HMM not available"""
        returns = price_data["close"].pct_change().fillna(0)

        # Use running mean and std
        regimes = np.zeros(len(returns))

        for i in range(20, len(returns)):
            window_return = returns.iloc[i - 20 : i].mean()
            window_vol = returns.iloc[i - 20 : i].std()

            if window_return > 0.0005 and window_vol < 0.02:
                regimes[i] = 2  # Bull
            elif window_return < -0.0005 or window_vol > 0.03:
                regimes[i] = 1  # Bear
            else:
                regimes[i] = 0  # Sideways

        return regimes.astype(int)


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


def main():
    """Test regime detection"""
    from data_ingestion import DataIngestion

    print("Loading data...")
    ingestion = DataIngestion(lookback_days=365)
    data = ingestion.process_stock_data("RELIANCE.NS")

    if data is not None:
        print("\n=== Regime Detection using Clustering ===")
        detector = RegimeDetector(n_regimes=3)
        regimes, features = detector.detect_regimes(data)

        print(f"Current regime: {detector.get_current_regime()}")
        print(f"Regime counts: {np.bincount(regimes)}")

        # Regime characteristics
        chars = detector.get_regime_characteristics(data, regimes)
        print("\nRegime Characteristics:")
        for regime_name, stats in chars.items():
            print(f"\n{regime_name}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}")

        print("\n=== Volatility Regime Detection ===")
        vol_detector = VolatilityRegimeDetector()
        vol_regimes = vol_detector.detect_regimes(data)

        print(f"Current volatility regime: {vol_regimes['current_regime']}")
        print(f"Current volatility: {vol_regimes['current_volatility']:.4f}")
        print(f"Regime distribution: {np.bincount(vol_regimes['regimes'])}")

        print("\n✓ Regime detection complete!")


if __name__ == "__main__":
    main()

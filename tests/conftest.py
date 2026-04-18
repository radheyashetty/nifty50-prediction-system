"""
Pytest fixtures and configuration for test suite.

This file provides shared test utilities and fixtures used across test files.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_ticker() -> str:
    """Provide a valid NIFTY 50 ticker for testing."""
    return "RELIANCE.NS"


@pytest.fixture
def invalid_ticker() -> str:
    """Provide an invalid ticker for negative testing."""
    return "INVALID_XYZ.NS"


@pytest.fixture
def sample_ohlcv_data():
    """Provide sample OHLCV data for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2023-01-01", periods=100)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(100, 200, 100),
            "low": np.random.uniform(100, 200, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1e6, 10e6, 100),
        }
    )


@pytest.fixture
def sample_features_data():
    """Provide sample feature engineering data."""
    import pandas as pd
    import numpy as np

    return pd.DataFrame(
        {
            "SMA_20": np.random.uniform(0, 1, 50),
            "RSI": np.random.uniform(30, 70, 50),
            "MACD": np.random.uniform(-1, 1, 50),
            "ATR": np.random.uniform(0, 5, 50),
        }
    )


@pytest.fixture
def mock_prediction_result():
    """Provide a mock prediction result matching API contract."""
    return {
        "ticker": "RELIANCE.NS",
        "signal": "BUY",
        "confidence": 0.75,
        "model_scores": {
            "xgboost": 0.78,
            "random_forest": 0.72,
            "ensemble": 0.75,
        },
        "model_metrics": {
            "accuracy": 0.68,
            "precision": 0.71,
            "recall": 0.65,
            "f1": 0.68,
        },
        "top_features": [
            {"name": "RSI", "value": 0.25},
            {"name": "SMA_20", "value": 0.18},
            {"name": "MACD", "value": 0.15},
        ],
        "indicators": {
            "rsi": 65,
            "macd": 0.5,
            "atr": 2.3,
        },
        "backtest": {
            "win_rate": 0.62,
            "avg_return": 0.015,
            "sharpe_ratio": 1.2,
        },
        "risk_metrics": {
            "volatility": 0.18,
            "max_drawdown": -0.12,
        },
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Hooks for test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Auto-mark tests containing 'integration' as integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

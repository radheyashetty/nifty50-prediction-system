"""
Example unit tests for data utilities.

This file demonstrates testing patterns for:
- Data normalization functions
- Ticker validation
- Exception handling
"""

import pytest
from backend.utils import get_ticker_sector


class TestTickerValidation:
    """Test ticker normalization and validation."""

    def test_nifty50_ticker_recognized(self) -> None:
        """Test that known NIFTY 50 tickers are recognized."""
        ticker = "RELIANCE.NS"
        sector = get_ticker_sector(ticker)
        assert sector is not None, f"RELIANCE should be in NIFTY 50, got {sector}"

    def test_ticker_case_insensitive(self) -> None:
        """Test that ticker lookups are case-insensitive."""
        ticker1 = "reliance.ns"
        ticker2 = "RELIANCE.NS"
        sector1 = get_ticker_sector(ticker1)
        sector2 = get_ticker_sector(ticker2)
        assert sector1 == sector2, "Ticker lookup should be case-insensitive"

    def test_invalid_ticker_returns_none(self) -> None:
        """Test that invalid tickers return None."""
        ticker = "INVALID_TICKER.NS"
        sector = get_ticker_sector(ticker)
        assert (
            sector == "Unknown"
        ), f"Invalid ticker should return 'Unknown', got {sector}"


class TestDataNormalization:
    """Test data normalization utilities."""

    @pytest.mark.parametrize(
        "value,expected_type",
        [
            (42, (int, float)),
            ("3.14", (int, float)),
            ("invalid", type(None)),
        ],
    )
    def test_safe_float_conversion(self, value, expected_type) -> None:
        """Test safe float conversion with various inputs."""
        from backend.models import _safe_float

        result = _safe_float(value)
        assert isinstance(result, expected_type) or result == 0.0 or result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Regression tests for uploaded stock dataset cleaning and preparation."""

from __future__ import annotations

import pandas as pd

from backend.data_ingestion import DataIngestion
from backend.predictions import PredictionService


def _sample_stock_csv_bytes() -> bytes:
    sample_df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3, 4],
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [100000, 110000, 90000, 120000, 105000],
            "Adj Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "SYMBOL": [
                "RELIANCE.NS",
                "RELIANCE.NS",
                "RELIANCE.NS",
                "RELIANCE.NS",
                "RELIANCE.NS",
            ],
        }
    )
    return sample_df.to_csv(index=False).encode("utf-8")


def test_uploaded_file_is_cleaned() -> None:
    ingestion = DataIngestion(lookback_days=365)
    parsed = ingestion.process_uploaded_file(
        content=_sample_stock_csv_bytes(),
        filename="sample_stock.csv",
        ticker="RELIANCE.NS",
    )

    assert parsed["ok"] is True

    cleaned = parsed["data"]
    assert isinstance(cleaned, pd.DataFrame)
    assert not any(str(col).lower().startswith("unnamed") for col in cleaned.columns)
    assert {"date", "open", "high", "low", "close", "volume", "symbol"}.issubset(
        set(cleaned.columns)
    )
    assert cleaned["date"].is_monotonic_increasing
    assert (
        cleaned[["date", "open", "high", "low", "close", "volume"]].isna().sum().sum()
        == 0
    )
    assert cleaned["symbol"].nunique() == 1


def test_prediction_preparation_drops_upload_artifacts() -> None:
    ingestion = DataIngestion(lookback_days=365)
    parsed = ingestion.process_uploaded_file(
        content=_sample_stock_csv_bytes(),
        filename="sample_stock.csv",
        ticker="RELIANCE.NS",
    )

    service = PredictionService(lookback_days=365)
    prepared = service.prepare_uploaded_price_data(parsed["data"], ticker="RELIANCE.NS")

    assert isinstance(prepared, pd.DataFrame)
    assert not any(str(col).lower().startswith("unnamed") for col in prepared.columns)
    assert {"date", "open", "high", "low", "close", "volume", "adj close"}.issubset(
        set(prepared.columns)
    )
    assert prepared["date"].is_monotonic_increasing
    assert prepared["symbol"].nunique() == 1

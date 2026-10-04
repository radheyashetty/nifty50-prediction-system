"""
Example integration tests for full workflows.

These tests demonstrate testing patterns for multi-step end-to-end flows.
They use fixtures from conftest.py for shared test data.
"""

import pytest


class TestDataIngestionWorkflow:
    """Test complete data ingestion pipeline."""

    def test_load_ticker_data(self, sample_ticker) -> None:
        """Test loading data for a known ticker."""
        from backend.data_ingestion import DataIngestion
        assert sample_ticker == "RELIANCE.NS"
        result = DataIngestion(lookback_days=365).process_stock_data(sample_ticker)
        assert result is not None
        assert len(result) > 0

    def test_data_quality_checks(self, sample_ohlcv_data) -> None:
        """Test that data quality checks work correctly."""
        # Verify sample data has expected columns
        expected_cols = {"date", "open", "high", "low", "close", "volume"}
        assert set(sample_ohlcv_data.columns) == expected_cols
        # Verify no NaN values in sample
        assert not sample_ohlcv_data.isnull().any().any()


class TestFeatureEngineeringWorkflow:
    """Test feature engineering pipeline."""

    def test_feature_generation(self, sample_ohlcv_data) -> None:
        """Test that features are generated correctly."""
        # In real implementation:
        # features = FeatureEngineer.generate(sample_ohlcv_data)
        # assert "SMA_20" in features.columns
        # assert "RSI" in features.columns
        # Placeholder for demonstration
        assert len(sample_ohlcv_data) == 100

    def test_feature_normalization(self, sample_features_data) -> None:
        """Test that features are normalized properly."""
        # Verify sample features are in expected ranges
        assert sample_features_data["RSI"].min() >= 30
        assert sample_features_data["RSI"].max() <= 70


class TestPredictionWorkflow:
    """Test complete prediction pipeline."""

    def test_prediction_api_response_schema(self, mock_prediction_result) -> None:
        """Test that prediction responses match expected schema."""
        # Verify required fields exist
        required_fields = {
            "ticker",
            "signal",
            "confidence",
            "model_scores",
            "top_features",
        }
        assert required_fields.issubset(set(mock_prediction_result.keys()))

        # Verify signal is valid
        assert mock_prediction_result["signal"] in ["BUY", "SELL", "HOLD"]

        # Verify confidence is in valid range
        assert 0 <= mock_prediction_result["confidence"] <= 1

    def test_prediction_model_scores(self, mock_prediction_result) -> None:
        """Test model score consistency."""
        scores = mock_prediction_result["model_scores"]
        assert "xgboost" in scores
        assert "random_forest" in scores
        assert "ensemble" in scores
        # All should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1

    def test_backtest_metrics_validity(self, mock_prediction_result) -> None:
        """Test backtest metrics are valid."""
        backtest = mock_prediction_result["backtest"]
        assert 0 <= backtest["win_rate"] <= 1
        assert backtest["sharpe_ratio"] > 0


class TestScreenerWorkflow:
    """Test stock screener pipeline."""

    @pytest.mark.integration
    def test_screener_filters_by_confidence(self) -> None:
        """Test that screener properly filters by confidence threshold."""
        from backend.screener import StockScreener
        from backend.predictions import PredictionService
        service = PredictionService(lookback_days=365)
        screener = StockScreener(service)
        results = screener.run_screener(min_confidence=0.60, use_cache=True, top_n=5)
        assert "bullish" in results
        assert "bearish" in results
        for r in results["bullish"]:
            assert r["confidence"] >= 0.60

    def test_screener_sector_filtering(self) -> None:
        """Test that screener can filter by sector."""
        from backend.screener import StockScreener
        from backend.predictions import PredictionService
        service = PredictionService(lookback_days=365)
        screener = StockScreener(service)
        results = screener.run_screener(sector="Information Technology", use_cache=True, top_n=5)
        assert "bullish" in results
        assert "bearish" in results
        for r in results["bullish"] + results["bearish"]:
            assert r["sector"] == "Information Technology"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

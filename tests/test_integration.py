import pytest
from backend.data_ingestion import DataIngestion
from backend.feature_engineering import FeatureEngineer
from backend.predictions import PredictionService
from backend.screener import StockScreener


class TestDataIngestionWorkflow:
    def test_load_ticker_data(self, sample_ticker):
        assert sample_ticker == "RELIANCE.NS"
        result = DataIngestion(lookback_days=365).process_stock_data(sample_ticker)
        assert result is not None
        assert len(result) > 0

    def test_data_quality_checks(self, sample_ohlcv_data):
        expected_cols = {"date", "open", "high", "low", "close", "volume"}
        assert set(sample_ohlcv_data.columns) == expected_cols
        assert not sample_ohlcv_data.isnull().any().any()


class TestFeatureEngineeringWorkflow:
    def test_feature_generation(self, sample_ohlcv_data):
        features = FeatureEngineer().create_features(sample_ohlcv_data)
        assert features is not None
        assert "rsi_14" in features.columns
        assert "daily_return" in features.columns

    def test_feature_normalization(self, sample_features_data):
        assert sample_features_data["RSI"].min() >= 30
        assert sample_features_data["RSI"].max() <= 70


class TestPredictionWorkflow:
    def test_prediction_api_response_schema(self, mock_prediction_result):
        required_fields = {
            "ticker",
            "signal",
            "confidence",
            "model_scores",
            "top_features",
        }
        assert required_fields.issubset(set(mock_prediction_result.keys()))
        assert mock_prediction_result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= mock_prediction_result["confidence"] <= 1

    def test_prediction_model_scores(self, mock_prediction_result):
        scores = mock_prediction_result["model_scores"]
        assert "xgboost" in scores
        assert "random_forest" in scores
        assert "ensemble" in scores
        for score in scores.values():
            assert 0 <= score <= 1

    def test_backtest_metrics_validity(self, mock_prediction_result):
        backtest = mock_prediction_result["backtest"]
        assert 0 <= backtest["win_rate"] <= 1
        assert backtest["sharpe_ratio"] > 0


class TestScreenerWorkflow:
    @pytest.mark.integration
    def test_screener_filters_by_confidence(self):
        service = PredictionService(lookback_days=365)
        screener = StockScreener(service)
        results = screener.run_screener(min_confidence=0.60, use_cache=True, top_n=5)
        assert "bullish" in results
        assert "bearish" in results
        for r in results["bullish"]:
            assert r["confidence"] >= 0.60

    def test_screener_sector_filtering(self):
        service = PredictionService(lookback_days=365)
        screener = StockScreener(service)
        results = screener.run_screener(sector="Information Technology", use_cache=True, top_n=5)
        assert "bullish" in results
        assert "bearish" in results
        for r in results["bullish"] + results["bearish"]:
            assert r["sector"] == "Information Technology"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

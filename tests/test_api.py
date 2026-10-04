import pytest
from fastapi.testclient import TestClient
from frontend.web_app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestDashboardEndpoint:
    def test_get_dashboard_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestAnalyzeEndpoint:
    def test_analyze_valid_ticker(self, client):
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "RELIANCE.NS",
                "lookback_days": 365,
                "analysis_mode": "cache",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "signal" in data
        assert "model_scores" in data

    def test_analyze_invalid_ticker_returns_error(self, client):
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "INVALID_XYZ_123.NS",
                "lookback_days": 365,
                "analysis_mode": "cache",
            },
        )
        assert response.status_code in [400, 404, 422, 500]

    def test_analyze_validates_lookback_range(self, client):
        response = client.post(
            "/api/analyze",
            json={
                "ticker": "RELIANCE.NS",
                "lookback_days": 10,
                "analysis_mode": "cache",
            },
        )
        assert response.status_code == 422


class TestScreenerEndpoint:
    def test_screener_returns_bullish_and_bearish(self, client):
        response = client.post(
            "/api/screener",
            json={"sector": None, "min_confidence": 0.55},
        )
        assert response.status_code == 200
        data = response.json()
        assert "bullish" in data
        assert "bearish" in data

    def test_screener_sector_filter(self, client):
        response = client.post(
            "/api/screener",
            json={"sector": "Information Technology"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "bullish" in data or "bearish" in data


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/data-health")
        assert response.status_code == 200
        data = response.json()
        assert "lookback_days" in data
        assert "coverage" in data

    def test_stocks_endpoint(self, client):
        response = client.get("/api/stocks")
        assert response.status_code == 200
        data = response.json()
        assert "stocks" in data
        assert "by_sector" in data
        assert data["total_count"] > 0


class TestPortfolioEndpoint:
    def test_portfolio_optimization(self, client):
        response = client.post(
            "/api/portfolio",
            json={"tickers": ["RELIANCE.NS", "TCS.NS"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "optimizations" in data
        assert "max_sharpe" in data["optimizations"]
        assert "min_volatility" in data["optimizations"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


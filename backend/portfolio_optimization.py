"""
Portfolio Optimization Module
Implements Modern Portfolio Theory for optimal stock allocation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize portfolio optimizer

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculations
        """
        self.risk_free_rate = risk_free_rate
        self.prices_data = {}
        self.returns_data: pd.DataFrame | None = None
        self.cov_matrix: pd.DataFrame | None = None
        self.mean_returns: pd.Series | None = None

    def add_asset(self, ticker: str, price_series: pd.Series):
        """Add asset to portfolio"""
        self.prices_data[ticker] = price_series

    def calculate_statistics(self):
        """Calculate returns and correlation statistics"""
        # Convert prices to returns
        prices_df = pd.DataFrame(self.prices_data)
        returns_data = prices_df.pct_change().dropna()
        self.returns_data = returns_data

        # Calculate mean returns and covariance
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, risk, and Sharpe ratio

        Args:
            weights: Asset weights (should sum to 1)

        Returns:
            (return, risk, sharpe_ratio)
        """
        if self.mean_returns is None or self.cov_matrix is None:
            self.calculate_statistics()
        assert self.mean_returns is not None
        assert self.cov_matrix is not None
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights))
        ) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std

        return portfolio_return, portfolio_std, sharpe_ratio

    def negative_sharpe(self, weights: np.ndarray) -> float:
        """Objective function: minimize negative Sharpe ratio"""
        return -self.portfolio_performance(weights)[2]

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Objective: minimize portfolio volatility"""
        return self.portfolio_performance(weights)[1]

    def optimize_max_sharpe(self) -> Dict:
        """
        Find portfolio with maximum Sharpe ratio

        Returns:
            Dictionary with optimal weights and performance
        """
        if self.mean_returns is None:
            self.calculate_statistics()
        assert self.mean_returns is not None

        n_assets = len(self.mean_returns)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1 / n_assets] * n_assets)

        result = minimize(
            self.negative_sharpe,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        opt_return, opt_risk, opt_sharpe = self.portfolio_performance(result.x)

        return {
            "weights": dict(zip(self.mean_returns.index, result.x)),
            "return": opt_return,
            "risk": opt_risk,
            "sharpe_ratio": opt_sharpe,
            "success": result.success,
        }

    def optimize_min_volatility(self) -> Dict:
        """
        Find portfolio with minimum volatility

        Returns:
            Dictionary with optimal weights and performance
        """
        if self.mean_returns is None:
            self.calculate_statistics()
        assert self.mean_returns is not None

        n_assets = len(self.mean_returns)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1 / n_assets] * n_assets)

        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        opt_return, opt_risk, opt_sharpe = self.portfolio_performance(result.x)

        return {
            "weights": dict(zip(self.mean_returns.index, result.x)),
            "return": opt_return,
            "risk": opt_risk,
            "sharpe_ratio": opt_sharpe,
            "success": result.success,
        }

    def optimize_target_return(self, target_return: float) -> Dict:
        """
        Find minimum volatility portfolio with target return

        Args:
            target_return: Target annual return

        Returns:
            Dictionary with optimal weights
        """
        if self.mean_returns is None:
            self.calculate_statistics()
        assert self.mean_returns is not None

        n_assets = len(self.mean_returns)

        # Constraints: sum to 1, and achieve target return
        constraints = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {
                "type": "eq",
                "fun": lambda x: self.portfolio_performance(x)[0] - target_return,
            },
        )

        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1 / n_assets] * n_assets)

        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            opt_return, opt_risk, opt_sharpe = self.portfolio_performance(result.x)

            return {
                "weights": dict(zip(self.mean_returns.index, result.x)),
                "return": opt_return,
                "risk": opt_risk,
                "sharpe_ratio": opt_sharpe,
                "success": True,
            }
        else:
            return {
                "success": False,
                "message": "Target return not achievable with given assets",
            }

    def efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios

        Args:
            num_portfolios: Number of random portfolios to evaluate

        Returns:
            DataFrame with portfolio returns, risks, and Sharpe ratios
        """
        if self.mean_returns is None:
            self.calculate_statistics()
        assert self.mean_returns is not None

        n_assets = len(self.mean_returns)
        results = np.zeros((3, num_portfolios))

        np.random.seed(42)
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            # Calculate performance
            ret, risk, sharpe = self.portfolio_performance(weights)

            results[0, i] = ret
            results[1, i] = risk
            results[2, i] = sharpe

        frontier_df = pd.DataFrame(
            {"Return": results[0], "Risk": results[1], "Sharpe": results[2]}
        )

        return frontier_df.sort_values("Risk")

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Analyze correlation between assets

        Returns:
            Correlation matrix
        """
        if self.returns_data is None:
            self.calculate_statistics()
        assert self.returns_data is not None

        return self.returns_data.corr()

    def risk_allocation(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk contribution by asset

        Args:
            weights: Portfolio weights

        Returns:
            Risk contribution by asset
        """
        if self.mean_returns is None or self.cov_matrix is None:
            self.calculate_statistics()
        assert self.mean_returns is not None
        assert self.cov_matrix is not None
        portfolio_volatility = self.portfolio_performance(weights)[1]

        # Marginal contribution to risk
        mcr = np.dot(self.cov_matrix, weights) / portfolio_volatility

        # Contribution to risk
        risk_contrib = weights * mcr

        return dict(zip(self.mean_returns.index, risk_contrib))


class SectorWeighting:
    """Sector-based portfolio weighting"""

    def __init__(self):
        self.sector_stocks = {}

    def add_stock_to_sector(self, stock: str, sector: str):
        """Map stock to sector"""
        if sector not in self.sector_stocks:
            self.sector_stocks[sector] = []
        self.sector_stocks[sector].append(stock)

    def get_sector_allocation(
        self, stock_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate sector-level allocation from stock weights

        Args:
            stock_weights: Individual stock weights

        Returns:
            Sector weights
        """
        sector_weights = {}

        for sector, stocks in self.sector_stocks.items():
            sector_weight = sum(stock_weights.get(stock, 0) for stock in stocks)
            if sector_weight > 0:
                sector_weights[sector] = sector_weight

        return sector_weights

    def recommend_sector_allocation(self) -> Dict[str, float]:
        """
        Provide recommended sector allocation (simple equal-weight)

        Returns:
            Recommended sector weights
        """
        n_sectors = len(self.sector_stocks)
        return {sector: 1 / n_sectors for sector in self.sector_stocks.keys()}


def main():
    """Test portfolio optimization"""
    from data_ingestion import DataIngestion

    print("Loading NIFTY 50 stocks data...")
    ingestion = DataIngestion(lookback_days=365)

    # Select a subset of stocks
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS", "ICICIBANK.NS"]

    optimizer = PortfolioOptimizer(risk_free_rate=0.05)

    for ticker in tickers:
        data = ingestion.process_stock_data(ticker)
        if data is not None:
            optimizer.add_asset(ticker, data["close"])

    # Calculate statistics
    optimizer.calculate_statistics()

    # Optimize for max Sharpe
    print("\n=== Maximum Sharpe Ratio Optimization ===")
    max_sharpe = optimizer.optimize_max_sharpe()
    print(f"✓ Optimization successful: {max_sharpe['success']}")
    print(f"Expected Return: {max_sharpe['return']*100:.2f}%")
    print(f"Risk (Volatility): {max_sharpe['risk']*100:.2f}%")
    print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.2f}")
    print("\nOptimal Allocation:")
    for ticker, weight in sorted(
        max_sharpe["weights"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {ticker}: {weight*100:.1f}%")

    # Optimize for min volatility
    print("\n=== Minimum Volatility Optimization ===")
    min_vol = optimizer.optimize_min_volatility()
    print(f"Expected Return: {min_vol['return']*100:.2f}%")
    print(f"Risk (Volatility): {min_vol['risk']*100:.2f}%")
    print(f"Sharpe Ratio: {min_vol['sharpe_ratio']:.2f}")

    # Correlation analysis
    print("\n=== Correlation Matrix ===")
    corr = optimizer.correlation_analysis()
    print(corr.round(3))

    print("\n✓ Portfolio optimization complete!")


if __name__ == "__main__":
    main()

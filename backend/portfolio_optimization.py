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

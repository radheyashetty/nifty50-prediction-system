"""
Backtesting Engine
Simulates trading strategies and evaluates performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


class BacktestEngine:
    """Trading strategy backtesting engine"""

    def __init__(
        self, initial_capital: float = 100000, transaction_cost_pct: float = 0.001
    ):
        """
        Initialize backtests engine

        Args:
            initial_capital: Starting portfolio value
            transaction_cost_pct: Transaction cost as % of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.results = {}

    def simple_ma_strategy(
        self, prices: pd.Series, short_window: int = 20, long_window: int = 50
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Simple moving average crossover strategy (long-only)

        Args:
            prices: Close prices
            short_window: Short MA period
            long_window: Long MA period

        Returns:
            Signals (1=long, 0=cash), Positions
        """
        signals = pd.Series(0, index=prices.index, dtype=float)

        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()

        # Long-only: 1 when short MA > long MA (bullish), 0 otherwise
        signals[short_ma > long_ma] = 1

        # Position: actual positions held
        positions = signals.diff()

        return signals, positions

    def rsi_strategy(
        self,
        prices: pd.Series,
        rsi_values: np.ndarray,
        oversold: float = 30,
        overbought: float = 70,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        RSI-based strategy (long-only mean reversion)

        Args:
            prices: Close prices
            rsi_values: RSI indicator values
            oversold: Oversold threshold
            overbought: Overbought threshold

        Returns:
            Signals (1=long, 0=cash), Positions
        """
        signals = pd.Series(0, index=prices.index, dtype=float)

        # Long-only: buy when oversold, go to cash when overbought
        in_position = False
        for i in range(len(signals)):
            rsi_val = float(rsi_values[i]) if i < len(rsi_values) else 50.0
            if rsi_val < oversold:
                in_position = True
            elif rsi_val > overbought:
                in_position = False
            signals.iloc[i] = 1.0 if in_position else 0.0

        positions = signals.diff()

        return signals, positions


    def backtest_strategy(
        self, prices: pd.Series, signals: pd.Series, strategy_name: str = "Strategy"
    ) -> Dict:
        """
        Run backtest on strategy signals

        Args:
            prices: Close prices
            signals: Buy/sell signals
            strategy_name: Name of strategy

        Returns:
            Dictionary with performance metrics
        """
        # Calculate returns, clamp extreme daily moves (data quality guard)
        returns = prices.pct_change().fillna(0)
        returns = returns.clip(-0.20, 0.20)  # Cap at ±20% per day

        # Strategy returns (multiply by signals - we take positions based on signals)
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)

        # Apply transaction costs when entering/exiting
        positions_diff = signals.diff().abs()
        transaction_costs = positions_diff * self.transaction_cost_pct
        strategy_returns -= transaction_costs

        # Calculate cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        buy_hold_cumulative = (1 + returns).cumprod()

        # Performance metrics (clamp to sane range)
        total_return = float(
            np.clip((strategy_cumulative.iloc[-1] - 1) * 100, -500.0, 500.0)
        )
        buy_hold_return = float(
            np.clip((buy_hold_cumulative.iloc[-1] - 1) * 100, -500.0, 500.0)
        )

        # Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe(strategy_returns)

        # Max Drawdown
        max_drawdown = self._calculate_max_drawdown(strategy_cumulative)

        # Win rate
        win_rate = self._calculate_win_rate(strategy_returns)

        # Trades count
        num_trades = positions_diff.sum()

        # Annualized return via CAGR formula, clamped to avoid extreme values
        n_periods = max(len(strategy_returns), 1)
        cum_final = max(strategy_cumulative.iloc[-1], 1e-9)  # avoid negative/zero
        exponent = 252.0 / n_periods
        # Cap exponent to avoid astronomical blow-up on short periods
        if exponent > 4.0:
            exponent = 4.0
        annualized_return = (cum_final**exponent - 1) * 100
        # Clamp to sane range
        annualized_return = float(np.clip(annualized_return, -999.0, 999.0))
        profit_factor = self._calculate_profit_factor(strategy_returns)
        calmar_ratio = self._calculate_calmar(annualized_return, max_drawdown)

        results = {
            "strategy_name": strategy_name,
            "total_return": total_return,
            "total_return_pct": total_return,
            "annualized_return_pct": float(annualized_return),
            "buy_hold_return": buy_hold_return,
            "excess_return": total_return - buy_hold_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown,
            "win_rate": win_rate,
            "win_rate_pct": win_rate,
            "num_trades": int(round(float(num_trades))),
            "total_trades": int(round(float(num_trades))),
            "profit_factor": profit_factor,
            "calmar_ratio": calmar_ratio,
            "strategy_cumulative": strategy_cumulative,
            "buy_hold_cumulative": buy_hold_cumulative,
            "strategy_returns": strategy_returns,
        }

        return results

    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) == 0:
            return 0

        excess_returns = returns - risk_free_rate / 252  # Daily rate
        if excess_returns.std() == 0:
            return 0

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return float(sharpe)

    @staticmethod
    def _calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min()) * 100

    @staticmethod
    def _calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate"""
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])

        if total_days == 0:
            return 0

        return (winning_days / total_days) * 100

    @staticmethod
    def _calculate_profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor = gross profit / gross loss."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = returns[returns < 0].abs().sum()
        if gross_loss == 0:
            return float(gross_profit > 0) * 10.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def _calculate_calmar(
        annualized_return_pct: float, max_drawdown_pct: float
    ) -> float:
        """Calculate Calmar ratio using annualized return and max drawdown."""
        if max_drawdown_pct == 0:
            return 0.0
        return float(annualized_return_pct / abs(max_drawdown_pct))

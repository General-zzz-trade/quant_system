"""Backtest result analysis — performance metrics and statistics.

Computes standard quantitative performance metrics from a return series
or equity curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int    # bars
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    turnover: float              # average daily turnover
    skewness: float
    kurtosis: float


def compute_metrics(
    returns: Sequence[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    trade_pnls: Optional[Sequence[float]] = None,
) -> PerformanceMetrics:
    """Compute performance metrics from a return series.

    Args:
        returns: Period returns (e.g., daily).
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods in a year (252 for daily).
        trade_pnls: Individual trade PnLs for win rate / profit factor.
    """
    n = len(returns)
    if n == 0:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, calmar_ratio=0, win_rate=0,
            profit_factor=0, avg_win=0, avg_loss=0, total_trades=0,
            turnover=0, skewness=0, kurtosis=0,
        )

    # Total and annualized return
    cum = 1.0
    for r in returns:
        cum *= (1 + r)
    total_return = cum - 1.0
    ann_return = (cum ** (periods_per_year / max(n, 1))) - 1.0

    # Volatility
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / max(n - 1, 1)
    vol = math.sqrt(variance)
    ann_vol = vol * math.sqrt(periods_per_year)

    # Sharpe
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = [r - rf_per_period for r in returns]
    excess_mean = sum(excess) / n
    sharpe = (excess_mean / vol * math.sqrt(periods_per_year)) if vol > 0 else 0.0

    # Sortino
    downside = [r for r in excess if r < 0]
    if downside:
        down_var = sum(r ** 2 for r in downside) / len(downside)
        down_dev = math.sqrt(down_var)
        sortino = (excess_mean / down_dev * math.sqrt(periods_per_year)) if down_dev > 0 else 0.0
    else:
        sortino = float("inf") if excess_mean > 0 else 0.0

    # Max drawdown
    peak = 1.0
    equity = 1.0
    max_dd = 0.0
    dd_start = 0
    max_dd_duration = 0
    current_dd_start = 0

    for i, r in enumerate(returns):
        equity *= (1 + r)
        if equity > peak:
            duration = i - current_dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration
            peak = equity
            current_dd_start = i
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    # Calmar
    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    # Trade-level metrics
    if trade_pnls is not None and len(trade_pnls) > 0:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_trades = len(trade_pnls)
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0
        total_trades = 0

    # Higher moments
    if n >= 3:
        skew = sum((r - mean_r) ** 3 for r in returns) / (n * vol ** 3) if vol > 0 else 0.0
        kurt = sum((r - mean_r) ** 4 for r in returns) / (n * vol ** 4) - 3.0 if vol > 0 else 0.0
    else:
        skew = kurt = 0.0

    # Turnover (approximate: sum of absolute returns as proxy)
    turnover = sum(abs(r) for r in returns) / max(n, 1) * periods_per_year

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=total_trades,
        turnover=turnover,
        skewness=skew,
        kurtosis=kurt,
    )


def compute_rolling_sharpe(
    returns: Sequence[float],
    window: int = 60,
    periods_per_year: int = 252,
) -> list[Optional[float]]:
    """Compute rolling Sharpe ratio."""
    result: list[Optional[float]] = []
    for i in range(len(returns)):
        if i < window - 1:
            result.append(None)
            continue
        w = returns[i - window + 1: i + 1]
        mean = sum(w) / len(w)
        var = sum((r - mean) ** 2 for r in w) / max(len(w) - 1, 1)
        std = math.sqrt(var)
        if std > 0:
            result.append(mean / std * math.sqrt(periods_per_year))
        else:
            result.append(0.0)
    return result

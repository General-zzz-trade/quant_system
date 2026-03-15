# execution/sim/live_comparison.py
"""Backtest vs Live comparison framework.

Compares backtest predictions with actual live trading results to
measure backtest fidelity. The gap between backtest and live is the
"reality discount" that should be applied to future backtest results.

Usage:
    from execution.sim.live_comparison import BacktestLiveComparison

    comp = BacktestLiveComparison()
    comp.add_backtest_trade(bar=100, side=1, entry=2100, exit=2150, pnl=47.5)
    comp.add_live_trade(bar=100, side=1, entry=2101.5, exit=2148.2, pnl=44.1)
    report = comp.compute_report()
    print(report)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComparisonTrade:
    bar: int
    side: int
    entry_price: float
    exit_price: float
    pnl: float
    fees: float = 0.0
    slippage: float = 0.0
    source: str = ""  # "backtest" or "live"


@dataclass
class ComparisonReport:
    """Report comparing backtest vs live performance."""
    n_backtest_trades: int = 0
    n_live_trades: int = 0
    n_matched: int = 0              # trades that appear in both

    backtest_total_pnl: float = 0.0
    live_total_pnl: float = 0.0
    pnl_gap: float = 0.0           # backtest - live
    pnl_gap_pct: float = 0.0       # gap as % of backtest PnL

    backtest_win_rate: float = 0.0
    live_win_rate: float = 0.0

    avg_entry_slippage_bps: float = 0.0  # avg |backtest_entry - live_entry| / price
    avg_exit_slippage_bps: float = 0.0

    avg_pnl_per_trade_bt: float = 0.0
    avg_pnl_per_trade_live: float = 0.0

    reality_discount: float = 0.0  # live/backtest ratio (0.8 = 80% of backtest survives)


class BacktestLiveComparison:
    """Compares backtest predictions with actual live trading results."""

    def __init__(self) -> None:
        self._bt_trades: list[ComparisonTrade] = []
        self._live_trades: list[ComparisonTrade] = []

    def add_backtest_trade(self, bar: int, side: int, entry: float, exit_price: float,
                           pnl: float, fees: float = 0, slippage: float = 0) -> None:
        self._bt_trades.append(ComparisonTrade(
            bar=bar, side=side, entry_price=entry, exit_price=exit_price,
            pnl=pnl, fees=fees, slippage=slippage, source="backtest",
        ))

    def add_live_trade(self, bar: int, side: int, entry: float, exit_price: float,
                       pnl: float, fees: float = 0, slippage: float = 0) -> None:
        self._live_trades.append(ComparisonTrade(
            bar=bar, side=side, entry_price=entry, exit_price=exit_price,
            pnl=pnl, fees=fees, slippage=slippage, source="live",
        ))

    def compute_report(self) -> ComparisonReport:
        """Compute comparison report between backtest and live trades."""
        report = ComparisonReport()
        report.n_backtest_trades = len(self._bt_trades)
        report.n_live_trades = len(self._live_trades)

        if not self._bt_trades or not self._live_trades:
            return report

        # Match trades by bar and side
        bt_by_bar = {(t.bar, t.side): t for t in self._bt_trades}
        live_by_bar = {(t.bar, t.side): t for t in self._live_trades}
        matched_keys = set(bt_by_bar.keys()) & set(live_by_bar.keys())
        report.n_matched = len(matched_keys)

        # PnL comparison
        bt_pnls = [t.pnl for t in self._bt_trades]
        live_pnls = [t.pnl for t in self._live_trades]
        report.backtest_total_pnl = sum(bt_pnls)
        report.live_total_pnl = sum(live_pnls)
        report.pnl_gap = report.backtest_total_pnl - report.live_total_pnl
        if report.backtest_total_pnl != 0:
            report.pnl_gap_pct = report.pnl_gap / abs(report.backtest_total_pnl) * 100

        # Win rates
        report.backtest_win_rate = sum(1 for p in bt_pnls if p > 0) / max(len(bt_pnls), 1)
        report.live_win_rate = sum(1 for p in live_pnls if p > 0) / max(len(live_pnls), 1)

        # Entry/exit slippage on matched trades
        entry_slips = []
        exit_slips = []
        for key in matched_keys:
            bt = bt_by_bar[key]
            live = live_by_bar[key]
            if bt.entry_price > 0:
                entry_slips.append(abs(bt.entry_price - live.entry_price) / bt.entry_price * 10000)
            if bt.exit_price > 0:
                exit_slips.append(abs(bt.exit_price - live.exit_price) / bt.exit_price * 10000)

        report.avg_entry_slippage_bps = float(np.mean(entry_slips)) if entry_slips else 0
        report.avg_exit_slippage_bps = float(np.mean(exit_slips)) if exit_slips else 0

        # Per-trade averages
        report.avg_pnl_per_trade_bt = sum(bt_pnls) / max(len(bt_pnls), 1)
        report.avg_pnl_per_trade_live = sum(live_pnls) / max(len(live_pnls), 1)

        # Reality discount
        if report.backtest_total_pnl > 0:
            report.reality_discount = report.live_total_pnl / report.backtest_total_pnl
        elif report.backtest_total_pnl < 0 and report.live_total_pnl < 0:
            report.reality_discount = report.backtest_total_pnl / report.live_total_pnl

        return report

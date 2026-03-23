"""Unified strategy interface — same contract for backtest, paper, and live.

All strategies implement StrategyRunner protocol. The runner/main.py
dispatches to the right strategy + mode combination.

Usage:
    python3 -m runner.main --strategy alpha --mode live
    python3 -m runner.main --strategy alpha --mode paper
    python3 -m runner.main --strategy alpha --mode backtest --data data_files/BTCUSDT_1h.csv
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Protocol, runtime_checkable


class RunMode(Enum):
    LIVE = "live"
    PAPER = "paper"       # dry_run=True, real data feed
    BACKTEST = "backtest"  # historical data, no exchange


@runtime_checkable
class StrategyRunner(Protocol):
    """Unified strategy interface.

    Every strategy (alpha, HFT, MM) must implement these methods.
    The runner framework handles data feed, mode switching, and lifecycle.
    """

    @property
    def symbol(self) -> str:
        """Trading symbol (e.g. BTCUSDT)."""
        ...

    def warmup(self, limit: int) -> int:
        """Warm up with historical bars. Returns bars processed."""
        ...

    def process_bar(self, bar: dict) -> dict:
        """Process one OHLCV bar. Returns action/signal dict."""
        ...

    def stop(self) -> None:
        """Graceful shutdown."""
        ...


class BacktestEngine:
    """Run a StrategyRunner on historical data.

    Feeds bars from CSV to strategy.process_bar() and tracks PnL.
    Same signal logic as live — only data source differs.

    Usage:
        engine = BacktestEngine(strategy, data_path="data_files/BTCUSDT_1h.csv")
        result = engine.run()
        print(f"Sharpe: {result['sharpe']}, Return: {result['return_pct']}%")
    """

    def __init__(
        self,
        strategy: StrategyRunner,
        data_path: str,
        initial_equity: float = 100.0,
    ):
        self._strategy = strategy
        self._data_path = data_path
        self._initial_equity = initial_equity

    def run(self) -> Dict[str, Any]:
        """Run backtest and return metrics."""
        import numpy as np
        import pandas as pd

        df = pd.read_csv(self._data_path)
        closes = df["close"].values

        # Warmup
        warmup_bars = min(800, len(df) // 2)
        results = []

        for i in range(len(df)):
            bar = {
                "close": float(df["close"].iloc[i]),
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "open": float(df["open"].iloc[i]),
                "volume": float(df["volume"].iloc[i]),
            }
            result = self._strategy.process_bar(bar)
            results.append(result)

        # Compute metrics from results
        signals = [r.get("signal", 0) for r in results]
        _equity = self._initial_equity  # noqa: F841
        trades = []
        pos = 0
        entry_idx = 0

        for i in range(warmup_bars, len(signals)):
            sig = signals[i]
            if pos != 0 and sig != pos:
                # Close
                pnl_pct = pos * (closes[i] - closes[entry_idx]) / closes[entry_idx]
                trades.append(pnl_pct)
                pos = 0
            if pos == 0 and sig != 0:
                pos = sig
                entry_idx = i

        if not trades:
            return {"sharpe": 0, "return_pct": 0, "trades": 0, "win_rate": 0}

        arr = np.array(trades)
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 0 else 0
        return_pct = float(np.sum(arr) * 100)
        win_rate = float(np.mean(arr > 0) * 100)

        return {
            "sharpe": round(sharpe, 2),
            "return_pct": round(return_pct, 1),
            "trades": len(trades),
            "win_rate": round(win_rate, 1),
        }

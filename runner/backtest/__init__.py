"""Backtest support utilities -- adapter, CSV I/O, metrics.

These modules are used by runner/backtest_runner.py (the backtest entry point)
and by other runners that need backtest execution capabilities.

Entry point: runner/backtest_runner.py
Support: runner/backtest/adapter.py, csv_io.py, metrics.py, pnl_compare.py
"""
from .csv_io import OhlcvBar, iter_ohlcv_csv, _parse_ts, _dec, _to_key
from .adapter import BacktestExecutionAdapter, _sign, _make_id
from .metrics import (
    EquityPoint,
    _max_drawdown,
    _build_trades_from_fills,
    _build_summary,
    _json_safe,
    _safe_dec,
)

__all__ = [
    "OhlcvBar",
    "iter_ohlcv_csv",
    "_parse_ts",
    "_dec",
    "_to_key",
    "BacktestExecutionAdapter",
    "_sign",
    "_make_id",
    "EquityPoint",
    "_max_drawdown",
    "_build_trades_from_fills",
    "_build_summary",
    "_json_safe",
    "_safe_dec",
]

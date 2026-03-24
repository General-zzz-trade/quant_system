"""PnLTracker — unified P&L tracking, Rust-backed via RustPnLTracker."""
from __future__ import annotations

import logging
import math
from collections import defaultdict

from _quant_hotpath import RustPnLTracker

logger = logging.getLogger(__name__)


class PnLTracker:
    """Unified P&L tracking. Delegates core computation to RustPnLTracker."""

    def __init__(self):
        self._inner = RustPnLTracker()
        self._trades: list[dict] = []
        self._symbol_pnl: dict[str, float] = defaultdict(float)
        self._symbol_trades: dict[str, list[float]] = defaultdict(list)
        self._horizon_pnl: dict[int, float] = defaultdict(float)
        self._horizon_trades: dict[int, list[float]] = defaultdict(list)

    @property
    def total_pnl(self) -> float:
        return self._inner.total_pnl

    @property
    def peak_equity(self) -> float:
        return self._inner.peak_equity

    @property
    def trade_count(self) -> int:
        return int(self._inner.trade_count)

    @property
    def win_count(self) -> int:
        return int(self._inner.win_count)

    @property
    def trades(self) -> list[dict]:
        return self._trades

    @property
    def win_rate(self) -> float:
        return self._inner.win_rate

    @property
    def drawdown_pct(self) -> float:
        return self._inner.drawdown_pct

    def record_close(self, symbol: str, side: int, entry_price: float,
                     exit_price: float, size: float, reason: str = "",
                     *, horizon: int | None = None) -> dict:
        """Record a position close. Returns trade info dict."""
        for label, price in [("entry_price", entry_price), ("exit_price", exit_price)]:
            if price <= 0 or math.isnan(price):
                logger.warning("record_close %s: invalid %s=%.6f, skipping", symbol, label, price)
                return {"symbol": symbol, "side": side, "entry": entry_price,
                        "exit": exit_price, "size": size, "pnl_usd": 0.0,
                        "pnl_pct": 0.0, "reason": reason, "total_pnl": self.total_pnl,
                        "trade_count": self.trade_count, "error": f"invalid_{label}"}

        trade = self._inner.record_close(symbol, side, entry_price, exit_price, size, reason)
        if horizon is not None:
            trade["horizon"] = horizon
        self._trades.append(trade)
        if len(self._trades) > 100:
            self._trades = self._trades[-100:]
        self._record_attribution(symbol, trade.get("pnl_usd", 0.0), horizon)
        return trade

    def _record_attribution(self, symbol: str, pnl_usd: float, horizon: int | None) -> None:
        self._symbol_pnl[symbol] += pnl_usd
        self._symbol_trades[symbol].append(pnl_usd)
        if horizon is not None:
            self._horizon_pnl[horizon] += pnl_usd
            self._horizon_trades[horizon].append(pnl_usd)

    @property
    def pnl_by_symbol(self) -> dict[str, float]:
        return dict(self._symbol_pnl)

    @property
    def pnl_by_horizon(self) -> dict[int, float]:
        return dict(self._horizon_pnl)

    @property
    def best_symbol(self) -> str:
        if not self._symbol_pnl:
            return ""
        return max(self._symbol_pnl, key=self._symbol_pnl.get)  # type: ignore[arg-type]

    @property
    def worst_symbol(self) -> str:
        if not self._symbol_pnl:
            return ""
        return min(self._symbol_pnl, key=self._symbol_pnl.get)  # type: ignore[arg-type]

    def per_symbol_sharpe(self, symbol: str) -> float:
        """Rolling Sharpe ratio for a single symbol (annualised, 365-day)."""
        trades = self._symbol_trades.get(symbol, [])
        if len(trades) < 2:
            return 0.0
        mean = sum(trades) / len(trades)
        var = sum((t - mean) ** 2 for t in trades) / (len(trades) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        return (mean / std) * math.sqrt(365) if std > 0 else 0.0

    def summary(self) -> dict:
        """Full attribution summary for logging/monitoring."""
        base = self._inner.summary()
        base["pnl_by_symbol"] = self.pnl_by_symbol
        base["pnl_by_horizon"] = self.pnl_by_horizon
        base["best_symbol"] = self.best_symbol
        base["worst_symbol"] = self.worst_symbol
        return base

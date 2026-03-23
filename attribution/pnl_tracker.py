"""PnLTracker — unified P&L tracking for all alpha sources."""
from __future__ import annotations

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from _quant_hotpath import RustPnLTracker as _RustPnLTracker
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class PnLTracker:
    """Unified P&L tracking. All trade closes go through here.

    Replaces duplicated PnL logic across AlphaRunner, PortfolioManager,
    and PortfolioCombiner. Tracks total PnL, win rate, peak equity,
    drawdown, and recent trade log.

    Delegates to RustPnLTracker when _quant_hotpath is available,
    falls back to pure Python otherwise.
    """

    def __init__(self):
        if _HAS_RUST:
            self._inner = _RustPnLTracker()
            self._use_rust = True
            self._trades: list[dict] = []  # Python-side mirror for .trades access
        else:
            self._use_rust = False
            # Pure-Python state (only populated when Rust unavailable)
            self._total_pnl: float = 0.0
            self._peak_equity: float = 0.0
            self._trade_count: int = 0
            self._win_count: int = 0
            self._trades = []  # recent trade log

        # Per-symbol attribution (always Python-side, works with both Rust and Python)
        self._symbol_pnl: dict[str, float] = defaultdict(float)
        self._symbol_trades: dict[str, list[float]] = defaultdict(list)
        # Per-horizon attribution
        self._horizon_pnl: dict[int, float] = defaultdict(float)
        self._horizon_trades: dict[int, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Attribute accessors (expose same names callers expect)
    # ------------------------------------------------------------------

    @property
    def total_pnl(self) -> float:
        if self._use_rust:
            return self._inner.total_pnl
        return self._total_pnl

    @property
    def peak_equity(self) -> float:
        if self._use_rust:
            return self._inner.peak_equity
        return self._peak_equity

    @property
    def trade_count(self) -> int:
        if self._use_rust:
            return int(self._inner.trade_count)
        return self._trade_count

    @property
    def win_count(self) -> int:
        if self._use_rust:
            return int(self._inner.win_count)
        return self._win_count

    @property
    def trades(self) -> list[dict]:
        """Recent trade log (last 100). Always returns a Python list."""
        return self._trades

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def record_close(self, symbol: str, side: int, entry_price: float,
                     exit_price: float, size: float, reason: str = "",
                     *, horizon: int | None = None) -> dict:
        """Record a position close. Returns trade info dict.

        Args:
            symbol: exchange symbol (e.g. "ETHUSDT")
            side: position side that was closed (+1 = was long, -1 = was short)
            entry_price: average entry price
            exit_price: exit/close price
            size: position size in base asset
            reason: close reason (e.g. "signal_change", "stop_loss", "drawdown")
            horizon: optional prediction horizon in bars (e.g. 24, 96)

        Returns:
            dict with pnl_usd, pnl_pct, total_pnl, trade_count, win_count
        """
        if entry_price <= 0 or math.isnan(entry_price):
            logger.warning("record_close %s: invalid entry_price=%.6f, skipping", symbol, entry_price)
            return {"symbol": symbol, "side": side, "entry": entry_price,
                    "exit": exit_price, "size": size, "pnl_usd": 0.0,
                    "pnl_pct": 0.0, "reason": reason, "total_pnl": self.total_pnl,
                    "trade_count": self.trade_count, "error": "invalid_entry_price"}
        if exit_price <= 0 or math.isnan(exit_price):
            logger.warning("record_close %s: invalid exit_price=%.6f, skipping", symbol, exit_price)
            return {"symbol": symbol, "side": side, "entry": entry_price,
                    "exit": exit_price, "size": size, "pnl_usd": 0.0,
                    "pnl_pct": 0.0, "reason": reason, "total_pnl": self.total_pnl,
                    "trade_count": self.trade_count, "error": "invalid_exit_price"}

        if self._use_rust:
            trade = self._inner.record_close(symbol, side, entry_price, exit_price, size, reason)
            if horizon is not None:
                trade["horizon"] = horizon
            self._trades.append(trade)
            if len(self._trades) > 100:
                self._trades = self._trades[-100:]
            pnl_usd = trade.get("pnl_usd", 0.0)
            self._record_attribution(symbol, pnl_usd, horizon)
            return trade

        if side == 1:  # was long
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # was short
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        pnl_usd = pnl_pct / 100 * entry_price * size
        self._total_pnl += pnl_usd
        self._trade_count += 1
        if pnl_usd > 0:
            self._win_count += 1

        self._peak_equity = max(self._peak_equity, self._total_pnl)

        trade = {
            "symbol": symbol, "side": side, "entry": entry_price,
            "exit": exit_price, "size": size, "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct, "reason": reason,
            "total_pnl": self._total_pnl, "trade_count": self._trade_count,
        }
        if horizon is not None:
            trade["horizon"] = horizon
        self._trades.append(trade)
        if len(self._trades) > 100:
            self._trades = self._trades[-100:]

        self._record_attribution(symbol, pnl_usd, horizon)
        return trade

    @property
    def win_rate(self) -> float:
        """Win rate as percentage (0-100)."""
        if self._use_rust:
            return self._inner.win_rate
        return self._win_count / self._trade_count * 100 if self._trade_count > 0 else 0

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak equity as percentage."""
        if self._use_rust:
            return self._inner.drawdown_pct
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._total_pnl) / self._peak_equity * 100

    # ------------------------------------------------------------------
    # Attribution internals
    # ------------------------------------------------------------------

    def _record_attribution(self, symbol: str, pnl_usd: float,
                            horizon: int | None) -> None:
        """Update per-symbol and per-horizon accumulators."""
        self._symbol_pnl[symbol] += pnl_usd
        self._symbol_trades[symbol].append(pnl_usd)
        if horizon is not None:
            self._horizon_pnl[horizon] += pnl_usd
            self._horizon_trades[horizon].append(pnl_usd)

    # ------------------------------------------------------------------
    # Per-symbol attribution
    # ------------------------------------------------------------------

    @property
    def pnl_by_symbol(self) -> dict[str, float]:
        """Cumulative PnL keyed by symbol."""
        return dict(self._symbol_pnl)

    @property
    def pnl_by_horizon(self) -> dict[int, float]:
        """Cumulative PnL keyed by prediction horizon (bars)."""
        return dict(self._horizon_pnl)

    @property
    def best_symbol(self) -> str:
        """Symbol with highest cumulative PnL. Empty string if no trades."""
        if not self._symbol_pnl:
            return ""
        return max(self._symbol_pnl, key=self._symbol_pnl.get)  # type: ignore[arg-type]

    @property
    def worst_symbol(self) -> str:
        """Symbol with lowest cumulative PnL. Empty string if no trades."""
        if not self._symbol_pnl:
            return ""
        return min(self._symbol_pnl, key=self._symbol_pnl.get)  # type: ignore[arg-type]

    def per_symbol_sharpe(self, symbol: str) -> float:
        """Rolling Sharpe ratio for a single symbol (annualised, 365-day).

        Returns 0.0 if fewer than 2 trades recorded for the symbol.
        """
        trades = self._symbol_trades.get(symbol, [])
        if len(trades) < 2:
            return 0.0
        mean = sum(trades) / len(trades)
        var = sum((t - mean) ** 2 for t in trades) / (len(trades) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0.0:
            return 0.0
        # Annualise assuming ~1 trade per day baseline; caller can adjust
        return (mean / std) * math.sqrt(365)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return full attribution summary for logging/monitoring."""
        if self._use_rust:
            base = self._inner.summary()
        else:
            base = {
                "total_pnl": self._total_pnl, "trades": self._trade_count,
                "wins": self._win_count, "win_rate": self.win_rate,
                "peak": self._peak_equity, "drawdown": self.drawdown_pct,
            }
        # Extend with attribution data
        base["pnl_by_symbol"] = self.pnl_by_symbol
        base["pnl_by_horizon"] = self.pnl_by_horizon
        base["best_symbol"] = self.best_symbol
        base["worst_symbol"] = self.worst_symbol
        return base

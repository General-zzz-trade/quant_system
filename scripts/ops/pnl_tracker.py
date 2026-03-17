"""PnLTracker — unified P&L tracking for all alpha sources."""
from __future__ import annotations

import logging

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

    @trades.setter
    def trades(self, value: list[dict]) -> None:
        self._trades = value

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def record_close(self, symbol: str, side: int, entry_price: float,
                     exit_price: float, size: float, reason: str = "") -> dict:
        """Record a position close. Returns trade info dict.

        Args:
            symbol: exchange symbol (e.g. "ETHUSDT")
            side: position side that was closed (+1 = was long, -1 = was short)
            entry_price: average entry price
            exit_price: exit/close price
            size: position size in base asset
            reason: close reason (e.g. "signal_change", "stop_loss", "drawdown")

        Returns:
            dict with pnl_usd, pnl_pct, total_pnl, trade_count, win_count
        """
        if self._use_rust:
            trade = self._inner.record_close(symbol, side, entry_price, exit_price, size, reason)
            self._trades.append(trade)
            if len(self._trades) > 100:
                self._trades = self._trades[-100:]
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
        self._trades.append(trade)
        if len(self._trades) > 100:
            self._trades = self._trades[-100:]

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

    def summary(self) -> dict:
        """Return summary dict for logging/monitoring."""
        if self._use_rust:
            return self._inner.summary()
        return {
            "total_pnl": self._total_pnl, "trades": self._trade_count,
            "wins": self._win_count, "win_rate": self.win_rate,
            "peak": self._peak_equity, "drawdown": self.drawdown_pct,
        }

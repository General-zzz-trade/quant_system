"""PnLTracker — unified P&L tracking for all alpha sources."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PnLTracker:
    """Unified P&L tracking. All trade closes go through here.

    Replaces duplicated PnL logic across AlphaRunner, PortfolioManager,
    and PortfolioCombiner. Tracks total PnL, win rate, peak equity,
    drawdown, and recent trade log.
    """

    def __init__(self):
        self.total_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.trade_count: int = 0
        self.win_count: int = 0
        self.trades: list[dict] = []  # recent trade log

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
        if side == 1:  # was long
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # was short
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        pnl_usd = pnl_pct / 100 * entry_price * size
        self.total_pnl += pnl_usd
        self.trade_count += 1
        if pnl_usd > 0:
            self.win_count += 1

        self.peak_equity = max(self.peak_equity, self.total_pnl)

        trade = {
            "symbol": symbol, "side": side, "entry": entry_price,
            "exit": exit_price, "size": size, "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct, "reason": reason,
            "total_pnl": self.total_pnl, "trade_count": self.trade_count,
        }
        self.trades.append(trade)
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]

        return trade

    @property
    def win_rate(self) -> float:
        """Win rate as percentage (0-100)."""
        return self.win_count / self.trade_count * 100 if self.trade_count > 0 else 0

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak equity as percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.total_pnl) / self.peak_equity * 100

    def summary(self) -> dict:
        """Return summary dict for logging/monitoring."""
        return {
            "total_pnl": self.total_pnl, "trades": self.trade_count,
            "wins": self.win_count, "win_rate": self.win_rate,
            "peak": self.peak_equity, "drawdown": self.drawdown_pct,
        }

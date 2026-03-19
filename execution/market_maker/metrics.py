"""Market maker performance metrics and reporting."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """Single fill event."""
    ts: float
    side: str
    qty: float
    price: float
    rpnl: float


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics summary."""
    ts: float = 0.0
    uptime_s: float = 0.0
    total_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    total_pnl: float = 0.0
    maker_rebate: float = 0.0
    avg_spread_bps: float = 0.0
    fill_rate: float = 0.0        # fills per minute
    inventory_qty: float = 0.0
    max_drawdown: float = 0.0
    quotes_sent: int = 0
    cancels_sent: int = 0


class MetricsCollector:
    """Collect and compute market maker performance metrics."""

    def __init__(self, maker_rebate_bps: float = -1.0) -> None:
        """
        Args:
            maker_rebate_bps: Maker fee in bps (negative = rebate).
                              Binance ETHUSDT: -1.0 bps = -0.01% = rebate.
        """
        self._rebate_bps = maker_rebate_bps
        self._start_time = time.time()
        self._fills: list[FillRecord] = []
        self._spreads: list[float] = []
        self._peak_pnl: float = 0.0
        self._max_dd: float = 0.0
        self._quotes_sent: int = 0
        self._cancels_sent: int = 0
        self._total_volume: float = 0.0

    def record_fill(self, side: str, qty: float, price: float, rpnl: float) -> None:
        self._fills.append(FillRecord(
            ts=time.time(), side=side, qty=qty, price=price, rpnl=rpnl,
        ))
        notional = qty * price
        self._total_volume += notional
        # Track drawdown
        cum_pnl = sum(f.rpnl for f in self._fills)
        self._peak_pnl = max(self._peak_pnl, cum_pnl)
        self._max_dd = min(self._max_dd, cum_pnl - self._peak_pnl)

    def record_spread(self, spread_bps: float) -> None:
        self._spreads.append(spread_bps)
        # Keep last 10000
        if len(self._spreads) > 10000:
            self._spreads = self._spreads[-5000:]

    def record_quote(self) -> None:
        self._quotes_sent += 1

    def record_cancel(self) -> None:
        self._cancels_sent += 1

    @property
    def maker_rebate_earned(self) -> float:
        """Total maker rebate earned from fill volume."""
        return self._total_volume * abs(self._rebate_bps) * 1e-4

    def snapshot(
        self,
        inventory_qty: float = 0.0,
        unrealised_pnl: float = 0.0,
    ) -> MetricsSnapshot:
        now = time.time()
        uptime = now - self._start_time
        rpnl = sum(f.rpnl for f in self._fills)
        fill_rate = len(self._fills) / max(uptime / 60, 1.0)
        avg_spread = (
            sum(self._spreads) / len(self._spreads) if self._spreads else 0.0
        )

        return MetricsSnapshot(
            ts=now,
            uptime_s=uptime,
            total_fills=len(self._fills),
            buy_fills=sum(1 for f in self._fills if f.side == "buy"),
            sell_fills=sum(1 for f in self._fills if f.side == "sell"),
            realised_pnl=rpnl,
            unrealised_pnl=unrealised_pnl,
            total_pnl=rpnl + unrealised_pnl,
            maker_rebate=self.maker_rebate_earned,
            avg_spread_bps=avg_spread,
            fill_rate=fill_rate,
            inventory_qty=inventory_qty,
            max_drawdown=self._max_dd,
            quotes_sent=self._quotes_sent,
            cancels_sent=self._cancels_sent,
        )

    def log_summary(self, inventory_qty: float = 0.0, unrealised_pnl: float = 0.0) -> None:
        s = self.snapshot(inventory_qty, unrealised_pnl)
        log.info(
            "METRICS uptime=%.0fs fills=%d (B%d/S%d) rpnl=$%.4f "
            "rebate=$%.4f spread=%.1fbps fill_rate=%.1f/min inv=%.4f dd=$%.4f",
            s.uptime_s,
            s.total_fills,
            s.buy_fills,
            s.sell_fills,
            s.realised_pnl,
            s.maker_rebate,
            s.avg_spread_bps,
            s.fill_rate,
            s.inventory_qty,
            s.max_drawdown,
        )

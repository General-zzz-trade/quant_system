"""HFT risk checker — pre-trade risk controls for tick-level strategies."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Tuple

from engine.tick_engine import HFTOrder
from state.shared_position import SharedPositionStore


@dataclass(frozen=True, slots=True)
class HFTRiskConfig:
    max_position_notional: float = 10_000.0  # USD
    max_orders_per_second: int = 8
    max_daily_loss: float = 500.0  # USD
    cooldown_after_loss_s: float = 60.0


@dataclass
class HFTRiskChecker:
    """Pre-trade risk gate for HFT orders.

    Checks:
    1. Position notional limit
    2. Order rate limit (orders/second)
    3. Daily loss limit with cooldown
    """

    cfg: HFTRiskConfig = field(default_factory=HFTRiskConfig)
    position_store: Optional[SharedPositionStore] = None
    reference_prices: dict[str, float] = field(default_factory=dict)

    _order_timestamps: list[float] = field(default_factory=list)
    _daily_pnl: float = 0.0
    _last_loss_time: Optional[float] = None
    _killed: bool = False

    def kill(self) -> None:
        """Emergency kill switch — reject all orders."""
        self._killed = True

    def unkill(self) -> None:
        self._killed = False

    def update_pnl(self, pnl_delta: float) -> None:
        """Called on each fill to track daily PnL."""
        self._daily_pnl += pnl_delta
        if pnl_delta < 0:
            self._last_loss_time = time.monotonic()

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._last_loss_time = None

    def check(self, order: HFTOrder) -> Tuple[bool, str]:
        """Check if an order passes risk controls.

        Returns (allowed, reason).
        """
        if self._killed:
            return False, "kill_switch_active"

        # Daily loss limit
        if self._daily_pnl < -self.cfg.max_daily_loss:
            return False, f"daily_loss_exceeded: {self._daily_pnl:.2f}"

        # Loss cooldown
        if self._last_loss_time is not None:
            elapsed = time.monotonic() - self._last_loss_time
            if (
                elapsed < self.cfg.cooldown_after_loss_s
                and self._daily_pnl < -self.cfg.max_daily_loss * 0.5
            ):
                return False, f"loss_cooldown: {elapsed:.1f}s"

        # Order rate limit
        now = time.monotonic()
        cutoff = now - 1.0
        self._order_timestamps = [
            t for t in self._order_timestamps if t > cutoff
        ]
        if len(self._order_timestamps) >= self.cfg.max_orders_per_second:
            return False, "rate_limit_exceeded"

        # Position notional limit
        if self.position_store is not None:
            pos = float(self.position_store.get_position(order.symbol))
            ref_price = self.reference_prices.get(order.symbol, 0.0)
            if ref_price > 0:
                # Check if this order would exceed max notional
                if order.side == "buy":
                    new_pos = pos + order.qty
                else:
                    new_pos = pos - order.qty
                notional = abs(new_pos) * ref_price
                if notional > self.cfg.max_position_notional:
                    if not order.reduce_only:
                        return False, f"position_notional_exceeded: {notional:.2f}"

        self._order_timestamps.append(now)
        return True, "ok"

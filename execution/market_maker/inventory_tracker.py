"""Inventory and PnL tracking for market maker."""

from __future__ import annotations

import time


class InventoryTracker:
    """Track net position, realised PnL, and enforce inventory limits.

    Side-blocking: when |net| exceeds limit, only allow orders that
    reduce exposure.
    """

    def __init__(self, max_notional: float = 50.0, daily_loss_limit: float = 10.0) -> None:
        self._max_notional = max_notional
        self._daily_loss_limit = daily_loss_limit

        # Position state
        self.net_qty: float = 0.0          # positive = long
        self.avg_entry: float = 0.0        # average entry price

        # PnL
        self.realised_pnl: float = 0.0     # session total
        self.unrealised_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self._day_start_ts: float = time.time()

        # Fill tracking
        self.total_fills: int = 0
        self.buy_fills: int = 0
        self.sell_fills: int = 0
        self._consecutive_losses: int = 0

    @property
    def net_notional(self) -> float:
        """Absolute notional exposure (uses avg_entry as ref)."""
        if self.avg_entry <= 0:
            return 0.0
        return abs(self.net_qty) * self.avg_entry

    def net_notional_at(self, ref_price: float) -> float:
        """Absolute notional exposure at a given reference price."""
        if ref_price <= 0:
            return 0.0
        return abs(self.net_qty) * ref_price

    @property
    def inventory_utilisation(self) -> float:
        """0.0 to 1.0+ fraction of max inventory used."""
        if self._max_notional <= 0:
            return 0.0
        return self.net_notional / self._max_notional

    def can_buy(self, ref_price: float) -> bool:
        """True if buying would not breach inventory limit.

        Uses ref_price (current mid) for notional calc, NOT avg_entry.
        This fixes the bug where avg_entry=0 made notional=0 → no blocking.
        """
        if ref_price <= 0:
            return False
        # If we're short, buying reduces exposure — always allowed
        if self.net_qty < 0:
            return True
        return (self.net_qty * ref_price) < self._max_notional

    def can_sell(self, ref_price: float) -> bool:
        """True if selling would not breach inventory limit.

        Uses ref_price (current mid) for notional calc.
        """
        if ref_price <= 0:
            return False
        if self.net_qty > 0:
            return True
        return (abs(self.net_qty) * ref_price) < self._max_notional

    def on_fill(self, side: str, qty: float, price: float) -> float:
        """Process a fill. Returns realised PnL from this fill (0 if opening)."""
        self.total_fills += 1
        if side == "buy":
            self.buy_fills += 1
        else:
            self.sell_fills += 1

        signed_qty = qty if side == "buy" else -qty
        rpnl = 0.0

        # Check if this fill reduces position (generates realised PnL)
        if self.net_qty != 0 and (
            (self.net_qty > 0 and signed_qty < 0) or
            (self.net_qty < 0 and signed_qty > 0)
        ):
            close_qty = min(abs(signed_qty), abs(self.net_qty))
            if self.net_qty > 0:
                rpnl = close_qty * (price - self.avg_entry)
            else:
                rpnl = close_qty * (self.avg_entry - price)
            self.realised_pnl += rpnl
            self.daily_pnl += rpnl
            if rpnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

        # Update position
        old_qty = self.net_qty
        new_qty = old_qty + signed_qty

        if new_qty == 0:
            self.avg_entry = 0.0
        elif (old_qty >= 0 and new_qty > 0 and signed_qty > 0):
            # Adding to long
            total_cost = self.avg_entry * old_qty + price * qty
            self.avg_entry = total_cost / new_qty
        elif (old_qty <= 0 and new_qty < 0 and signed_qty < 0):
            # Adding to short
            total_cost = self.avg_entry * abs(old_qty) + price * qty
            self.avg_entry = total_cost / abs(new_qty)
        elif abs(new_qty) > 0 and (
            (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0)
        ):
            # Position flipped
            self.avg_entry = price

        self.net_qty = new_qty
        return rpnl

    def update_unrealised(self, mark_price: float) -> None:
        """Update unrealised PnL at current mark price."""
        if self.net_qty == 0 or self.avg_entry <= 0:
            self.unrealised_pnl = 0.0
            return
        if self.net_qty > 0:
            self.unrealised_pnl = self.net_qty * (mark_price - self.avg_entry)
        else:
            self.unrealised_pnl = abs(self.net_qty) * (self.avg_entry - mark_price)

    @property
    def total_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl

    @property
    def hit_daily_limit(self) -> bool:
        return self.daily_pnl <= -self._daily_loss_limit

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def reset_daily(self) -> None:
        """Reset daily counters (call at UTC midnight or session start)."""
        self.daily_pnl = 0.0
        self._day_start_ts = time.time()
        self._consecutive_losses = 0

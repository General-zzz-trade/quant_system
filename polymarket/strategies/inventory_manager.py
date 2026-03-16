"""Inventory manager for Polymarket market-making.

Tracks YES/NO contract inventory and enforces position limits
to prevent excessive directional exposure near market expiry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class InventorySnapshot:
    """Point-in-time inventory state."""

    yes_qty: float
    no_qty: float
    net: float
    utilization: float  # |net| / max_inventory
    timestamp_ms: int = 0


class InventoryManager:
    """Manages inventory limits and expiry-aware quoting decisions.

    Parameters
    ----------
    max_inventory : float
        Maximum absolute net inventory (|YES - NO|).
    warn_pct : float
        Utilization threshold for switching to opposite-side-only quoting.
    """

    def __init__(
        self,
        max_inventory: float = 100.0,
        warn_pct: float = 0.80,
    ) -> None:
        if max_inventory <= 0:
            raise ValueError(f"max_inventory must be positive, got {max_inventory}")
        if not 0.0 < warn_pct <= 1.0:
            raise ValueError(f"warn_pct must be in (0, 1], got {warn_pct}")

        self.max_inventory = max_inventory
        self.warn_pct = warn_pct

        self._yes_qty: float = 0.0
        self._no_qty: float = 0.0
        self._history: List[InventorySnapshot] = []

    @property
    def net_inventory(self) -> float:
        """Net inventory: positive = long YES, negative = long NO."""
        return self._yes_qty - self._no_qty

    @property
    def utilization(self) -> float:
        """Fraction of max_inventory currently used."""
        return abs(self.net_inventory) / self.max_inventory

    @property
    def yes_qty(self) -> float:
        return self._yes_qty

    @property
    def no_qty(self) -> float:
        return self._no_qty

    def update(self, yes_qty: float, no_qty: float, timestamp_ms: int = 0) -> None:
        """Set absolute YES/NO quantities (from exchange state)."""
        self._yes_qty = yes_qty
        self._no_qty = no_qty
        snap = InventorySnapshot(
            yes_qty=yes_qty,
            no_qty=no_qty,
            net=self.net_inventory,
            utilization=self.utilization,
            timestamp_ms=timestamp_ms,
        )
        self._history.append(snap)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

    def add_fill(self, side: str, qty: float) -> None:
        """Record a fill. side is 'yes' or 'no'."""
        if side == "yes":
            self._yes_qty += qty
        elif side == "no":
            self._no_qty += qty
        else:
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")

    def should_quote_side(self, side: str) -> bool:
        """Whether we should quote on the given side.

        Rules:
        - Below warn_pct utilization: quote both sides.
        - At/above warn_pct: only quote the side that reduces inventory.
        - At 100% utilization: do not quote the side that increases inventory.
        """
        net = self.net_inventory
        util = self.utilization

        if util < self.warn_pct:
            return True

        # At high utilization, only allow quotes that reduce exposure
        if side == "yes":
            # Buying YES increases net; only allow if we're short (net < 0)
            return net < 0
        elif side == "no":
            # Buying NO decreases net; only allow if we're long (net > 0)
            return net > 0
        else:
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")

    def time_to_expiry_action(self, seconds_remaining: float) -> str:
        """Determine action based on time remaining until market resolution.

        Returns
        -------
        str
            One of:
            - "normal": continue quoting normally
            - "reduce_only": only place orders that reduce inventory
            - "cancel_all": cancel all open orders
            - "taker_reduce": aggressively take to flatten inventory
        """
        if seconds_remaining <= 0:
            return "cancel_all"
        if seconds_remaining < 30:
            return "cancel_all"
        if seconds_remaining < 60:
            if self.utilization > 0.1:
                return "taker_reduce"
            return "cancel_all"
        if seconds_remaining < 120:
            return "reduce_only"
        return "normal"

    def reset(self) -> None:
        """Reset inventory state (e.g., after market expiry)."""
        self._yes_qty = 0.0
        self._no_qty = 0.0
        self._history.clear()

    def snapshot(self) -> InventorySnapshot:
        """Current inventory snapshot."""
        return InventorySnapshot(
            yes_qty=self._yes_qty,
            no_qty=self._no_qty,
            net=self.net_inventory,
            utilization=self.utilization,
        )

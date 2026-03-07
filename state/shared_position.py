"""Thread-safe shared position store for bar + tick engine coordination."""
from __future__ import annotations

import threading
from decimal import Decimal
from typing import Dict


class SharedPositionStore:
    """Thread-safe position store shared between bar and tick engines.

    Uses RLock to allow re-entrant locking from the same thread.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._positions: Dict[str, Decimal] = {}

    def get_position(self, symbol: str) -> Decimal:
        with self._lock:
            return self._positions.get(symbol, Decimal("0"))

    def update_position(self, symbol: str, qty: Decimal) -> None:
        with self._lock:
            self._positions[symbol] = qty

    def add_fill(self, symbol: str, filled_qty: Decimal, side: str) -> Decimal:
        """Apply a fill to the position. Returns new position."""
        with self._lock:
            current = self._positions.get(symbol, Decimal("0"))
            if side == "buy":
                new_pos = current + filled_qty
            else:
                new_pos = current - filled_qty
            self._positions[symbol] = new_pos
            return new_pos

    def all_positions(self) -> Dict[str, Decimal]:
        with self._lock:
            return dict(self._positions)

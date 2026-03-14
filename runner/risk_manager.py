"""RiskManager — pre-trade risk checks: kill switch + position/notional limits.

Replaces the gate chain complexity of LiveRunner with a simple, testable class.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk: kill switch + position/notional limits + open order cap."""

    def __init__(
        self,
        kill_switch: Any,
        max_position: float = 1.0,
        max_notional: float = 10_000.0,
        max_open_orders: int = 5,
    ) -> None:
        self.kill_switch = kill_switch
        self.max_position = max_position
        self.max_notional = max_notional
        self.max_open_orders = max_open_orders

    def check(self, signal: Any, osm_open_count: int = 0) -> tuple[bool, str]:
        """Run all pre-trade risk checks.

        Returns (allowed, reason). reason is empty string if allowed.
        """
        # 1. Kill switch
        if self.kill_switch.is_killed():
            return False, "kill_switch_active"

        # 2. Position size
        qty = getattr(signal, "qty", 0)
        if abs(qty) > self.max_position:
            return False, f"position:{abs(qty):.4f}>{self.max_position}"

        # 3. Notional value
        notional = getattr(signal, "notional", 0)
        if abs(notional) > self.max_notional:
            return False, f"notional:{abs(notional):.0f}>{self.max_notional}"

        # 4. Open order count
        if osm_open_count >= self.max_open_orders:
            return False, f"open_orders:{osm_open_count}>={self.max_open_orders}"

        return True, ""

    def kill(self, reason: str) -> None:
        """Activate kill switch."""
        self.kill_switch.activate(reason=reason)
        logger.warning("Kill switch activated: %s", reason)

    def checkpoint(self) -> dict:
        """Serialize kill switch state for recovery."""
        state = {}
        if hasattr(self.kill_switch, "get_state"):
            state["kill_switch"] = self.kill_switch.get_state()
        return state

    def restore(self, state: dict) -> None:
        """Restore kill switch state from checkpoint."""
        if "kill_switch" in state and hasattr(self.kill_switch, "restore_state"):
            self.kill_switch.restore_state(state["kill_switch"])

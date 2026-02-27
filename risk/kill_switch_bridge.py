# risk/kill_switch_bridge.py
"""KillSwitchBridge — wraps an execution adapter with KillSwitch gating."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from risk.kill_switch import KillMode

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchBridge:
    """Wraps an execution adapter with KillSwitch gate.

    Checks kill_switch.allow_order() before every order submission.

    Behavior by mode:
        HARD_KILL:   Reject all orders; optionally cancel pending via cancel_fn.
        REDUCE_ONLY: Only allow orders with reduce_only=True through.

    The inner adapter must implement the ExecutionAdapter protocol
    (i.e. have a send_order method).
    """

    inner: Any  # ExecutionAdapter (has send_order)
    kill_switch: Any  # KillSwitch
    cancel_fn: Optional[Callable[[], None]] = None
    on_reject: Optional[Callable[[Any, Any], None]] = None

    _rejected_count: int = field(default=0, init=False, repr=False)

    def send_order(self, order_event: Any) -> Iterable[Any]:
        """Gate order through kill switch before delegating to inner adapter."""
        symbol = getattr(order_event, "symbol", "")
        strategy_id = getattr(order_event, "strategy_id", None)
        reduce_only = getattr(order_event, "reduce_only", False)

        allowed, record = self.kill_switch.allow_order(
            symbol=symbol,
            strategy_id=strategy_id,
            reduce_only=reduce_only,
        )

        if not allowed:
            self._rejected_count += 1
            logger.warning(
                "Order rejected by KillSwitch: symbol=%s strategy=%s mode=%s reason=%s",
                symbol,
                strategy_id,
                record.mode.value if record else "unknown",
                record.reason if record else "",
            )

            if record and record.mode == KillMode.HARD_KILL and self.cancel_fn is not None:
                try:
                    self.cancel_fn()
                except Exception:
                    logger.exception("cancel_fn failed during HARD_KILL")

            if self.on_reject is not None:
                try:
                    self.on_reject(order_event, record)
                except Exception:
                    logger.exception("on_reject callback failed")

            return []

        return self.inner.send_order(order_event)

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

"""Entry readiness: new exposure needs complete, fresh inputs from the bar that produced it."""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from trading.application.ports import OrderRequest
from trading.application.risk.entry_gate import GateResult


class BarInputs(Protocol):
    entry_ready: bool
    entry_reason: str


class EntryGate(Protocol):
    def check(self, event: OrderRequest) -> GateResult: ...


class EntryReadinessGate:
    """Blocks new exposure unless the bar that produced it had complete, fresh inputs."""

    def __init__(self, session: BarInputs, inner: EntryGate) -> None:
        self.session, self.inner = session, inner

    def check(self, event: OrderRequest) -> GateResult:
        try:
            qty = Decimal(str(event.qty))
            reducing = qty == 0 or getattr(event, "reduce_only", False) is True
            if not reducing and not self.session.entry_ready:
                return GateResult(False, "inputs: " + self.session.entry_reason)
            return self.inner.check(event)
        except Exception as exc:
            return GateResult(False, f"entry gate failed: {type(exc).__name__}")


__all__ = ["EntryReadinessGate"]

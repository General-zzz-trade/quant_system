"""Entry readiness: new exposure needs complete, fresh inputs from the bar that produced it."""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from trading.application.ports import OrderRequest
from trading.application.risk.entry_gate import GateResult
from trading.domain.symbol_state import SymbolState


class BarInputs(Protocol):
    entry_ready: bool
    entry_reason: str
    state: SymbolState


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
            result = self.inner.check(event)
            # ADR-0014 point 3, a backstop after the gate (so its own refusal reason is kept): only a READY symbol
            # enters. Every condition that makes a symbol not READY already refuses above; this catches any that
            # would not. A limiter reservation the gate made for such an entry is then counted but not used.
            if result.allowed and not reducing and self.session.state is not SymbolState.READY:
                return GateResult(False, f"state: {self.session.state.value}")
            return result
        except Exception as exc:
            return GateResult(False, f"entry gate failed: {type(exc).__name__}")


__all__ = ["EntryReadinessGate"]

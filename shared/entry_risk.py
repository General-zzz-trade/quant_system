"""Fail-closed execution boundary; reduction orders remain available."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str = ""


class CompositeRiskGate:
    def __init__(self, limiter: Any, kill_switch: Any = None, portfolio_reader: Any = None,
                 venue: str = "okx", drawdown_guard: Any = None, paused: Any = None) -> None:
        self._limiter = limiter
        self._kill_switch = kill_switch
        self._portfolio_reader = portfolio_reader
        self._venue = venue
        self._drawdown_guard = drawdown_guard
        self._paused = paused

    def check(self, event: Any) -> GateResult:
        try:
            qty = Decimal(str(event.qty))
            if not qty.is_finite() or qty < 0:
                return GateResult(False, "invalid order quantity")
            if str(event.side).lower() not in ("buy", "sell") or not event.symbol:
                return GateResult(False, "invalid order side or symbol")
            # qty=0 is the existing close-position convention. Adapters must
            # implement it as reduce-only, never as a new directional order.
            if qty == 0 or getattr(event, "reduce_only", False) is True:
                return GateResult(True)
            price = Decimal(str(event.price))
            if not price.is_finite() or price <= 0:
                return GateResult(False, "entry requires a finite positive reference price")
            if self._kill_switch is not None:
                allowed, reason = self._kill_switch.allow_order(symbol=str(event.symbol))
                if not allowed:
                    return GateResult(False, f"kill_switch: {reason}")
            if self._drawdown_guard is not None:
                allowed, reason = self._drawdown_guard.check()
                if not allowed:
                    return GateResult(False, f"drawdown: {reason}")
            if self._paused is not None and self._paused(str(event.symbol)):
                return GateResult(False, "runner paused")
            if self._portfolio_reader is None:
                return GateResult(False, "portfolio risk reader unavailable")
            allowed, reason = self._portfolio_reader.check_would_exceed(
                symbol=str(event.symbol), venue=self._venue, side=str(event.side).lower(),
                proposed_notional_usd=float(qty * price),
            )
            if not allowed:
                return GateResult(False, f"portfolio_gate: {reason}")
            result = self._limiter.check_order(qty=qty, price=price)
            return GateResult(result.allowed, result.detail or result.violated_rule or "")
        except Exception as exc:
            return GateResult(False, f"risk check failed: {type(exc).__name__}")

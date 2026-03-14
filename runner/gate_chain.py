# runner/gate_chain.py
"""GateChain — modular order gate pipeline extracted from LiveRunner._emit().

Each gate checks an ORDER event against a subsystem (correlation, risk, alpha
health, etc.) and either rejects it, scales its quantity, or passes it through.
The chain short-circuits on rejection: later gates are never called.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a single gate check."""
    allowed: bool
    scale: float = 1.0   # qty multiplier (1.0 = no change)
    reason: str = ""


class Gate(Protocol):
    """Protocol for order gates."""
    name: str

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        ...


class GateChain:
    """Processes an event through a sequence of gates.

    If any gate rejects (allowed=False), processing stops and None is returned.
    Scaling gates multiply into the event's qty field cumulatively.
    """

    def __init__(self, gates: Sequence[Gate]) -> None:
        self._gates = list(gates)

    def process(self, ev: Any, context: Dict[str, Any]) -> Optional[Any]:
        """Run event through all gates. Returns modified event or None if rejected."""
        for gate in self._gates:
            result = gate.check(ev, context)
            if not result.allowed:
                logger.warning(
                    "%s REJECTED order for %s: %s",
                    gate.name, getattr(ev, "symbol", "?"), result.reason,
                )
                return None
            if result.scale < 1.0:
                _apply_scale(ev, result.scale, gate.name)
        return ev

    def process_with_audit(
        self, ev: Any, context: Dict[str, Any]
    ) -> tuple[Optional[Any], List[tuple[str, GateResult]]]:
        """Run event through all gates, returning audit trail.

        Returns (modified_event_or_None, list_of_(gate_name, GateResult)).
        """
        trail: List[tuple[str, GateResult]] = []
        for gate in self._gates:
            result = gate.check(ev, context)
            trail.append((gate.name, result))
            if not result.allowed:
                logger.warning(
                    "%s REJECTED order for %s: %s",
                    gate.name, getattr(ev, "symbol", "?"), result.reason,
                )
                return None, trail
            if result.scale < 1.0:
                _apply_scale(ev, result.scale, gate.name)
        return ev, trail

    @property
    def gates(self) -> List[Gate]:
        return list(self._gates)


def _apply_scale(ev: Any, scale: float, gate_name: str) -> None:
    """Scale the qty field on an event."""
    raw_qty = getattr(ev, "qty", None) or getattr(ev, "quantity", None)
    if raw_qty is not None:
        scaled_qty = float(raw_qty) * scale
        object.__setattr__(ev, "qty", scaled_qty)
        logger.info(
            "%s scaled order for %s: %.4f → %.4f (scale=%.2f)",
            gate_name, getattr(ev, "symbol", "?"),
            float(raw_qty), scaled_qty, scale,
        )


# ============================================================
# Concrete gate implementations
# ============================================================

class CorrelationCheckGate:
    """Gate 1: Reject orders for correlated symbols."""
    name = "CorrelationGate"

    def __init__(self, correlation_gate: Any, get_state_view: Callable) -> None:
        self._gate = correlation_gate
        self._get_state_view = get_state_view

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        view = self._get_state_view()
        positions = view.get("positions", {})
        existing = [s for s, p in positions.items() if float(getattr(p, "qty", 0)) != 0]
        sym = getattr(ev, "symbol", "")
        decision = self._gate.should_allow(sym, existing)
        if not decision.ok:
            msg = decision.violations[0].message if decision.violations else "blocked"
            return GateResult(allowed=False, reason=msg)
        return GateResult(allowed=True)


class RiskSizeGate:
    """Gate 2: Risk size / notional check."""
    name = "RiskGate"

    def __init__(self, risk_gate: Any) -> None:
        self._gate = risk_gate

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        result = self._gate.check(ev)
        if not result.allowed:
            return GateResult(allowed=False, reason=result.reason)
        return GateResult(allowed=True)


class PortfolioRiskGate:
    """Gate 3: Portfolio-level risk check."""
    name = "PortfolioRisk"

    def __init__(self, portfolio_aggregator: Any) -> None:
        self._agg = portfolio_aggregator

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        try:
            decision = self._agg.evaluate_order(ev)
            if not decision.ok:
                msgs = [v.message for v in decision.violations]
                return GateResult(allowed=False, reason="; ".join(msgs))
        except Exception:
            sym = getattr(ev, "symbol", "?")
            logger.warning("PortfolioRisk check failed for %s", sym, exc_info=True)
        return GateResult(allowed=True)


class AlphaHealthGate:
    """Gate 4: Alpha health position scaling.

    NOTE: Intentional divergence from backtest. Live uses IC-based
    AlphaHealthMonitor for position scaling. Backtest uses feature-based
    RegimeGate (see backtest_module.py). Both produce 0.0/0.5/1.0 scale.
    """
    name = "AlphaHealth"

    def __init__(self, alpha_health_monitor: Any) -> None:
        self._ahm = alpha_health_monitor

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        sym = getattr(ev, "symbol", "")
        scale = self._ahm.position_scale(sym)
        if scale <= 0.0:
            return GateResult(allowed=False, reason="position_scale=0.0")
        return GateResult(allowed=True, scale=scale)


class RegimeSizerGate:
    """Gate 5: Regime-aware position scaling."""
    name = "RegimeSizer"

    def __init__(self, regime_sizer: Any) -> None:
        self._sizer = regime_sizer

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        sym = getattr(ev, "symbol", "")
        scale = self._sizer.position_scale(sym)
        return GateResult(allowed=True, scale=scale)


class PortfolioAllocatorGate:
    """Gate 6: Portfolio allocator order scaling."""
    name = "PortfolioAllocator"

    def __init__(self, portfolio_allocator: Any, get_state_view: Callable) -> None:
        self._alloc = portfolio_allocator
        self._get_state_view = get_state_view

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        raw_qty = getattr(ev, "qty", None) or getattr(ev, "quantity", None)
        raw_price = getattr(ev, "price", None)
        if raw_qty is None or raw_price is None:
            return GateResult(allowed=True)

        equity = 0.0
        try:
            acct = self._get_state_view().get("account")
            if acct is not None:
                equity = float(getattr(acct, "balance", 0))
        except Exception:
            pass

        if equity <= 0:
            return GateResult(allowed=True)

        sym = getattr(ev, "symbol", "")
        scaled_qty = self._alloc.scale_order(
            sym, float(raw_qty), equity, float(raw_price),
        )
        if abs(scaled_qty) < abs(float(raw_qty)):
            scale = abs(scaled_qty) / abs(float(raw_qty)) if float(raw_qty) != 0 else 1.0
            return GateResult(allowed=True, scale=scale)
        return GateResult(allowed=True)


class ExecQualityGate:
    """Gate 7: Execution quality feedback (slippage-based sizing)."""
    name = "ExecQuality"

    def __init__(self, hook: Any) -> None:
        self._hook = hook

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if self._hook is None or self._hook.execution_quality is None:
            return GateResult(allowed=True)
        sym = getattr(ev, "symbol", "")
        scale = self._hook.execution_quality.should_reduce_size(sym)
        if scale <= 0.0:
            return GateResult(allowed=False, reason="slippage too high")
        return GateResult(allowed=True, scale=scale)


class WeightRecGate:
    """Gate 8: Attribution weight recommendations."""
    name = "WeightRec"

    def __init__(self, hook: Any) -> None:
        self._hook = hook

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if self._hook is None or not self._hook.weight_recommendations:
            return GateResult(allowed=True)
        sym = getattr(ev, "symbol", "")
        wr = self._hook.weight_recommendations.get(sym, 1.0)
        if wr <= 0.0:
            return GateResult(allowed=False, reason="weight=0.0")
        return GateResult(allowed=True, scale=wr)


def build_gate_chain(
    *,
    correlation_gate: Any,
    risk_gate: Any,
    get_state_view: Callable,
    portfolio_aggregator: Optional[Any] = None,
    alpha_health_monitor: Optional[Any] = None,
    regime_sizer: Optional[Any] = None,
    portfolio_allocator: Optional[Any] = None,
    hook: Optional[Any] = None,
) -> GateChain:
    """Build the standard gate chain with all available subsystems."""
    gates: List[Gate] = [
        CorrelationCheckGate(correlation_gate, get_state_view),
        RiskSizeGate(risk_gate),
    ]
    if portfolio_aggregator is not None:
        gates.append(PortfolioRiskGate(portfolio_aggregator))
    if alpha_health_monitor is not None:
        gates.append(AlphaHealthGate(alpha_health_monitor))
    if regime_sizer is not None:
        gates.append(RegimeSizerGate(regime_sizer))
    if portfolio_allocator is not None:
        gates.append(PortfolioAllocatorGate(portfolio_allocator, get_state_view))
    if hook is not None:
        gates.append(ExecQualityGate(hook))
        gates.append(WeightRecGate(hook))
    return GateChain(gates)

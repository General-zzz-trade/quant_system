# runner/gate_chain.py
"""GateChain — modular order gate pipeline extracted from LiveRunner._emit().

Each gate checks an ORDER event against a subsystem (correlation, risk, alpha
health, etc.) and either rejects it, scales its quantity, or passes it through.
The chain short-circuits on rejection: later gates are never called.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
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
                ev = _apply_scale(ev, result.scale, gate.name)
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
                ev = _apply_scale(ev, result.scale, gate.name)
        return ev, trail

    @property
    def gates(self) -> List[Gate]:
        return list(self._gates)


def _apply_scale(ev: Any, scale: float, gate_name: str) -> Any:
    """Scale the qty field on an event. Returns new event if frozen, else mutates.

    For frozen dataclasses (OrderEvent), uses dataclasses.replace() to create
    a new instance instead of the old object.__setattr__ hack.
    For mutable objects, mutates in place.
    """
    raw_qty = getattr(ev, "qty", None) or getattr(ev, "quantity", None)
    if raw_qty is None:
        return ev
    if isinstance(raw_qty, Decimal):
        scaled_qty = raw_qty * Decimal(str(scale))
    else:
        scaled_qty = float(raw_qty) * scale

    logger.info(
        "%s scaled order for %s: %s → %s (scale=%.2f)",
        gate_name, getattr(ev, "symbol", "?"),
        raw_qty, scaled_qty, scale,
    )

    # Prefer dataclasses.replace() for frozen dataclasses (no mutation hack)
    import dataclasses
    if dataclasses.is_dataclass(ev) and getattr(type(ev), "__dataclass_params__", None):
        if getattr(type(ev).__dataclass_params__, "frozen", False):
            return dataclasses.replace(ev, qty=scaled_qty)
    # Mutable fallback
    try:
        ev.qty = scaled_qty
    except (AttributeError, TypeError):
        object.__setattr__(ev, "qty", scaled_qty)
    return ev


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

    def __init__(self, portfolio_aggregator: Any, kill_switch: Any = None) -> None:
        self._agg = portfolio_aggregator
        self._kill_switch = kill_switch

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        try:
            decision = self._agg.evaluate_order(ev)
            if not decision.ok:
                # If KILL action, trigger kill switch to prevent all future orders
                if self._kill_switch is not None:
                    from risk.decisions import RiskAction
                    if decision.action == RiskAction.KILL:
                        from risk.kill_switch import KillScope, KillMode
                        self._kill_switch.trigger(
                            scope=KillScope.GLOBAL,
                            key="*",
                            mode=KillMode.HARD_KILL,
                            reason=f"RiskAggregator KILL: {'; '.join(v.message for v in decision.violations)}",
                            source="risk_aggregator",
                        )
                        logger.critical(
                            "KillSwitch triggered by RiskAggregator KILL for %s",
                            getattr(ev, "symbol", "?"),
                        )
                msgs = [v.message for v in decision.violations]
                return GateResult(allowed=False, reason="; ".join(msgs))
        except Exception:
            sym = getattr(ev, "symbol", "?")
            logger.warning("PortfolioRisk check failed for %s", sym, exc_info=True)
            return GateResult(allowed=False, reason="risk_check_error")
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


class StagedRiskGate:
    """Gate: Staged risk management — blocks when halted, scales by drawdown."""
    name = "StagedRisk"

    def __init__(self, staged_risk: Any) -> None:
        self._staged = staged_risk

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._staged.can_trade:
            return GateResult(
                allowed=False,
                reason=f"staged_risk_halted: {self._staged.stage.label}",
            )
        scale = self._staged.position_scale()
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
        except Exception as e:
            logger.error("Failed to get account equity for portfolio gate: %s", e, exc_info=True)

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


class RustDrawdownGate:
    """Gate 7b: Rust-accelerated drawdown + kill-switch check.

    Uses RustRiskEvaluator.check_drawdown() for O(1) drawdown evaluation
    and RustKillSwitch.allow_order() for kill-switch enforcement.
    Arms the kill switch on drawdown breach to halt all future orders.
    """
    name = "RustDrawdown"

    def __init__(self, risk_evaluator: Any, kill_switch: Any,
                 get_equity: Any = None) -> None:
        self._eval = risk_evaluator
        self._ks = kill_switch
        self._get_equity = get_equity  # callable returning (equity, peak_equity)

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        sym = getattr(ev, "symbol", "?")

        # 1. Kill switch check first (fast path — already killed?)
        allowed, reason = self._ks.allow_order(symbol=sym)
        if not allowed:
            return GateResult(allowed=False, reason=f"kill_switch: {reason}")

        # 2. Drawdown check via Rust evaluator
        equity = context.get("equity", 0.0)
        peak_equity = context.get("peak_equity", 0.0)
        if self._get_equity is not None:
            try:
                equity, peak_equity = self._get_equity()
            except Exception as e:
                logger.error("Failed to get equity for drawdown gate: %s", e, exc_info=True)

        if peak_equity > 0 and equity > 0:
            breached = self._eval.check_drawdown(equity=equity, peak_equity=peak_equity)
            if breached:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                reason_str = f"drawdown {dd_pct:.1f}% breached"
                self._ks.arm("global", "*", "halt", reason_str,
                             source="RustDrawdownGate")
                logger.critical(
                    "RustDrawdownGate KILL: %s (equity=%.2f peak=%.2f)",
                    reason_str, equity, peak_equity,
                )
                return GateResult(allowed=False, reason=reason_str)

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
    staged_risk: Optional[Any] = None,
    portfolio_allocator: Optional[Any] = None,
    hook: Optional[Any] = None,
    kill_switch: Optional[Any] = None,
    rust_risk_evaluator: Optional[Any] = None,
    rust_kill_switch: Optional[Any] = None,
    get_equity: Optional[Callable] = None,
    # --- New sizing gates (P2-07) ---
    equity_leverage_gate: Optional[Any] = None,
    consensus_scaling_gate: Optional[Any] = None,
) -> GateChain:
    """Build the standard gate chain with all available subsystems."""
    gates: List[Gate] = [
        CorrelationCheckGate(correlation_gate, get_state_view),
        RiskSizeGate(risk_gate),
    ]
    if portfolio_aggregator is not None:
        gates.append(PortfolioRiskGate(portfolio_aggregator, kill_switch=kill_switch))
    if rust_risk_evaluator is not None and rust_kill_switch is not None:
        gates.append(RustDrawdownGate(rust_risk_evaluator, rust_kill_switch,
                                      get_equity=get_equity))
    if alpha_health_monitor is not None:
        gates.append(AlphaHealthGate(alpha_health_monitor))
    if regime_sizer is not None:
        gates.append(RegimeSizerGate(regime_sizer))
    if staged_risk is not None:
        gates.append(StagedRiskGate(staged_risk))
    # Sizing gates: after StagedRiskGate, before PortfolioAllocatorGate
    if equity_leverage_gate is not None:
        gates.append(equity_leverage_gate)
    if consensus_scaling_gate is not None:
        gates.append(consensus_scaling_gate)
    if portfolio_allocator is not None:
        gates.append(PortfolioAllocatorGate(portfolio_allocator, get_state_view))
    if hook is not None:
        gates.append(ExecQualityGate(hook))
        gates.append(WeightRecGate(hook))
    return GateChain(gates)

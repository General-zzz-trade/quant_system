"""Risk interceptors — bridge RiskAggregator and KillSwitch to PipelineInterceptor.

These interceptors make risk evaluation a mandatory, non-bypassable pipeline
stage.  Any event flowing through the InterceptorChain hits the risk gate
automatically — no code path can skip it.

Design:
    - ``KillSwitchInterceptor``: checks the kill-switch state *before* reduction.
      If a kill-switch is active for the event's symbol/strategy, the event is
      blocked (REJECT) or the pipeline is halted (KILL).
    - ``RiskInterceptor``: wraps ``RiskAggregator`` and evaluates Intent/Order
      events through the full rule-set.  Other event kinds pass through.
"""
from __future__ import annotations

from typing import Any, Optional

from core.interceptors import InterceptAction, InterceptResult, PipelineInterceptor
from core.types import Envelope
from risk.decisions import RiskAction, RiskDecision


# ── KillSwitch Interceptor ───────────────────────────────

class KillSwitchInterceptor:
    """Pipeline interceptor backed by a KillSwitch.

    Checks the kill-switch state before every reduction.  If the symbol
    or strategy of the incoming event is killed, the event is rejected
    (or the pipeline is halted for HARD_KILL on global scope).

    Parameters
    ----------
    kill_switch : KillSwitch
        The kill-switch instance to query.
    hard_kill_halts_pipeline : bool
        If True (default), a global HARD_KILL triggers InterceptAction.KILL
        instead of REJECT, halting the entire pipeline.
    """

    def __init__(
        self,
        kill_switch: Any,
        *,
        hard_kill_halts_pipeline: bool = True,
    ) -> None:
        self._ks = kill_switch
        self._halt_on_hard = hard_kill_halts_pipeline

    @property
    def name(self) -> str:
        return "kill_switch"

    def before_reduce(self, envelope: Envelope, state: Any) -> InterceptResult:
        symbol = _extract_symbol(envelope)
        strategy_id = _extract_strategy_id(envelope)

        rec = self._ks.is_killed(symbol=symbol, strategy_id=strategy_id)
        if rec is None:
            return InterceptResult.ok(self.name)

        from risk.kill_switch import KillMode, KillScope

        reason = f"kill-switch active: scope={rec.scope.value} key={rec.key} mode={rec.mode.value}"
        if rec.reason:
            reason += f" reason={rec.reason}"

        # Global HARD_KILL → halt the pipeline entirely
        if (
            self._halt_on_hard
            and rec.scope == KillScope.GLOBAL
            and rec.mode == KillMode.HARD_KILL
        ):
            return InterceptResult.kill(self.name, reason)

        return InterceptResult.reject(self.name, reason)

    def after_reduce(
        self, envelope: Envelope, old_state: Any, new_state: Any
    ) -> InterceptResult:
        return InterceptResult.ok(self.name)


# ── Risk Aggregator Interceptor ──────────────────────────

class RiskInterceptor:
    """Pipeline interceptor backed by a RiskAggregator.

    Only evaluates events that the aggregator understands (Intent / Order).
    All other event kinds pass through with CONTINUE.

    The ``RiskDecision`` is attached to the ``InterceptResult.adjustment``
    field so downstream code can inspect violations and adjustments.

    Parameters
    ----------
    aggregator : RiskAggregator
        The aggregator instance with loaded rules.
    reject_on_reduce : bool
        If True, ``RiskAction.REDUCE`` is mapped to REJECT (conservative).
        If False (default), REDUCE maps to CONTINUE and the decision is
        attached for the executor to apply adjustments.
    """

    def __init__(
        self,
        aggregator: Any,
        *,
        reject_on_reduce: bool = False,
    ) -> None:
        self._agg = aggregator
        self._reject_on_reduce = reject_on_reduce

    @property
    def name(self) -> str:
        return "risk_aggregator"

    def before_reduce(self, envelope: Envelope, state: Any) -> InterceptResult:
        event = envelope.event

        # Only evaluate events the aggregator can handle
        decision: Optional[RiskDecision] = None

        from event.types import IntentEvent, OrderEvent

        if isinstance(event, IntentEvent):
            decision = self._agg.evaluate_intent(event)
        elif isinstance(event, OrderEvent):
            decision = self._agg.evaluate_order(event)
        else:
            return InterceptResult.ok(self.name)

        return _decision_to_result(
            decision,
            interceptor_name=self.name,
            reject_on_reduce=self._reject_on_reduce,
        )

    def after_reduce(
        self, envelope: Envelope, old_state: Any, new_state: Any
    ) -> InterceptResult:
        return InterceptResult.ok(self.name)


# ── Helpers ──────────────────────────────────────────────

def _extract_symbol(envelope: Envelope) -> Optional[str]:
    """Best-effort symbol extraction from an envelope's event."""
    event = envelope.event
    sym = getattr(event, "symbol", None)
    if isinstance(sym, str) and sym:
        return sym
    # Try canonical form if it's a Symbol object
    canonical = getattr(sym, "canonical", None)
    if isinstance(canonical, str):
        return canonical
    return None


def _extract_strategy_id(envelope: Envelope) -> Optional[str]:
    """Best-effort strategy_id extraction."""
    event = envelope.event
    for attr in ("strategy_id", "origin"):
        val = getattr(event, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _decision_to_result(
    decision: RiskDecision,
    *,
    interceptor_name: str,
    reject_on_reduce: bool,
) -> InterceptResult:
    """Map a RiskDecision to an InterceptResult."""
    if decision.action == RiskAction.ALLOW:
        return InterceptResult.ok(interceptor_name)

    # Build a human-readable reason from violations
    reasons = [v.message for v in decision.violations[:3]]
    reason = "; ".join(reasons) if reasons else f"risk {decision.action.value}"

    if decision.action == RiskAction.KILL:
        return InterceptResult(
            action=InterceptAction.KILL,
            interceptor=interceptor_name,
            reason=reason,
            adjustment=decision,
        )

    if decision.action == RiskAction.REJECT:
        return InterceptResult(
            action=InterceptAction.REJECT,
            interceptor=interceptor_name,
            reason=reason,
            adjustment=decision,
        )

    if decision.action == RiskAction.REDUCE:
        if reject_on_reduce:
            return InterceptResult(
                action=InterceptAction.REJECT,
                interceptor=interceptor_name,
                reason=f"reduce→reject: {reason}",
                adjustment=decision,
            )
        # REDUCE maps to CONTINUE with the decision attached
        return InterceptResult(
            action=InterceptAction.CONTINUE,
            interceptor=interceptor_name,
            reason=reason,
            adjustment=decision,
        )

    # Unknown action — conservative reject
    return InterceptResult.reject(interceptor_name, f"unknown risk action: {decision.action}")

"""Tests for risk.interceptor — KillSwitchInterceptor and RiskInterceptor."""
from __future__ import annotations

from typing import Any, Mapping

from core.interceptors import InterceptAction, InterceptResult, InterceptorChain
from core.types import Envelope, EventKind, EventMetadata
from risk.decisions import (
    RiskAction,
    RiskAdjustment,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)
from risk.interceptor import KillSwitchInterceptor, RiskInterceptor
from risk.kill_switch import KillMode, KillScope as KillSwitchScope, KillSwitch


# ── Helpers ──────────────────────────────────────────────

def _make_envelope(event: Any, kind: EventKind = EventKind.CONTROL) -> Envelope:
    meta = EventMetadata.create(source="test")
    return Envelope(event=event, metadata=meta, kind=kind)


class _StubEvent:
    """Minimal event with symbol and strategy_id."""
    def __init__(self, symbol: str = "BTCUSDT", strategy_id: str = "alpha_1"):
        self.symbol = symbol
        self.strategy_id = strategy_id


# ── KillSwitchInterceptor Tests ──────────────────────────


class TestKillSwitchInterceptor:
    def test_no_kill_continues(self) -> None:
        ks = KillSwitch()
        ic = KillSwitchInterceptor(ks)
        env = _make_envelope(_StubEvent())
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.CONTINUE

    def test_symbol_kill_rejects(self) -> None:
        ks = KillSwitch()
        ks.trigger(scope=KillSwitchScope.SYMBOL, key="BTCUSDT", reason="test")
        ic = KillSwitchInterceptor(ks)
        env = _make_envelope(_StubEvent(symbol="BTCUSDT"))
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.REJECT
        assert "BTCUSDT" in result.reason

    def test_global_hard_kill_halts_pipeline(self) -> None:
        ks = KillSwitch()
        ks.trigger(scope=KillSwitchScope.GLOBAL, key="*", mode=KillMode.HARD_KILL, reason="meltdown")
        ic = KillSwitchInterceptor(ks, hard_kill_halts_pipeline=True)
        env = _make_envelope(_StubEvent())
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.KILL
        assert "meltdown" in result.reason

    def test_global_hard_kill_without_halt_rejects(self) -> None:
        ks = KillSwitch()
        ks.trigger(scope=KillSwitchScope.GLOBAL, key="*", mode=KillMode.HARD_KILL)
        ic = KillSwitchInterceptor(ks, hard_kill_halts_pipeline=False)
        env = _make_envelope(_StubEvent())
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.REJECT

    def test_strategy_kill_rejects(self) -> None:
        ks = KillSwitch()
        ks.trigger(scope=KillSwitchScope.STRATEGY, key="alpha_1", reason="broken")
        ic = KillSwitchInterceptor(ks)
        env = _make_envelope(_StubEvent(strategy_id="alpha_1"))
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.REJECT

    def test_after_reduce_always_ok(self) -> None:
        ks = KillSwitch()
        ks.trigger(scope=KillSwitchScope.GLOBAL, key="*")
        ic = KillSwitchInterceptor(ks)
        env = _make_envelope(_StubEvent())
        result = ic.after_reduce(env, old_state=None, new_state=None)
        assert result.action == InterceptAction.CONTINUE

    def test_name(self) -> None:
        ic = KillSwitchInterceptor(KillSwitch())
        assert ic.name == "kill_switch"


# ── RiskInterceptor Tests ────────────────────────────────


class _MockAggregator:
    """Mock RiskAggregator that returns a fixed decision."""
    def __init__(self, intent_decision: RiskDecision, order_decision: RiskDecision):
        self._intent_d = intent_decision
        self._order_d = order_decision

    def evaluate_intent(self, intent: Any) -> RiskDecision:
        return self._intent_d

    def evaluate_order(self, order: Any) -> RiskDecision:
        return self._order_d


class TestRiskInterceptor:
    def test_non_intent_order_passes_through(self) -> None:
        agg = _MockAggregator(RiskDecision.allow(), RiskDecision.allow())
        ic = RiskInterceptor(agg)
        env = _make_envelope({"data": "market"}, kind=EventKind.MARKET)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.CONTINUE

    def test_intent_allow_continues(self) -> None:
        from event.types import IntentEvent
        from decimal import Decimal

        agg = _MockAggregator(RiskDecision.allow(), RiskDecision.allow())
        ic = RiskInterceptor(agg)

        intent = IntentEvent(
            header=None,
            intent_id="i1",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("0.1"),
            reason_code="signal",
            origin="alpha_1",
        )
        env = _make_envelope(intent, kind=EventKind.SIGNAL)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.CONTINUE

    def test_intent_reject_rejects(self) -> None:
        from event.types import IntentEvent
        from decimal import Decimal

        v = RiskViolation(code=RiskCode.MAX_POSITION, message="too big")
        d = RiskDecision.reject((v,), scope=RiskScope.SYMBOL)
        agg = _MockAggregator(d, RiskDecision.allow())
        ic = RiskInterceptor(agg)

        intent = IntentEvent(
            header=None,
            intent_id="i2",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("10"),
            reason_code="signal",
            origin="alpha_1",
        )
        env = _make_envelope(intent, kind=EventKind.SIGNAL)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.REJECT
        assert "too big" in result.reason
        # Decision attached for downstream inspection
        assert isinstance(result.adjustment, RiskDecision)

    def test_intent_kill_kills(self) -> None:
        from event.types import IntentEvent
        from decimal import Decimal

        v = RiskViolation(code=RiskCode.LIQUIDATION_RISK, message="imminent liq")
        d = RiskDecision.kill((v,), scope=RiskScope.GLOBAL)
        agg = _MockAggregator(d, RiskDecision.allow())
        ic = RiskInterceptor(agg)

        intent = IntentEvent(
            header=None,
            intent_id="i3",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("5"),
            reason_code="signal",
            origin="alpha_1",
        )
        env = _make_envelope(intent, kind=EventKind.SIGNAL)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.KILL

    def test_reduce_default_continues_with_adjustment(self) -> None:
        from event.types import IntentEvent
        from decimal import Decimal

        v = RiskViolation(code=RiskCode.MAX_POSITION, message="scale down")
        adj = RiskAdjustment(max_qty=0.05)
        d = RiskDecision.reduce((v,), adjustment=adj, scope=RiskScope.SYMBOL)
        agg = _MockAggregator(d, RiskDecision.allow())
        ic = RiskInterceptor(agg, reject_on_reduce=False)

        intent = IntentEvent(
            header=None,
            intent_id="i4",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("1"),
            reason_code="signal",
            origin="alpha_1",
        )
        env = _make_envelope(intent, kind=EventKind.SIGNAL)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.CONTINUE
        assert result.adjustment is not None
        assert result.adjustment.adjustment.max_qty == 0.05

    def test_reduce_reject_on_reduce_rejects(self) -> None:
        from event.types import IntentEvent
        from decimal import Decimal

        v = RiskViolation(code=RiskCode.MAX_POSITION, message="scale down")
        adj = RiskAdjustment(max_qty=0.05)
        d = RiskDecision.reduce((v,), adjustment=adj, scope=RiskScope.SYMBOL)
        agg = _MockAggregator(d, RiskDecision.allow())
        ic = RiskInterceptor(agg, reject_on_reduce=True)

        intent = IntentEvent(
            header=None,
            intent_id="i5",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("1"),
            reason_code="signal",
            origin="alpha_1",
        )
        env = _make_envelope(intent, kind=EventKind.SIGNAL)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.REJECT

    def test_order_evaluation(self) -> None:
        from event.types import OrderEvent
        from decimal import Decimal

        v = RiskViolation(code=RiskCode.MAX_NOTIONAL, message="notional breach")
        d = RiskDecision.reject((v,), scope=RiskScope.SYMBOL)
        agg = _MockAggregator(RiskDecision.allow(), d)
        ic = RiskInterceptor(agg)

        order = OrderEvent(
            header=None,
            order_id="o1",
            intent_id="i1",
            symbol="BTCUSDT",
            side="buy",
            qty=Decimal("5"),
            price=Decimal("50000"),
        )
        env = _make_envelope(order, kind=EventKind.CONTROL)
        result = ic.before_reduce(env, state=None)
        assert result.action == InterceptAction.REJECT

    def test_chain_integration(self) -> None:
        """RiskInterceptor works correctly within an InterceptorChain."""
        from event.types import IntentEvent
        from decimal import Decimal

        v = RiskViolation(code=RiskCode.MAX_POSITION, message="blocked")
        d = RiskDecision.reject((v,), scope=RiskScope.SYMBOL)
        agg = _MockAggregator(d, RiskDecision.allow())

        chain = InterceptorChain([RiskInterceptor(agg)])

        intent = IntentEvent(
            header=None,
            intent_id="i6",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("1"),
            reason_code="signal",
            origin="alpha_1",
        )
        env = _make_envelope(intent, kind=EventKind.SIGNAL)
        result = chain.run_before(env, state=None)
        assert result.action == InterceptAction.REJECT

    def test_name(self) -> None:
        agg = _MockAggregator(RiskDecision.allow(), RiskDecision.allow())
        ic = RiskInterceptor(agg)
        assert ic.name == "risk_aggregator"

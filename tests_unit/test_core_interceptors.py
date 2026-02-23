"""Tests for core.interceptors — InterceptorChain and built-in interceptors."""
from __future__ import annotations

from core.interceptors import (
    EventKindGate,
    InterceptAction,
    InterceptorChain,
    InterceptResult,
    PassthroughInterceptor,
)
from core.types import Envelope, EventKind, EventMetadata, Priority


def _env(kind: EventKind = EventKind.MARKET) -> Envelope:
    meta = EventMetadata.create(source="test")
    return Envelope(event={}, metadata=meta, kind=kind)


class TestInterceptResult:
    def test_ok(self) -> None:
        r = InterceptResult.ok("test")
        assert r.action == InterceptAction.CONTINUE
        assert r.interceptor == "test"

    def test_reject(self) -> None:
        r = InterceptResult.reject("risk", "leverage exceeded")
        assert r.action == InterceptAction.REJECT
        assert r.reason == "leverage exceeded"

    def test_kill(self) -> None:
        r = InterceptResult.kill("risk", "max drawdown")
        assert r.action == InterceptAction.KILL


class TestPassthroughInterceptor:
    def test_always_continues(self) -> None:
        ic = PassthroughInterceptor()
        assert ic.name == "passthrough"
        r1 = ic.before_reduce(_env(), state=None)
        assert r1.action == InterceptAction.CONTINUE
        r2 = ic.after_reduce(_env(), old_state=None, new_state=None)
        assert r2.action == InterceptAction.CONTINUE


class TestEventKindGate:
    def test_blocks_specified_kinds(self) -> None:
        gate = EventKindGate(
            blocked_kinds=frozenset({EventKind.ORDER}),
            reason="kill switch active",
        )
        # ORDER should be rejected
        r = gate.before_reduce(_env(EventKind.ORDER), state=None)
        assert r.action == InterceptAction.REJECT
        assert "kill switch" in r.reason

    def test_allows_non_blocked_kinds(self) -> None:
        gate = EventKindGate(blocked_kinds=frozenset({EventKind.ORDER}))
        r = gate.before_reduce(_env(EventKind.MARKET), state=None)
        assert r.action == InterceptAction.CONTINUE

    def test_after_reduce_always_ok(self) -> None:
        gate = EventKindGate(blocked_kinds=frozenset({EventKind.ORDER}))
        r = gate.after_reduce(_env(EventKind.ORDER), old_state=None, new_state=None)
        assert r.action == InterceptAction.CONTINUE


class TestInterceptorChain:
    def test_empty_chain_continues(self) -> None:
        chain = InterceptorChain()
        r = chain.run_before(_env(), state=None)
        assert r.action == InterceptAction.CONTINUE

    def test_single_passthrough(self) -> None:
        chain = InterceptorChain([PassthroughInterceptor()])
        r = chain.run_before(_env(), state=None)
        assert r.action == InterceptAction.CONTINUE

    def test_fail_fast_on_reject(self) -> None:
        """First interceptor rejects → second never runs."""
        gate = EventKindGate(
            blocked_kinds=frozenset({EventKind.ORDER}),
            reason="blocked",
        )
        passthrough = PassthroughInterceptor()
        chain = InterceptorChain([gate, passthrough])

        r = chain.run_before(_env(EventKind.ORDER), state=None)
        assert r.action == InterceptAction.REJECT
        assert r.interceptor == "event_kind_gate"

    def test_all_continue_returns_ok(self) -> None:
        chain = InterceptorChain([
            PassthroughInterceptor(),
            PassthroughInterceptor(),
        ])
        r = chain.run_before(_env(), state=None)
        assert r.action == InterceptAction.CONTINUE

    def test_after_reduce_fail_fast_on_kill(self) -> None:
        class KillAfterReduce:
            @property
            def name(self) -> str:
                return "killer"

            def before_reduce(self, envelope, state):
                return InterceptResult.ok(self.name)

            def after_reduce(self, envelope, old_state, new_state):
                return InterceptResult.kill(self.name, "drawdown")

        chain = InterceptorChain([KillAfterReduce(), PassthroughInterceptor()])
        r = chain.run_after(_env(), old_state=None, new_state=None)
        assert r.action == InterceptAction.KILL
        assert r.interceptor == "killer"

    def test_mixed_interceptors_order_matters(self) -> None:
        """Passthrough first, then gate — ORDER still gets rejected."""
        chain = InterceptorChain([
            PassthroughInterceptor(),
            EventKindGate(blocked_kinds=frozenset({EventKind.ORDER})),
        ])
        r = chain.run_before(_env(EventKind.ORDER), state=None)
        assert r.action == InterceptAction.REJECT

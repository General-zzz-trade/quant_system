# tests/unit/risk/test_kill_switch_bridge.py
"""Tests for KillSwitchBridge — gating orders through the kill switch."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

from risk.kill_switch import KillMode, KillRecord, KillScope, KillSwitch
from risk.kill_switch_bridge import KillSwitchBridge


# ── Stubs ────────────────────────────────────────────────────


def _order_event(
    *,
    symbol: str = "BTCUSDT",
    strategy_id: str | None = "strat_a",
    reduce_only: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="order",
        order_id="ord-001",
        symbol=symbol,
        strategy_id=strategy_id,
        reduce_only=reduce_only,
        side="buy",
        qty="1.0",
        price="40000",
    )


class _FakeInner:
    """Fake execution adapter that records calls."""

    def __init__(self, fills: list | None = None):
        self.fills = fills or [SimpleNamespace(event_type="fill")]
        self.received: List[Any] = []

    def send_order(self, order_event: Any) -> list:
        self.received.append(order_event)
        return self.fills


# ── Tests: Order pass-through ────────────────────────────────


class TestAllowed:
    def test_allowed_order_passes_through(self):
        ks = KillSwitch()
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        order = _order_event()
        results = list(bridge.send_order(order))

        assert len(results) == 1
        assert results[0].event_type == "fill"
        assert len(inner.received) == 1

    def test_no_strategy_id_on_order(self):
        ks = KillSwitch()
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        order = _order_event(strategy_id=None)
        results = list(bridge.send_order(order))

        assert len(results) == 1


# ── Tests: HARD_KILL ─────────────────────────────────────────


class TestHardKill:
    def test_hard_kill_rejects_order(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL)
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        results = list(bridge.send_order(_order_event(symbol="BTCUSDT")))

        assert len(results) == 0
        assert len(inner.received) == 0

    def test_hard_kill_rejects_reduce_only_too(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL)
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        results = list(bridge.send_order(_order_event(symbol="BTCUSDT", reduce_only=True)))

        assert len(results) == 0

    def test_hard_kill_calls_cancel_fn(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL)
        inner = _FakeInner()
        cancel_calls: List[bool] = []
        bridge = KillSwitchBridge(
            inner=inner, kill_switch=ks, cancel_fn=lambda: cancel_calls.append(True),
        )

        list(bridge.send_order(_order_event(symbol="BTCUSDT")))

        assert len(cancel_calls) == 1

    def test_hard_kill_other_symbol_allowed(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL)
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        results = list(bridge.send_order(_order_event(symbol="ETHUSDT")))

        assert len(results) == 1


# ── Tests: REDUCE_ONLY ───────────────────────────────────────


class TestReduceOnly:
    def test_reduce_only_allows_reduce_order(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY)
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        results = list(bridge.send_order(_order_event(symbol="BTCUSDT", reduce_only=True)))

        assert len(results) == 1
        assert len(inner.received) == 1

    def test_reduce_only_rejects_opening_order(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY)
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        results = list(bridge.send_order(_order_event(symbol="BTCUSDT", reduce_only=False)))

        assert len(results) == 0
        assert len(inner.received) == 0

    def test_reduce_only_does_not_call_cancel_fn(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.REDUCE_ONLY)
        inner = _FakeInner()
        cancel_calls: List[bool] = []
        bridge = KillSwitchBridge(
            inner=inner, kill_switch=ks, cancel_fn=lambda: cancel_calls.append(True),
        )

        list(bridge.send_order(_order_event(symbol="BTCUSDT", reduce_only=False)))

        assert len(cancel_calls) == 0  # cancel_fn only called for HARD_KILL


# ── Tests: on_reject callback ────────────────────────────────


class TestOnReject:
    def test_on_reject_called_with_order_and_record(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.SYMBOL, key="BTCUSDT", mode=KillMode.HARD_KILL, reason="test")
        inner = _FakeInner()
        rejected: List[Tuple[Any, Any]] = []
        bridge = KillSwitchBridge(
            inner=inner, kill_switch=ks, on_reject=lambda o, r: rejected.append((o, r)),
        )

        order = _order_event(symbol="BTCUSDT")
        list(bridge.send_order(order))

        assert len(rejected) == 1
        assert rejected[0][0] is order
        assert rejected[0][1].mode == KillMode.HARD_KILL

    def test_rejected_count_increments(self):
        ks = KillSwitch()
        ks.trigger(scope=KillScope.GLOBAL, key="*", mode=KillMode.HARD_KILL)
        inner = _FakeInner()
        bridge = KillSwitchBridge(inner=inner, kill_switch=ks)

        assert bridge.rejected_count == 0
        list(bridge.send_order(_order_event()))
        assert bridge.rejected_count == 1
        list(bridge.send_order(_order_event()))
        assert bridge.rejected_count == 2

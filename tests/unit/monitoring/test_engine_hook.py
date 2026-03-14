# tests/unit/monitoring/test_engine_hook.py
"""Tests for EngineMonitoringHook."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from monitoring.engine_hook import EngineMonitoringHook


# ── Stubs ────────────────────────────────────────────────────

def _account(balance: str = "10000", equity: str | None = None) -> SimpleNamespace:
    ns = SimpleNamespace(balance=Decimal(balance))
    if equity is not None:
        ns.equity = Decimal(equity)
    return ns


def _market(close: float) -> SimpleNamespace:
    return SimpleNamespace(close=close, last_price=close)


def _position(qty: float) -> SimpleNamespace:
    return SimpleNamespace(qty=qty, quantity=qty)


def _pipeline_output(
    *,
    markets: Dict[str, Any] | None = None,
    positions: Dict[str, Any] | None = None,
    account: Any = None,
    event_index: int = 1,
    advanced: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        markets=markets or {},
        positions=positions or {},
        account=account or _account(),
        event_index=event_index,
        advanced=advanced,
        snapshot=None,
    )


class _MockHealthMonitor:
    def __init__(self):
        self.market_data_calls = 0
        self.balance_updates: List[Dict[str, Any]] = []

    def on_market_data(self, ts=None):
        self.market_data_calls += 1

    def on_balance_update(self, *, balance=None, equity=None):
        self.balance_updates.append({"balance": balance, "equity": equity})


class _MockMetrics:
    def __init__(self):
        self.gauges: Dict[str, Any] = {}
        self.counters: Dict[str, float] = {}

    def set_gauge(self, name, value, *, labels=None):
        key = name if labels is None else f"{name}:{labels}"
        self.gauges[key] = value

    def inc_counter(self, name, value=1.0, *, labels=None):
        key = name if labels is None else f"{name}:{labels}"
        self.counters[key] = self.counters.get(key, 0) + value


# ── Tests ────────────────────────────────────────────────────

class TestHealthMonitorIntegration:
    def test_on_market_data_called(self):
        health = _MockHealthMonitor()
        hook = EngineMonitoringHook(health=health)
        out = _pipeline_output()
        hook(out)
        assert health.market_data_calls == 1

    def test_balance_update_forwarded(self):
        health = _MockHealthMonitor()
        hook = EngineMonitoringHook(health=health)
        out = _pipeline_output(account=_account("9500"))
        hook(out)
        assert len(health.balance_updates) == 1
        assert health.balance_updates[0]["balance"] == Decimal("9500")

    def test_equity_forwarded(self):
        health = _MockHealthMonitor()
        hook = EngineMonitoringHook(health=health)
        acc = SimpleNamespace(balance=Decimal("9500"), equity=Decimal("9800"))
        out = _pipeline_output(account=acc)
        hook(out)
        assert health.balance_updates[0]["equity"] == Decimal("9800")

    def test_no_health_monitor_ok(self):
        hook = EngineMonitoringHook(health=None)
        out = _pipeline_output()
        hook(out)  # should not raise


class TestMetricsIntegration:
    def test_balance_gauge_set(self):
        metrics = _MockMetrics()
        hook = EngineMonitoringHook(metrics=metrics)
        out = _pipeline_output(account=_account("10500"))
        hook(out)
        assert metrics.gauges["balance_usdt"] == pytest.approx(10500.0)

    def test_equity_gauge_set(self):
        metrics = _MockMetrics()
        hook = EngineMonitoringHook(metrics=metrics)
        acc = SimpleNamespace(balance=Decimal("10000"), equity=Decimal("10500"))
        out = _pipeline_output(account=acc)
        hook(out)
        assert metrics.gauges["equity_usdt"] == pytest.approx(10500.0)

    def test_event_count_incremented(self):
        metrics = _MockMetrics()
        hook = EngineMonitoringHook(metrics=metrics)
        hook(_pipeline_output())
        hook(_pipeline_output(event_index=2))
        assert metrics.counters["pipeline_events_total"] == pytest.approx(2.0)

    def test_event_index_tracked(self):
        metrics = _MockMetrics()
        hook = EngineMonitoringHook(metrics=metrics)
        hook(_pipeline_output(event_index=42))
        assert metrics.gauges["event_index"] == pytest.approx(42.0)

    def test_per_symbol_price(self):
        metrics = _MockMetrics()
        hook = EngineMonitoringHook(metrics=metrics)
        out = _pipeline_output(
            markets={"BTCUSDT": _market(40000.0), "ETHUSDT": _market(3000.0)},
        )
        hook(out)
        assert metrics.gauges["price:{'symbol': 'BTCUSDT'}"] == pytest.approx(40000.0)
        assert metrics.gauges["price:{'symbol': 'ETHUSDT'}"] == pytest.approx(3000.0)

    def test_per_symbol_position(self):
        metrics = _MockMetrics()
        hook = EngineMonitoringHook(metrics=metrics)
        out = _pipeline_output(
            positions={"BTCUSDT": _position(1.5)},
        )
        hook(out)
        assert metrics.gauges["position_qty:{'symbol': 'BTCUSDT'}"] == pytest.approx(1.5)

    def test_no_metrics_ok(self):
        hook = EngineMonitoringHook(metrics=None)
        out = _pipeline_output()
        hook(out)  # should not raise


class TestEventCounting:
    def test_internal_counter(self):
        hook = EngineMonitoringHook()
        hook(_pipeline_output())
        hook(_pipeline_output())
        hook(_pipeline_output())
        assert hook._event_count == 3

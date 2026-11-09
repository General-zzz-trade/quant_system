"""Actual builder wiring with causal OI/LS fixtures and no exchange networking."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from types import SimpleNamespace

import pytest

from event.header import EventHeader
from event.types import EventType, MarketEvent
from runner.builders import alpha_builder as builder
from runner.execution_recovery import reconcile_execution
from shared.entry_risk import GateResult


HOUR = 3_600_000
NOW = 1_800_000_000_000


class FakeCache:
    def __init__(self):
        self.status = "ready"
        self.value = 999.0
        self.received = NOW
        self.offset = 0
        self.calls = []

    def get(self, *, as_of_ms=None):
        self.calls.append(as_of_ms)
        return {"status": self.status, "reason": "fixture", "open_interest": self.value,
                "ls_ratio": 2.0, "oi_event_ts_ms": as_of_ms + self.offset,
                "ls_event_ts_ms": as_of_ms, "received_at_ms": self.received}


def csv_inputs(path, timestamps):
    path.mkdir(exist_ok=True)
    (path / "BTCUSDT_open_interest.csv").write_text(
        "timestamp,sum_open_interest\n" + "".join(f"{ts},{100 + i}\n" for i, ts in enumerate(timestamps)))
    (path / "BTCUSDT_ls_ratio.csv").write_text(
        "timestamp,long_short_ratio\n" + "".join(f"{ts},{1 + i / 10}\n" for i, ts in enumerate(timestamps)))


@pytest.fixture
def sources(tmp_path, monkeypatch):
    csv_inputs(tmp_path, [NOW - 3 * HOUR, NOW - 2 * HOUR, NOW - HOUR])
    monkeypatch.setattr(builder, "DATA_DIR", tmp_path)
    cache = FakeCache()
    result = builder._build_data_sources("BTCUSDT", oi_cache=cache, required_warmup_bars=2)
    result.readiness.clock = lambda: NOW / 1000
    return result, cache


def finish_warmup(sources):
    for timestamp in (NOW - 3 * HOUR, NOW - 2 * HOUR):
        sources["_set_bar_ts"](timestamp)
        assert math.isfinite(sources["oi_source"]())
        assert math.isfinite(sources["ls_ratio_source"]())
    sources.readiness.set_live()
    sources["_set_bar_ts"](NOW - HOUR)


def test_historical_inputs_never_use_future_or_indefinitely_old_csv(sources):
    source, cache = sources
    source["_set_bar_ts"](NOW - 4 * HOUR)
    assert math.isnan(source["oi_source"]())
    source["_set_bar_ts"](NOW - 3 * HOUR)
    assert source["oi_source"]() == 100
    source["_set_bar_ts"](NOW + 2 * HOUR)
    assert math.isnan(source["oi_source"]())
    assert math.isnan(source["ls_ratio_source"]())
    assert source.readiness.valid_warmup_bars == 0
    assert cache.calls == []


def test_warmup_counts_unique_valid_bars_and_set_live_cannot_invent_history(sources):
    source, _ = sources
    source["_set_bar_ts"](NOW - 2 * HOUR)
    for _ in range(4):
        source["oi_source"]()
        source["ls_ratio_source"]()
        source["_set_bar_ts"](NOW - 2 * HOUR)
    assert source.readiness.valid_warmup_bars == 1
    source.readiness.set_live()
    source["_set_bar_ts"](NOW - HOUR)
    assert not source.readiness.snapshot()["ready"]
    assert "insufficient" in source.readiness.snapshot()["reason"]


def test_warmup_gap_restarts_consecutive_valid_bar_count(sources):
    source, _ = sources
    source["_set_bar_ts"](NOW - 3 * HOUR)
    source["_set_bar_ts"](NOW - HOUR)
    assert source.readiness.valid_warmup_bars == 1
    source.readiness.set_live()
    assert not source.readiness.snapshot()["ready"]


def test_closed_hourly_live_bar_uses_cache_instead_of_warmup_heuristic(sources):
    source, cache = sources
    finish_warmup(source)
    assert source.readiness.snapshot()["ready"]
    assert source["oi_source"]() == 999
    assert source["ls_ratio_source"]() == 2
    assert cache.calls and set(cache.calls) == {NOW - HOUR}


@pytest.mark.parametrize("status", ["stale", "error"])
def test_cache_failure_never_falls_back_to_csv_as_ready_input(sources, status):
    source, cache = sources
    finish_warmup(source)
    cache.status = status
    assert not source.readiness.snapshot()["ready"]
    assert math.isnan(source["oi_source"]())
    assert math.isnan(source["ls_ratio_source"]())
    assert json.loads(json.dumps(source.readiness.snapshot(), allow_nan=False))["open_interest"] is None


@pytest.mark.parametrize("defect", ["future", "too_old", "old_receipt", "future_receipt", "nan", "inf", "zero"])
def test_invalid_cache_fact_cannot_claim_readiness(sources, defect):
    source, cache = sources
    finish_warmup(source)
    if defect == "future":
        cache.offset = 1
    elif defect == "too_old":
        cache.offset = -3_900_001
    elif defect == "old_receipt":
        cache.received = NOW - 120_001
    elif defect == "future_receipt":
        cache.received = NOW + 1
    else:
        cache.value = {"nan": math.nan, "inf": math.inf, "zero": 0}[defect]
    assert not source.readiness.snapshot()["ready"]
    assert math.isnan(source["oi_source"]())


def test_cached_success_expires_without_another_market_bar(sources):
    source, _ = sources
    finish_warmup(source)
    assert source.readiness.snapshot()["ready"]
    source.readiness.clock = lambda: (NOW + 120_001) / 1000
    assert source.readiness.snapshot()["status"] == "stale"


def test_external_batch_condition_blocks_entries_but_preserves_valid_input_values(sources):
    source, _ = sources
    finish_warmup(source)
    source.readiness.set_external_status("batch_inputs", False, "batch CSV expired")
    assert not source.readiness.snapshot()["ready"]
    assert "batch_inputs" in source.readiness.snapshot()["reason"]
    assert source["oi_source"]() == 999
    source.readiness.set_external_status("batch_inputs", True)
    assert source.readiness.snapshot()["ready"]


@pytest.mark.parametrize("event", [SimpleNamespace(qty=Decimal(0)),
                                   SimpleNamespace(qty=Decimal(1), reduce_only=True)])
def test_data_failure_does_not_prevent_reduction_risk_checks(sources, event):
    source, cache = sources
    finish_warmup(source)
    cache.status = "error"
    events = []
    risk = SimpleNamespace(check=lambda order: events.append(order) or GateResult(True))
    gate = builder.DataReadinessGate(source.readiness, risk)
    assert gate.check(event).allowed
    assert events == [event]


def test_real_builder_checks_data_at_execution_boundary(tmp_path, monkeypatch):
    data = tmp_path / "data"
    csv_inputs(data, [NOW - 3 * HOUR, NOW - 2 * HOUR, NOW - HOUR])
    monkeypatch.setattr(builder, "DATA_DIR", data)
    monkeypatch.setitem(builder.SYMBOL_CONFIG, "BTCUSDT", {"symbol": "BTCUSDT", "interval": "60", "warmup": 2})
    monkeypatch.setenv("RISK_SCOPE", "bybit-readiness-test")
    monkeypatch.setenv("ORDER_JOURNAL_PATH", str(tmp_path / "orders.sqlite"))
    monkeypatch.setattr(builder, "get_initial_balance", lambda adapter: 1000)
    calls = []
    exchange = SimpleNamespace(
        venue="bybit", get_positions=lambda **kwargs: (), get_open_orders=lambda **kwargs: (),
        send_market_order=lambda *args, **kwargs: calls.append((args, kwargs)) or
        {"status": "submitted", "orderId": "fixture-order"},
        get_order=lambda *args, **kwargs: None,
    )
    risk_events = []
    risk = SimpleNamespace(check=lambda event: risk_events.append(event) or GateResult(True))
    info = {"zscore_window": 10, "zscore_warmup": 2, "horizon_models": [], "config": {},
            "deadzone": 0.5, "min_hold": 1, "max_hold": 10}
    cache = FakeCache()
    coord, module = builder.build_coordinator("BTCUSDT", "BTCUSDT", info, exchange,
                                              oi_cache=cache, dry_run=False, risk_gate=risk)
    coord._data_readiness.clock = lambda: NOW / 1000
    coord.start()
    try:
        assert not coord._data_readiness.snapshot()["ready"]
        for index, ts in enumerate((NOW - 3 * HOUR, NOW - 2 * HOUR, NOW - HOUR)):
            if index == 2:
                coord._data_readiness.set_live()
            coord.emit(MarketEvent(
                header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="offline-data-test"),
                ts=datetime.fromtimestamp(ts / 1000, tz=timezone.utc), symbol="BTCUSDT",
                open=Decimal(100), high=Decimal(102), low=Decimal(99), close=Decimal(101), volume=Decimal(10),
            ))
        assert coord._data_readiness.valid_warmup_bars == 2
        assert coord._data_readiness.snapshot()["ready"]
        assert isinstance(coord._execution_bridge.risk_gate, builder.DataReadinessGate)
        assert reconcile_execution(coord, module, restore=True) == []
        cache.status = "error"
        coord.emit(module._make_open_order(100, 1, Decimal(1))[0])
        assert calls == [] and risk_events == []
        cache.status = "ready"
        coord.emit(module._make_open_order(100, 1, Decimal(1))[0])
        assert len(calls) == 1 and len(risk_events) == 1
        cache.status = "stale"
        coord.emit(module._make_open_order(100, 1, Decimal(1))[0])
        assert len(calls) == 1 and len(risk_events) == 1
    finally:
        coord.stop()

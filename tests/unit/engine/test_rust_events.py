"""Tests for Phase 2: Rust event types and fast-path reducers.

Verifies that RustMarketEvent, RustFillEvent, RustFundingEvent produce
identical state mutations as Python SimpleNamespace events via both
RustStateStore and rust_pipeline_apply.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("_quant_hotpath")

from _quant_hotpath import (
    RustMarketEvent, RustFillEvent, RustFundingEvent,
    RustStateStore, RustMarketState, RustAccountState, RustPositionState,
    rust_pipeline_apply, rust_detect_event_kind,
)

_SCALE = 100_000_000


# ── detect_kind ──

def test_detect_kind_rust_market_event():
    me = RustMarketEvent("BTCUSDT", 100.0, 105.0, 99.0, 102.0, 500.0)
    assert rust_detect_event_kind(me) == "MARKET"


def test_detect_kind_rust_fill_event():
    fe = RustFillEvent("BTCUSDT", "buy", 1.5, 100.0)
    assert rust_detect_event_kind(fe) == "FILL"


def test_detect_kind_rust_funding_event():
    fu = RustFundingEvent("BTCUSDT", 0.0001, 40000.0, 2.0)
    assert rust_detect_event_kind(fu) == "FUNDING"


# ── State store parity: Rust events vs Python events ──

def _make_py_market(sym, o, h, l, c, v, ts=None):
    return SimpleNamespace(
        event_type="market",
        header=SimpleNamespace(event_type="market", ts=ts, event_id="m-1"),
        symbol=sym, open=str(o), high=str(h), low=str(l), close=str(c), volume=str(v),
    )


def _make_py_fill(sym, side, qty, price, fee=0, rpnl=0, mc=0, cd=0, ts=None):
    return SimpleNamespace(
        event_type="fill",
        header=SimpleNamespace(event_type="fill", ts=ts, event_id="f-1"),
        symbol=sym, side=side, qty=str(qty), price=str(price),
        fee=str(fee), realized_pnl=str(rpnl), margin_change=str(mc), cash_delta=str(cd),
    )


def _make_py_funding(sym, rate, mark, pos_qty, ts=None):
    return SimpleNamespace(
        event_type="funding",
        header=SimpleNamespace(event_type="funding", ts=ts, event_id="u-1"),
        symbol=sym, funding_rate=str(rate), mark_price=str(mark), position_qty=str(pos_qty),
    )


def test_store_market_parity():
    """Rust market event produces same state as Python market event."""
    rust_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)
    py_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)

    rust_ev = RustMarketEvent("BTCUSDT", 99.0, 101.0, 98.0, 100.0, 2.0, "2024-01-01T00:00:00Z")
    py_ev = _make_py_market("BTCUSDT", 99, 101, 98, 100, 2, "2024-01-01T00:00:00Z")

    r1 = rust_store.process_event(rust_ev, "BTCUSDT")
    r2 = py_store.process_event(py_ev, "BTCUSDT")

    assert r1.kind == r2.kind == "MARKET"
    assert r1.changed == r2.changed is True

    rm = rust_store.get_market("BTCUSDT")
    pm = py_store.get_market("BTCUSDT")
    assert rm.close_f == pytest.approx(pm.close_f)
    assert rm.open_f == pytest.approx(pm.open_f)
    assert rm.high_f == pytest.approx(pm.high_f)
    assert rm.low_f == pytest.approx(pm.low_f)


def test_store_fill_parity():
    """Rust fill event produces same state as Python fill event."""
    rust_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)
    py_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)

    rust_ev = RustFillEvent("BTCUSDT", "buy", 2.0, 100.0, fee=1.0, realized_pnl=5.0, margin_change=10.0)
    py_ev = _make_py_fill("BTCUSDT", "buy", 2, 100, fee=1, rpnl=5, mc=10)

    rust_store.process_event(rust_ev, "BTCUSDT")
    py_store.process_event(py_ev, "BTCUSDT")

    ra = rust_store.get_account()
    pa = py_store.get_account()
    assert ra.balance_f == pytest.approx(pa.balance_f)
    assert ra.fees_paid_f == pytest.approx(pa.fees_paid_f)

    rp = rust_store.get_position("BTCUSDT")
    pp = py_store.get_position("BTCUSDT")
    assert rp.qty_f == pytest.approx(pp.qty_f)
    assert rp.avg_price_f == pytest.approx(pp.avg_price_f)


def test_store_funding_parity():
    """Rust funding event produces same state as Python funding event."""
    rust_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)
    py_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)

    rust_ev = RustFundingEvent("BTCUSDT", 0.0001, 40000.0, 2.0)
    py_ev = _make_py_funding("BTCUSDT", 0.0001, 40000, 2)

    rust_store.process_event(rust_ev, "BTCUSDT")
    py_store.process_event(py_ev, "BTCUSDT")

    ra = rust_store.get_account()
    pa = py_store.get_account()
    assert ra.balance_f == pytest.approx(pa.balance_f, abs=1e-4)
    assert ra.fees_paid_f == pytest.approx(pa.fees_paid_f, abs=1e-4)


# ── Pipeline apply parity ──

def test_pipeline_apply_rust_market_event():
    """rust_pipeline_apply works with RustMarketEvent."""
    m = RustMarketState.empty("BTCUSDT")
    p = RustPositionState.empty("BTCUSDT")
    a = RustAccountState.initial(currency="USDT", balance=10000 * _SCALE)

    rust_ev = RustMarketEvent("BTCUSDT", 99.0, 101.0, 98.0, 100.0, 2.0)
    result = rust_pipeline_apply(m, p, a, rust_ev)
    assert result is not None
    new_m, new_p, new_a, changed = result
    assert changed
    assert new_m.close_f == pytest.approx(100.0)


def test_pipeline_apply_rust_fill_event():
    """rust_pipeline_apply works with RustFillEvent."""
    m = RustMarketState.empty("BTCUSDT")
    p = RustPositionState.empty("BTCUSDT")
    a = RustAccountState.initial(currency="USDT", balance=10000 * _SCALE)

    rust_ev = RustFillEvent("BTCUSDT", "buy", 2.0, 100.0, fee=1.0, realized_pnl=5.0, margin_change=10.0)
    result = rust_pipeline_apply(m, p, a, rust_ev)
    assert result is not None
    new_m, new_p, new_a, changed = result
    assert changed
    assert new_p.qty_f == pytest.approx(2.0)
    assert new_a.balance_f == pytest.approx(10004.0)  # 10000 + 5 - 1


def test_pipeline_apply_rust_funding_event():
    """rust_pipeline_apply works with RustFundingEvent."""
    m = RustMarketState.empty("BTCUSDT")
    p = RustPositionState.empty("BTCUSDT")
    a = RustAccountState.initial(currency="USDT", balance=10000 * _SCALE)

    rust_ev = RustFundingEvent("BTCUSDT", 0.0001, 40000.0, 2.0)
    result = rust_pipeline_apply(m, p, a, rust_ev)
    assert result is not None
    _, _, new_a, changed = result
    assert changed
    # funding = 2 * 40000 * 0.0001 = 8
    assert new_a.balance_f == pytest.approx(10000.0 - 8.0, abs=1e-4)


# ── Event property access ──

def test_market_event_properties():
    me = RustMarketEvent("BTCUSDT", 99.5, 101.0, 98.5, 100.25, 1500.0, "2024-01-01T00:00:00Z")
    assert me.symbol == "BTCUSDT"
    assert me.open_f == pytest.approx(99.5)
    assert me.close_f == pytest.approx(100.25)
    assert me.event_type == "market"
    assert me.ts == "2024-01-01T00:00:00Z"


def test_fill_event_side_normalization():
    f1 = RustFillEvent("BTCUSDT", "BUY", 1.0, 100.0)
    assert f1.side == "buy"
    f2 = RustFillEvent("BTCUSDT", "SELL", 1.0, 100.0)
    assert f2.side == "sell"
    f3 = RustFillEvent("BTCUSDT", "long", 1.0, 100.0)
    assert f3.side == "buy"
    f4 = RustFillEvent("BTCUSDT", "short", 1.0, 100.0)
    assert f4.side == "sell"


# ── Multi-event sequence parity ──

def test_full_sequence_parity():
    """Multi-event sequence: Rust events vs Python events produce same final state."""
    rust_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)
    py_store = RustStateStore(["BTCUSDT"], "USDT", 10000 * _SCALE)

    rust_events = [
        RustMarketEvent("BTCUSDT", 99.0, 101.0, 98.0, 100.0, 2.0),
        RustFillEvent("BTCUSDT", "buy", 2.0, 100.0, fee=1.0, realized_pnl=5.0, margin_change=10.0),
        RustFundingEvent("BTCUSDT", 0.0001, 40000.0, 2.0),
        RustFillEvent("BTCUSDT", "sell", 1.0, 110.0, fee=0.5, realized_pnl=10.0, margin_change=-5.0),
    ]
    py_events = [
        _make_py_market("BTCUSDT", 99, 101, 98, 100, 2),
        _make_py_fill("BTCUSDT", "buy", 2, 100, fee=1, rpnl=5, mc=10),
        _make_py_funding("BTCUSDT", 0.0001, 40000, 2),
        _make_py_fill("BTCUSDT", "sell", 1, 110, fee=0.5, rpnl=10, mc=-5),
    ]

    for re, pe in zip(rust_events, py_events):
        rust_store.process_event(re, "BTCUSDT")
        py_store.process_event(pe, "BTCUSDT")

    rm = rust_store.get_market("BTCUSDT")
    pm = py_store.get_market("BTCUSDT")
    assert rm.close_f == pytest.approx(pm.close_f)

    ra = rust_store.get_account()
    pa = py_store.get_account()
    assert ra.balance_f == pytest.approx(pa.balance_f, abs=1e-4)
    assert ra.fees_paid_f == pytest.approx(pa.fees_paid_f, abs=1e-4)

    rp = rust_store.get_position("BTCUSDT")
    pp = py_store.get_position("BTCUSDT")
    assert rp.qty_f == pytest.approx(pp.qty_f)

    assert rust_store.event_index == py_store.event_index

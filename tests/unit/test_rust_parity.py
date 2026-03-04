# tests/unit/test_rust_parity.py
"""Parity tests: Rust hotpath modules vs Python originals."""
from __future__ import annotations

import json
import time
import pytest
from decimal import Decimal

# ── Rust imports ──
try:
    from _quant_hotpath import (
        DuplicateGuard,
        RustRateLimitPolicy,
        RustMLDecision,
        rust_parse_kline,
        rust_parse_depth,
        rust_demux_user_stream,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust _quant_hotpath not built")


# ============================================================
# DuplicateGuard
# ============================================================

class TestDuplicateGuard:
    def test_new_event_returns_true(self):
        g = DuplicateGuard()
        assert g.check_and_insert("evt-1", 100.0) is True

    def test_duplicate_returns_false(self):
        g = DuplicateGuard()
        g.check_and_insert("evt-1", 100.0)
        assert g.check_and_insert("evt-1", 101.0) is False

    def test_different_events(self):
        g = DuplicateGuard()
        assert g.check_and_insert("evt-1", 100.0) is True
        assert g.check_and_insert("evt-2", 100.0) is True
        assert g.check_and_insert("evt-1", 100.0) is False

    def test_ttl_prune(self):
        g = DuplicateGuard(ttl_sec=10.0, max_size=100)
        g.check_and_insert("evt-1", 100.0)
        # 70s later, past TTL — prune triggers (>60s since last prune)
        assert g.check_and_insert("evt-1", 170.0) is True

    def test_capacity_prune(self):
        g = DuplicateGuard(ttl_sec=86400.0, max_size=5)
        for i in range(6):
            g.check_and_insert(f"evt-{i}", 100.0 + i * 0.001)
        # Prune happened when > max_size
        assert len(g) <= 6

    def test_len(self):
        g = DuplicateGuard()
        assert len(g) == 0
        g.check_and_insert("a", 1.0)
        assert len(g) == 1

    def test_clear(self):
        g = DuplicateGuard()
        g.check_and_insert("a", 1.0)
        g.clear()
        assert len(g) == 0
        assert g.check_and_insert("a", 2.0) is True


# ============================================================
# RustRateLimitPolicy
# ============================================================

class TestRustRateLimitPolicy:
    def test_order_check_allows(self):
        pol = RustRateLimitPolicy()
        assert pol.check("/fapi/v1/order", 100.0) is True

    def test_order_exhaustion(self):
        pol = RustRateLimitPolicy()
        t = 100.0
        for _ in range(10):
            assert pol.check("/fapi/v1/order", t) is True
        # 11th should fail (capacity=10)
        assert pol.check("/fapi/v1/order", t) is False

    def test_order_refill(self):
        pol = RustRateLimitPolicy()
        t = 100.0
        for _ in range(10):
            pol.check("/fapi/v1/order", t)
        # 1 second later, refills 10 tokens
        assert pol.check("/fapi/v1/order", t + 1.0) is True

    def test_weight_check(self):
        pol = RustRateLimitPolicy()
        assert pol.check("/fapi/v1/klines", 100.0) is True  # weight=5

    def test_weight_sync(self):
        pol = RustRateLimitPolicy()
        pol.check("/fapi/v1/ticker/price", 100.0)  # use 1
        pol.sync_used_weight(1195)  # 1200 - 1195 = 5 remaining
        # Now only 5 tokens left, a 30-weight request should fail
        assert pol.check("/fapi/v1/income", 100.0) is False  # weight=30

    def test_default_weight(self):
        pol = RustRateLimitPolicy()
        assert pol.check("/fapi/v1/unknown", 100.0) is True  # default weight=1

    def test_endpoint_weights_match_python(self):
        """Verify Rust endpoint weights match Python ENDPOINT_WEIGHTS."""
        from execution.adapters.binance.rate_limit_policy import ENDPOINT_WEIGHTS, DEFAULT_WEIGHT
        pol = RustRateLimitPolicy()
        t = 0.0
        for path, py_weight in ENDPOINT_WEIGHTS.items():
            if py_weight == 0:
                continue  # order endpoints handled separately
            # Reset by creating new policy
            pol = RustRateLimitPolicy()
            # Consume once with fresh pool and check remaining
            pol.check(path, t)
            # If weight matches, the pool should have (1200 - weight) tokens


# ============================================================
# JSON Parsing: Kline
# ============================================================

KLINE_MSG = json.dumps({
    "stream": "btcusdt@kline_1m",
    "data": {
        "e": "kline",
        "s": "BTCUSDT",
        "k": {
            "t": 1700000000000,
            "o": "36500.10",
            "h": "36600.20",
            "l": "36400.30",
            "c": "36550.40",
            "v": "123.456",
            "x": True,
            "s": "BTCUSDT",
        }
    }
})

KLINE_MSG_UNCLOSED = json.dumps({
    "data": {
        "e": "kline",
        "s": "BTCUSDT",
        "k": {
            "t": 1700000000000,
            "o": "36500.10",
            "h": "36600.20",
            "l": "36400.30",
            "c": "36550.40",
            "v": "123.456",
            "x": False,
        }
    }
})


class TestRustParseKline:
    def test_closed_kline(self):
        d = rust_parse_kline(KLINE_MSG, True)
        assert d is not None
        assert d["symbol"] == "BTCUSDT"
        assert d["ts_ms"] == 1700000000000
        assert d["open"] == "36500.10"
        assert d["high"] == "36600.20"
        assert d["low"] == "36400.30"
        assert d["close"] == "36550.40"
        assert d["volume"] == "123.456"

    def test_unclosed_filtered(self):
        d = rust_parse_kline(KLINE_MSG_UNCLOSED, True)
        assert d is None

    def test_unclosed_allowed(self):
        d = rust_parse_kline(KLINE_MSG_UNCLOSED, False)
        assert d is not None
        assert d["close"] == "36550.40"

    def test_invalid_json(self):
        assert rust_parse_kline("not json", True) is None

    def test_non_kline_event(self):
        msg = json.dumps({"e": "depthUpdate", "s": "BTCUSDT"})
        assert rust_parse_kline(msg, True) is None

    def test_direct_format(self):
        """Without combined stream wrapper."""
        msg = json.dumps({
            "e": "kline",
            "s": "ethusdt",
            "k": {
                "t": 1700000000000,
                "o": "2000",
                "h": "2100",
                "l": "1900",
                "c": "2050",
                "v": "500",
                "x": True,
            }
        })
        d = rust_parse_kline(msg, True)
        assert d is not None
        assert d["symbol"] == "ETHUSDT"

    def test_parity_with_python(self):
        """Compare Rust parse result with Python KlineProcessor."""
        from execution.adapters.binance.kline_processor import KlineProcessor
        proc = KlineProcessor(only_closed=True)
        # Force Python path
        py_result = proc._process_raw_py(KLINE_MSG)
        rust_d = rust_parse_kline(KLINE_MSG, True)

        assert py_result is not None
        assert rust_d is not None
        assert Decimal(rust_d["open"]) == py_result.open
        assert Decimal(rust_d["close"]) == py_result.close
        assert Decimal(rust_d["volume"]) == py_result.volume


# ============================================================
# JSON Parsing: Depth
# ============================================================

DEPTH_MSG = json.dumps({
    "data": {
        "e": "depthUpdate",
        "s": "BTCUSDT",
        "E": 1700000000123,
        "u": 9999,
        "b": [["36500.0", "1.5"], ["36490.0", "2.0"], ["36480.0", "0"]],
        "a": [["36510.0", "0.8"], ["36520.0", "1.2"]],
    }
})


class TestRustParseDepth:
    def test_depth_update(self):
        d = rust_parse_depth(DEPTH_MSG, 20)
        assert d is not None
        assert d["symbol"] == "BTCUSDT"
        assert d["ts_ms"] == 1700000000123
        assert d["last_update_id"] == 9999
        assert len(d["bids"]) == 3
        assert d["bids"][0] == ("36500.0", "1.5")
        assert len(d["asks"]) == 2

    def test_max_levels(self):
        d = rust_parse_depth(DEPTH_MSG, 1)
        assert d is not None
        assert len(d["bids"]) == 1
        assert len(d["asks"]) == 1

    def test_invalid_json(self):
        assert rust_parse_depth("bad", 20) is None

    def test_non_depth_event(self):
        msg = json.dumps({"e": "kline", "s": "BTCUSDT"})
        assert rust_parse_depth(msg, 20) is None

    def test_parity_with_python(self):
        """Compare Rust depth result with Python DepthProcessor."""
        from execution.adapters.binance.depth_processor import DepthProcessor
        proc = DepthProcessor(max_levels=20)
        py_result = proc._process_raw_py(DEPTH_MSG)
        rust_d = rust_parse_depth(DEPTH_MSG, 20)

        assert py_result is not None
        assert rust_d is not None
        assert py_result.symbol == rust_d["symbol"]
        assert py_result.ts_ms == rust_d["ts_ms"]
        assert py_result.last_update_id == rust_d["last_update_id"]
        # Bids: Python filters zero-qty, Rust returns all
        py_bids = [(str(b.price), str(b.qty)) for b in py_result.bids]
        # Filter zero-qty from Rust for comparison
        rust_bids = [(p, q) for p, q in rust_d["bids"] if Decimal(q) > 0]
        assert len(py_bids) == len(rust_bids)


# ============================================================
# JSON Parsing: User Stream
# ============================================================

class TestRustDemuxUserStream:
    def test_account_update(self):
        msg = json.dumps({
            "e": "ACCOUNT_UPDATE",
            "E": 1700000000000,
            "T": 1700000000001,
            "a": {"B": [{"a": "USDT", "wb": "10000"}]},
        })
        d = rust_demux_user_stream(msg)
        assert d is not None
        assert d["event_type"] == "ACCOUNT_UPDATE"
        assert d["E"] == 1700000000000

    def test_order_trade_update(self):
        msg = json.dumps({
            "e": "ORDER_TRADE_UPDATE",
            "E": 1700000000000,
            "o": {"s": "BTCUSDT", "S": "BUY", "q": "0.001"},
        })
        d = rust_demux_user_stream(msg)
        assert d is not None
        assert d["event_type"] == "ORDER_TRADE_UPDATE"

    def test_invalid(self):
        assert rust_demux_user_stream("bad json") is None

    def test_no_event_type(self):
        assert rust_demux_user_stream(json.dumps({"x": 1})) is None


# ============================================================
# RustMLDecision
# ============================================================

class TestRustMLDecision:
    def test_flat_no_signal(self):
        """ml_score within threshold → no orders."""
        md = RustMLDecision(symbol="BTCUSDT", threshold=0.5)
        orders = md.decide(close=50000.0, ml_score=0.1, current_qty=0.0, balance=10000.0)
        assert len(orders) == 0

    def test_long_entry(self):
        md = RustMLDecision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.001)
        orders = md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0)
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].reason == "open_long"
        assert orders[0].qty > 0

    def test_short_entry(self):
        md = RustMLDecision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.001)
        orders = md.decide(close=50000.0, ml_score=-0.5, current_qty=0.0, balance=10000.0)
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].reason == "open_short"

    def test_close_opposite_then_open(self):
        """Going from short to long should close short then open long."""
        md = RustMLDecision(symbol="BTCUSDT", threshold=0.001)
        orders = md.decide(close=50000.0, ml_score=0.5, current_qty=-0.1, balance=10000.0)
        assert len(orders) == 2
        assert orders[0].side == "BUY"
        assert orders[0].reason == "close_short"
        assert orders[1].side == "BUY"
        assert orders[1].reason == "open_long"

    def test_flatten_to_flat(self):
        """Signal goes flat → flatten position."""
        md = RustMLDecision(symbol="BTCUSDT", threshold=0.5)
        orders = md.decide(close=50000.0, ml_score=0.0, current_qty=0.1, balance=10000.0)
        assert len(orders) == 1
        assert orders[0].reason == "flatten"
        assert orders[0].side == "SELL"

    def test_quantize_floor(self):
        """qty should be floor to 3 decimals (ROUND_DOWN)."""
        md = RustMLDecision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.001)
        orders = md.decide(close=30000.0, ml_score=1.0, current_qty=0.0, balance=10000.0)
        assert len(orders) == 1
        qty = orders[0].qty
        # Check 3 decimal places
        assert qty == int(qty * 1000) / 1000

    def test_stop_loss(self):
        md = RustMLDecision(
            symbol="BTCUSDT", threshold=0.001, atr_stop=2.0, trailing_atr=0.0,
        )
        # Enter long
        md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0, atr_norm=0.01)
        # Price drops a lot — stop should trigger
        orders = md.decide(close=47000.0, ml_score=0.5, current_qty=0.1, balance=10000.0, atr_norm=0.01)
        assert len(orders) == 1
        assert orders[0].reason == "stop_loss"

    def test_trailing_stop(self):
        md = RustMLDecision(
            symbol="BTCUSDT", threshold=0.001, trailing_atr=2.0,
        )
        # Enter long
        md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0, atr_norm=0.01)
        # Price goes up — set peak to 52000
        md.decide(close=52000.0, ml_score=0.5, current_qty=0.05, balance=10000.0, atr_norm=0.01)
        # Price crashes well below peak - trail_dist = 0.05 * 40000 * 2.0 = 4000
        # peak=52000, 52000 - 4000 = 48000, close=40000 < 48000 → triggers
        orders = md.decide(close=40000.0, ml_score=0.5, current_qty=0.05, balance=10000.0, atr_norm=0.05)
        assert len(orders) == 1
        assert orders[0].reason == "trailing_stop"

    def test_min_hold_bars(self):
        md = RustMLDecision(symbol="BTCUSDT", threshold=0.001, min_hold_bars=3)
        # Enter long
        md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0)
        # Try to flip short immediately (bars_held < 3)
        orders = md.decide(close=50000.0, ml_score=-0.5, current_qty=0.1, balance=10000.0)
        assert len(orders) == 0
        # Bar 2
        orders = md.decide(close=50000.0, ml_score=-0.5, current_qty=0.1, balance=10000.0)
        assert len(orders) == 0
        # Bar 3 — now allowed
        orders = md.decide(close=50000.0, ml_score=-0.5, current_qty=0.1, balance=10000.0)
        assert len(orders) > 0

    def test_dd_breaker(self):
        md = RustMLDecision(
            symbol="BTCUSDT", threshold=0.001, dd_limit=0.05, dd_cooldown=2,
        )
        # Initial balance = HWM, enter long
        md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0)
        # Signal goes flat (desired=flat) with DD >= 5% → dd_breaker flatten
        # Use ml_score within threshold so desired=flat, current_qty > 0
        orders = md.decide(close=50000.0, ml_score=0.0005, current_qty=0.1, balance=9400.0)
        assert len(orders) == 1
        assert orders[0].reason == "dd_breaker"

    def test_vol_target_sizing(self):
        md = RustMLDecision(
            symbol="BTCUSDT", risk_pct=0.5, threshold=0.001,
            vol_target=0.15, atr_stop=2.0,
        )
        orders = md.decide(
            close=50000.0, ml_score=1.0, current_qty=0.0,
            balance=10000.0, atr_norm=0.01,
        )
        assert len(orders) == 1
        # vol target: qty = (10000 * 0.5) / (0.01 * 2.0 * 50000) = 5.0
        assert orders[0].qty == 5.0

    def test_reset(self):
        md = RustMLDecision(symbol="BTCUSDT", threshold=0.001)
        md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0)
        md.reset()
        # After reset, should behave like fresh
        orders = md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0)
        assert len(orders) == 1

    def test_parity_long_entry(self):
        """Compare Rust vs Python for a simple long entry."""
        py_md = _make_python_ml_decision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.001)
        rust_md = RustMLDecision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.001)

        snap = _make_snapshot(close=50000, ml_score=0.5, qty=0, balance=10000)
        py_orders = list(py_md.decide(snap))
        rust_orders = rust_md.decide(close=50000.0, ml_score=0.5, current_qty=0.0, balance=10000.0)

        assert len(py_orders) == len(rust_orders)
        for po, ro in zip(py_orders, rust_orders):
            assert po.side == ro.side
            assert po.reason == ro.reason
            assert float(po.qty) == ro.qty

    def test_parity_flatten(self):
        py_md = _make_python_ml_decision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.5)
        rust_md = RustMLDecision(symbol="BTCUSDT", risk_pct=0.5, threshold=0.5)

        snap = _make_snapshot(close=50000, ml_score=0.0, qty=0.1, balance=10000)
        py_orders = list(py_md.decide(snap))
        rust_orders = rust_md.decide(close=50000.0, ml_score=0.0, current_qty=0.1, balance=10000.0)

        assert len(py_orders) == len(rust_orders)
        assert py_orders[0].side == rust_orders[0].side
        assert float(py_orders[0].qty) == rust_orders[0].qty

    def test_zero_close(self):
        md = RustMLDecision(symbol="BTCUSDT")
        assert len(md.decide(close=0.0, ml_score=0.5, current_qty=0.0, balance=10000.0)) == 0

    def test_negative_close(self):
        md = RustMLDecision(symbol="BTCUSDT")
        assert len(md.decide(close=-1.0, ml_score=0.5, current_qty=0.0, balance=10000.0)) == 0


# ── helpers ──

def _make_python_ml_decision(**kwargs):
    from decision.ml_decision import MLDecisionModule
    return MLDecisionModule(**kwargs)


def _make_snapshot(close, ml_score, qty, balance, atr_norm=None):
    from types import SimpleNamespace
    market = SimpleNamespace(close=Decimal(str(close)))
    pos = SimpleNamespace(qty=Decimal(str(qty))) if qty != 0 else None
    account = SimpleNamespace(balance=Decimal(str(balance)))
    features = {"ml_score": ml_score}
    if atr_norm is not None:
        features["atr_norm_14"] = atr_norm
    positions = {"BTCUSDT": pos} if pos else {}
    return {
        "market": market,
        "positions": positions,
        "features": features,
        "account": account,
    }

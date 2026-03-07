"""Parity tests for Rust engine Sprint B: pipeline + guards."""
import pytest
from types import SimpleNamespace

from _quant_hotpath import (
    rust_detect_event_kind,
    rust_normalize_to_facts,
    RustGuardConfig,
    RustBasicGuard,
)


# ============================================================
# rust_detect_event_kind
# ============================================================

class TestDetectEventKind:
    @pytest.mark.parametrize("event_type,expected", [
        ("MARKET_DATA", "MARKET"),
        ("market_update", "MARKET"),
        ("FILL_REPORT", "FILL"),
        ("fill", "FILL"),
        ("FUNDING_RATE", "FUNDING"),
        ("funding_settlement", "FUNDING"),
        ("ORDER_UPDATE", "ORDER"),
        ("order_submit", "ORDER"),
        ("SIGNAL_ML", "SIGNAL"),
        ("signal_alpha", "SIGNAL"),
        ("INTENT_NEW", "INTENT"),
        ("RISK_CHECK", "RISK"),
        ("CONTROL_STOP", "CONTROL"),
        ("SOMETHING_ELSE", "UNKNOWN"),
        ("", "UNKNOWN"),
    ])
    def test_event_type_attr(self, event_type, expected):
        ev = SimpleNamespace(event_type=event_type)
        assert rust_detect_event_kind(ev) == expected

    def test_enum_with_value(self):
        """event_type may be an Enum with .value attribute."""
        class FakeEnum:
            value = "MARKET_DATA"
        ev = SimpleNamespace(event_type=FakeEnum())
        assert rust_detect_event_kind(ev) == "MARKET"

    def test_event_type_class_attr(self):
        """Falls back to EVENT_TYPE class attribute."""

        class EvWithClassAttr:
            EVENT_TYPE = "market_bar"

        assert rust_detect_event_kind(EvWithClassAttr()) == "MARKET"

    def test_no_attrs(self):
        """Object with no event_type or EVENT_TYPE -> UNKNOWN."""

        class Bare:
            pass

        assert rust_detect_event_kind(Bare()) == "UNKNOWN"


# ============================================================
# rust_normalize_to_facts
# ============================================================

class TestNormalizeToFacts:
    def test_market(self):
        ev = SimpleNamespace(
            event_type="MARKET_DATA", header=None, symbol="BTCUSDT",
            open="100", high="110", low="90", close="105", volume="1000", ts=123,
        )
        facts = rust_normalize_to_facts(ev)
        assert len(facts) == 1
        f = facts[0]
        assert f.event_type == "market"
        assert f.symbol == "BTCUSDT"
        assert f.open == "100"
        assert f.high == "110"
        assert f.low == "90"
        assert f.close == "105"
        assert f.volume == "1000"
        assert f.ts == 123

    def test_funding(self):
        ev = SimpleNamespace(
            event_type="FUNDING_RATE", header=None, symbol="BTCUSDT",
            funding_rate=0.001, mark_price=50000.0, ts=999,
        )
        facts = rust_normalize_to_facts(ev)
        assert len(facts) == 1
        assert facts[0].event_type == "funding"
        assert facts[0].funding_rate == 0.001
        assert facts[0].mark_price == 50000.0

    def test_order(self):
        ev = SimpleNamespace(
            event_type="ORDER_UPDATE", header=None, symbol="BTCUSDT",
            venue="binance", order_id="123", client_order_id="c456",
            status="FILLED", side="buy", order_type="LIMIT", tif="GTC",
            qty="1.0", price="50000", filled_qty="1.0", avg_price="50000",
            order_key="ok1", payload_digest="abc",
        )
        facts = rust_normalize_to_facts(ev)
        assert len(facts) == 1
        assert facts[0].event_type == "ORDER_UPDATE"
        assert facts[0].order_id == "123"

    def test_fill_buy(self):
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            side="buy", qty=0.5, price=100.0, fee=0.01,
            realized_pnl=0.0, margin_change=50.0,
        )
        facts = rust_normalize_to_facts(ev)
        assert len(facts) == 1
        f = facts[0]
        assert f.event_type == "FILL"
        assert f.side == "buy"
        assert f.qty == 0.5
        assert f.quantity == 0.5
        assert f.price == 100.0
        assert f.fee == 0.01
        assert f.realized_pnl == 0.0
        assert f.margin_change == 50.0

    def test_fill_sell(self):
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            side="sell", qty=1.0, price=200.0, fee=0.02,
            realized_pnl=10.0, margin_change=0.0,
        )
        facts = rust_normalize_to_facts(ev)
        assert facts[0].side == "sell"

    def test_fill_short_normalized(self):
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            side="short", qty=-1.0, price=200.0, fee=0.02,
            realized_pnl=10.0, margin_change=0.0,
        )
        facts = rust_normalize_to_facts(ev)
        assert facts[0].side == "sell"
        assert facts[0].qty == 1.0  # abs

    def test_fill_long_normalized(self):
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            side="long", qty=2.0, price=100.0, fee=0.0,
            realized_pnl=0.0, margin_change=0.0,
        )
        facts = rust_normalize_to_facts(ev)
        assert facts[0].side == "buy"

    def test_fill_missing_side_raises(self):
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            qty=1.0, price=100.0, fee=0.0,
            realized_pnl=0.0, margin_change=0.0,
        )
        with pytest.raises(RuntimeError, match="side"):
            rust_normalize_to_facts(ev)

    def test_fill_bad_side_raises(self):
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            side="invalid", qty=1.0, price=100.0, fee=0.0,
            realized_pnl=0.0, margin_change=0.0,
        )
        with pytest.raises(RuntimeError, match="fill side"):
            rust_normalize_to_facts(ev)

    def test_fill_quantity_fallback(self):
        """If .qty is None, falls back to .quantity."""
        ev = SimpleNamespace(
            event_type="FILL_REPORT", header=None, symbol="BTCUSDT",
            side="buy", quantity=3.0, price=100.0, fee=0.0,
            realized_pnl=0.0, margin_change=0.0,
        )
        facts = rust_normalize_to_facts(ev)
        assert facts[0].qty == 3.0

    def test_unknown_returns_empty(self):
        ev = SimpleNamespace(event_type="SOMETHING_UNKNOWN")
        assert rust_normalize_to_facts(ev) == []

    def test_signal_returns_empty(self):
        ev = SimpleNamespace(event_type="SIGNAL_ML")
        assert rust_normalize_to_facts(ev) == []


# ============================================================
# RustGuardConfig
# ============================================================

class TestGuardConfig:
    def test_defaults(self):
        cfg = RustGuardConfig()
        assert cfg.stop_on_fatal is True
        assert cfg.max_consecutive_errors == 5
        assert cfg.max_consecutive_domain_errors == 5
        assert cfg.max_consecutive_execution_errors == 2
        assert cfg.stop_on_unknown_exception is True
        assert cfg.default_retry_after_s == 0.2

    def test_custom(self):
        cfg = RustGuardConfig(
            stop_on_fatal=False,
            max_consecutive_errors=10,
            max_consecutive_domain_errors=8,
            max_consecutive_execution_errors=3,
            stop_on_unknown_exception=False,
            default_retry_after_s=0.5,
        )
        assert cfg.stop_on_fatal is False
        assert cfg.max_consecutive_errors == 10
        assert cfg.max_consecutive_domain_errors == 8
        assert cfg.max_consecutive_execution_errors == 3
        assert cfg.stop_on_unknown_exception is False
        assert cfg.default_retry_after_s == 0.5

    def test_repr(self):
        cfg = RustGuardConfig()
        r = repr(cfg)
        assert "RustGuardConfig" in r
        assert "stop_on_fatal=true" in r


# ============================================================
# RustBasicGuard
# ============================================================

class TestBasicGuard:
    def test_before_event(self):
        g = RustBasicGuard()
        assert g.before_event() == ("allow", "ok")

    def test_after_event_resets(self):
        g = RustBasicGuard()
        g.on_error("error", "state", "X")
        g.after_event()
        # After reset, next error should not be at threshold
        a, r, d = g.on_error("error", "state", "X")
        assert a == "drop"  # first error -> drop, not stop

    def test_fatal_stops(self):
        g = RustBasicGuard()
        a, r, d = g.on_error("fatal", "state", "CRASH")
        assert a == "stop"
        assert "fatal" in r
        assert d is None

    def test_fatal_with_stop_disabled(self):
        cfg = RustGuardConfig(stop_on_fatal=False)
        g = RustBasicGuard(cfg)
        a, r, d = g.on_error("fatal", "state", "CRASH")
        # Not stopped because stop_on_fatal=False, but also not retryable
        assert a == "drop"

    def test_invariant_always_stops(self):
        g = RustBasicGuard()
        a, r, d = g.on_error("error", "invariant", "BAD_STATE")
        assert a == "stop"
        assert "invariant" in r

    def test_execution_threshold(self):
        g = RustBasicGuard()
        a1, _, _ = g.on_error("error", "execution", "NETWORK")
        assert a1 != "stop"  # first error
        a2, r2, _ = g.on_error("error", "execution", "NETWORK")
        assert a2 == "stop"
        assert "execution errors >= 2" in r2

    def test_domain_consecutive_threshold(self):
        cfg = RustGuardConfig(max_consecutive_domain_errors=3)
        g = RustBasicGuard(cfg)
        g.on_error("error", "state", "X")
        g.on_error("error", "state", "X")
        a, r, _ = g.on_error("error", "state", "X")
        assert a == "stop"
        assert "state errors >= 3" in r

    def test_global_consecutive_threshold(self):
        cfg = RustGuardConfig(max_consecutive_errors=3, max_consecutive_domain_errors=10)
        g = RustBasicGuard(cfg)
        g.on_error("error", "a", "X")
        g.on_error("error", "b", "X")
        a, r, _ = g.on_error("error", "c", "X")
        assert a == "stop"
        assert "consecutive errors >= 3" in r

    def test_retryable(self):
        g = RustBasicGuard()
        a, r, d = g.on_error("error", "execution", "RETRYABLE")
        assert a == "retry"
        assert d == 0.2
        assert "retry" in r

    def test_timeout(self):
        g = RustBasicGuard()
        a, r, d = g.on_error("error", "network", "TIMEOUT")
        assert a == "retry"
        assert d == 0.2

    def test_default_drop(self):
        g = RustBasicGuard()
        a, r, d = g.on_error("error", "state", "SOME_ERROR")
        assert a == "drop"
        assert d is None

    def test_reset(self):
        g = RustBasicGuard()
        g.on_error("error", "execution", "X")
        g.reset()
        # After reset, should be back to zero
        a, _, _ = g.on_error("error", "execution", "X")
        assert a != "stop"  # first error again

    def test_repr(self):
        g = RustBasicGuard()
        assert "RustBasicGuard" in repr(g)

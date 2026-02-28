"""Tests for decision/utils.py, persistence, validators, policies, kill overlay, rebalancing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from decision.utils import stable_hash, dec_str, canonical_meta
from decision.persistence.decision_store import DecisionStore
from decision.intents.validators import IntentValidator
from decision.errors import PolicyViolation
from decision.types import OrderSpec
from decision.execution_policy.marketable_limit import MarketableLimitPolicy
from decision.execution_policy.passive import PassivePolicy
from decision.risk_overlay.kill_conditions import BasicKillOverlay
from decision.rebalancing.threshold import ThresholdRebalance
from decision.rebalancing.schedule import (
    AlwaysRebalance,
    BarCountSchedule,
    TimeIntervalSchedule,
)


# ── stable_hash / dec_str / canonical_meta ───────────────────────────

class TestUtils:
    def test_stable_hash_deterministic(self):
        h1 = stable_hash(["a", "b"], prefix="dec")
        h2 = stable_hash(["a", "b"], prefix="dec")
        assert h1 == h2
        assert h1.startswith("dec-")

    def test_stable_hash_different_input(self):
        h1 = stable_hash(["a"], prefix="x")
        h2 = stable_hash(["b"], prefix="x")
        assert h1 != h2

    def test_dec_str_decimal(self):
        assert dec_str(Decimal("1.23000")) == "1.23000"

    def test_dec_str_non_decimal(self):
        assert dec_str(42) == "42"
        assert dec_str("hello") == "hello"

    def test_canonical_meta_stable_order(self):
        m = {"b": 2, "a": 1}
        assert canonical_meta(m) == "a=1|b=2"

    def test_canonical_meta_empty(self):
        assert canonical_meta(None) == ""
        assert canonical_meta({}) == ""


# ── DecisionStore ────────────────────────────────────────────────────

class TestDecisionStore:
    def test_append_and_iter(self, tmp_path):
        path = str(tmp_path / "decisions.jsonl")
        store = DecisionStore(path=path)
        store.append({"action": "buy", "symbol": "BTC"})
        store.append({"action": "sell", "symbol": "ETH"})

        records = list(store.iter_records())
        assert len(records) == 2
        assert records[0]["action"] == "buy"
        assert records[1]["symbol"] == "ETH"

    def test_iter_empty(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        store = DecisionStore(path=path)
        assert list(store.iter_records()) == []

    def test_none_path_noop(self):
        store = DecisionStore(path=None)
        store.append({"x": 1})  # should not raise
        assert list(store.iter_records()) == []

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "out.jsonl")
        store = DecisionStore(path=path)
        store.append({"test": True})
        records = list(store.iter_records())
        assert len(records) == 1


# ── IntentValidator ──────────────────────────────────────────────────

class TestIntentValidator:
    def _order(self, qty=1, price=None, side="buy"):
        return OrderSpec(
            order_id="o1", intent_id="i1", symbol="BTC",
            side=side, qty=Decimal(str(qty)), price=Decimal(str(price)) if price else None,
        )

    def test_valid_order(self):
        v = IntentValidator()
        v.validate(self._order(qty=1, price=100))

    def test_zero_qty_raises(self):
        v = IntentValidator()
        with pytest.raises(PolicyViolation, match="qty must be > 0"):
            v.validate(self._order(qty=0))

    def test_negative_qty_raises(self):
        v = IntentValidator()
        with pytest.raises(PolicyViolation, match="qty must be > 0"):
            v.validate(self._order(qty=-1))

    def test_below_min_qty(self):
        v = IntentValidator(min_qty=Decimal("5"))
        with pytest.raises(PolicyViolation, match="min_qty"):
            v.validate(self._order(qty=2))

    def test_below_min_notional(self):
        v = IntentValidator(min_notional=Decimal("100"))
        with pytest.raises(PolicyViolation, match="min_notional"):
            v.validate(self._order(qty=1, price=50))

    def test_zero_price_raises(self):
        v = IntentValidator()
        order = OrderSpec(
            order_id="o1", intent_id="i1", symbol="BTC",
            side="buy", qty=Decimal("1"), price=Decimal("0"),
        )
        with pytest.raises(PolicyViolation, match="price must be > 0"):
            v.validate(order)

    def test_price_hint_used(self):
        v = IntentValidator(min_notional=Decimal("100"))
        with pytest.raises(PolicyViolation, match="min_notional"):
            v.validate(self._order(qty=1), price_hint=Decimal("50"))


# ── MarketableLimitPolicy ───────────────────────────────────────────

class TestMarketableLimitPolicy:
    def _snap(self, close):
        market = SimpleNamespace(close=close)
        return SimpleNamespace(market=market)

    def _order(self, side="buy"):
        return OrderSpec(
            order_id="o1", intent_id="i1", symbol="BTC",
            side=side, qty=Decimal("1"),
        )

    def test_buy_price_above_close(self):
        pol = MarketableLimitPolicy(slippage_bps=Decimal("10"))
        snap = self._snap(100)
        result = pol.apply(snap, self._order("buy"))
        assert result.price > Decimal("100")
        assert result.order_type == "limit"

    def test_sell_price_below_close(self):
        pol = MarketableLimitPolicy(slippage_bps=Decimal("10"))
        snap = self._snap(100)
        result = pol.apply(snap, self._order("sell"))
        assert result.price < Decimal("100")


# ── PassivePolicy ────────────────────────────────────────────────────

class TestPassivePolicy:
    def _snap(self, close):
        market = SimpleNamespace(close=close)
        return SimpleNamespace(market=market)

    def _order(self, side="buy"):
        return OrderSpec(
            order_id="o1", intent_id="i1", symbol="BTC",
            side=side, qty=Decimal("1"),
        )

    def test_buy_price_below_close(self):
        pol = PassivePolicy(offset_bps=Decimal("5"))
        snap = self._snap(100)
        result = pol.apply(snap, self._order("buy"))
        assert result.price < Decimal("100")

    def test_sell_price_above_close(self):
        pol = PassivePolicy(offset_bps=Decimal("5"))
        snap = self._snap(100)
        result = pol.apply(snap, self._order("sell"))
        assert result.price > Decimal("100")


# ── BasicKillOverlay ─────────────────────────────────────────────────

class TestBasicKillOverlay:
    def test_halted(self):
        snap = SimpleNamespace(risk=SimpleNamespace(halted=True, blocked=False))
        ok, reasons = BasicKillOverlay().allow(snap)
        assert ok is False
        assert "risk_halted" in reasons

    def test_blocked(self):
        snap = SimpleNamespace(risk=SimpleNamespace(halted=False, blocked=True))
        ok, reasons = BasicKillOverlay().allow(snap)
        assert ok is False
        assert "risk_blocked" in reasons

    def test_allowed(self):
        snap = SimpleNamespace(risk=SimpleNamespace(halted=False, blocked=False))
        ok, reasons = BasicKillOverlay().allow(snap)
        assert ok is True
        assert reasons == ()

    def test_no_risk(self):
        snap = SimpleNamespace(risk=None)
        ok, reasons = BasicKillOverlay().allow(snap)
        assert ok is True


# ── ThresholdRebalance ───────────────────────────────────────────────

class TestThresholdRebalance:
    def test_empty_positions_triggers(self):
        snap = SimpleNamespace(positions={})
        tr = ThresholdRebalance()
        assert tr.should_rebalance(snap) is True

    def test_zero_total_triggers(self):
        snap = SimpleNamespace(positions={
            "BTC": SimpleNamespace(qty=Decimal("0")),
        })
        tr = ThresholdRebalance()
        assert tr.should_rebalance(snap) is True

    def test_balanced_no_trigger(self):
        snap = SimpleNamespace(positions={
            "BTC": SimpleNamespace(qty=Decimal("10")),
            "ETH": SimpleNamespace(qty=Decimal("10")),
        })
        tr = ThresholdRebalance(drift_pct=Decimal("0.05"))
        assert tr.should_rebalance(snap) is False

    def test_drifted_triggers(self):
        snap = SimpleNamespace(positions={
            "BTC": SimpleNamespace(qty=Decimal("90")),
            "ETH": SimpleNamespace(qty=Decimal("10")),
        })
        tr = ThresholdRebalance(drift_pct=Decimal("0.05"))
        assert tr.should_rebalance(snap) is True


# ── Schedule rebalancing ─────────────────────────────────────────────

class TestAlwaysRebalance:
    def test_always_true(self):
        sched = AlwaysRebalance()
        snap = SimpleNamespace()
        assert sched.should_rebalance(snap) is True
        assert sched.should_rebalance(snap) is True


class TestBarCountSchedule:
    def test_triggers_at_interval(self):
        sched = BarCountSchedule(interval=3)
        snap = SimpleNamespace()
        results = [sched.should_rebalance(snap) for _ in range(6)]
        assert results == [False, False, True, False, False, True]


class TestTimeIntervalSchedule:
    def test_triggers_after_interval(self):
        sched = TimeIntervalSchedule(interval=timedelta(hours=1))
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        snap0 = SimpleNamespace(ts=t0)
        assert sched.should_rebalance(snap0) is True  # first call

        snap1 = SimpleNamespace(ts=t0 + timedelta(minutes=30))
        assert sched.should_rebalance(snap1) is False

        snap2 = SimpleNamespace(ts=t0 + timedelta(hours=1))
        assert sched.should_rebalance(snap2) is True

    def test_no_ts_always_true(self):
        sched = TimeIntervalSchedule()
        snap = SimpleNamespace()
        assert sched.should_rebalance(snap) is True

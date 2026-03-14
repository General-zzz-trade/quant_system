# tests/unit/decision/test_rebalance_module.py
"""Tests for RebalanceModule — DecisionModule that rebalances towards target weights."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from decision.rebalancing.module import RebalanceModule


# ── Stubs ────────────────────────────────────────────────────────

def _market(close: str) -> SimpleNamespace:
    return SimpleNamespace(close=Decimal(close))


def _position(qty: str) -> SimpleNamespace:
    return SimpleNamespace(qty=Decimal(qty))


def _account(balance: str) -> SimpleNamespace:
    return SimpleNamespace(balance=Decimal(balance))


def _snapshot(
    *,
    markets: dict,
    positions: dict | None = None,
    balance: str = "100000",
    ts: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        markets=markets,
        positions=positions or {},
        account=_account(balance),
        ts=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ── Basic behavior ──────────────────────────────────────────────

class TestRebalanceModuleBasic:
    def test_no_targets_returns_empty(self):
        mod = RebalanceModule()
        snap = _snapshot(markets={"BTCUSDT": _market("40000")})
        assert list(mod.decide(snap)) == []

    def test_drift_below_threshold_no_intent(self):
        """2% drift with 5% threshold → no rebalance."""
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5, "ETHUSDT": 0.5},
            drift_threshold=0.05,
            min_rebalance_interval=timedelta(0),
        )
        snap = _snapshot(
            markets={"BTCUSDT": _market("40000"), "ETHUSDT": _market("3000")},
            positions={
                "BTCUSDT": _position("1.2"),  # 48000 / 100000 = 0.48 (drift=0.02)
                "ETHUSDT": _position("17.3"),  # 51900 / 100000 = 0.519 (drift=0.019)
            },
        )
        intents = list(mod.decide(snap))
        assert len(intents) == 0

    def test_drift_above_threshold_triggers_intent(self):
        """10% drift with 5% threshold → triggers rebalance."""
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5, "ETHUSDT": 0.5},
            drift_threshold=0.05,
            min_rebalance_interval=timedelta(0),
        )
        snap = _snapshot(
            markets={"BTCUSDT": _market("40000"), "ETHUSDT": _market("3000")},
            positions={
                "BTCUSDT": _position("1.0"),  # 40000 / 100000 = 0.40 (drift=0.10)
                "ETHUSDT": _position("20.0"),  # 60000 / 100000 = 0.60 (drift=0.10)
            },
        )
        intents = list(mod.decide(snap))
        assert len(intents) == 2

        # BTC needs to go from 0.40 → 0.50 (buy)
        btc_intent = [i for i in intents if i.symbol == "BTCUSDT"][0]
        assert btc_intent.side == "buy"
        assert btc_intent.reason_code == "rebalance"

        # ETH needs to go from 0.60 → 0.50 (sell)
        eth_intent = [i for i in intents if i.symbol == "ETHUSDT"][0]
        assert eth_intent.side == "sell"

    def test_empty_positions_triggers_full_allocation(self):
        """No positions → all targets trigger buy intents."""
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.6, "ETHUSDT": 0.4},
            drift_threshold=0.05,
            min_rebalance_interval=timedelta(0),
        )
        snap = _snapshot(
            markets={"BTCUSDT": _market("40000"), "ETHUSDT": _market("3000")},
        )
        intents = list(mod.decide(snap))
        assert len(intents) == 2
        assert all(i.side == "buy" for i in intents)


# ── Schedule gating ─────────────────────────────────────────────

class TestScheduleGating:
    def test_time_interval_blocks_frequent_rebalance(self):
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5},
            drift_threshold=0.01,
            min_rebalance_interval=timedelta(hours=1),
        )
        t0 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        snap = _snapshot(
            markets={"BTCUSDT": _market("40000")},
            ts=t0,
        )

        # First call: should trigger
        intents1 = list(mod.decide(snap))
        assert len(intents1) > 0

        # Second call 10 minutes later: blocked by interval
        snap2 = _snapshot(
            markets={"BTCUSDT": _market("40000")},
            ts=t0 + timedelta(minutes=10),
        )
        intents2 = list(mod.decide(snap2))
        assert len(intents2) == 0

        # Third call 2 hours later: allowed
        snap3 = _snapshot(
            markets={"BTCUSDT": _market("40000")},
            ts=t0 + timedelta(hours=2),
        )
        intents3 = list(mod.decide(snap3))
        assert len(intents3) > 0


# ── Target update ───────────────────────────────────────────────

class TestTargetUpdate:
    def test_set_targets_changes_behavior(self):
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5},
            drift_threshold=0.01,
            min_rebalance_interval=timedelta(0),
        )
        snap = _snapshot(markets={"BTCUSDT": _market("40000")})

        intents1 = list(mod.decide(snap))
        assert len(intents1) == 1
        assert intents1[0].side == "buy"

        # Change target to 0 → should generate sell
        mod.set_targets({"BTCUSDT": 0.0})
        snap2 = _snapshot(
            markets={"BTCUSDT": _market("40000")},
            positions={"BTCUSDT": _position("1.25")},
        )
        # 0.0 target but drift_threshold is 0.01, drift = |0 - 0.5| = 0.5 > 0.01
        # Actually current weight = 1.25 * 40000 / 100000 = 0.5, target = 0 → drift = 0.5
        intents2 = list(mod.decide(snap2))
        assert len(intents2) == 1
        assert intents2[0].side == "sell"


# ── Intent structure ────────────────────────────────────────────

class TestIntentStructure:
    def test_intent_has_required_fields(self):
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5},
            drift_threshold=0.01,
            min_rebalance_interval=timedelta(0),
            origin="test_rebalancer",
        )
        snap = _snapshot(markets={"BTCUSDT": _market("40000")})
        intents = list(mod.decide(snap))
        assert len(intents) == 1

        intent = intents[0]
        assert intent.symbol == "BTCUSDT"
        assert intent.side == "buy"
        assert intent.target_qty > 0
        assert intent.reason_code == "rebalance"
        assert intent.origin == "test_rebalancer"
        assert intent.intent_id.startswith("rebal-BTCUSDT-")

    def test_qty_calculation_correct(self):
        """Target 50% of 100k at price 40k → delta_qty = 1.25 BTC."""
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5},
            drift_threshold=0.01,
            min_rebalance_interval=timedelta(0),
        )
        snap = _snapshot(
            markets={"BTCUSDT": _market("40000")},
            balance="100000",
        )
        intents = list(mod.decide(snap))
        assert len(intents) == 1
        # target_qty = |0.5 * 100000 / 40000| = 1.25
        assert intents[0].target_qty == pytest.approx(Decimal("1.25"))


# ── Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_equity_returns_empty(self):
        mod = RebalanceModule(target_weights={"BTCUSDT": 0.5})
        snap = _snapshot(markets={"BTCUSDT": _market("40000")}, balance="0")
        assert list(mod.decide(snap)) == []

    def test_missing_market_price_skips_symbol(self):
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5, "ETHUSDT": 0.5},
            drift_threshold=0.01,
            min_rebalance_interval=timedelta(0),
        )
        snap = _snapshot(
            markets={"BTCUSDT": _market("40000")},  # ETH market missing
        )
        intents = list(mod.decide(snap))
        # Only BTC should generate intent
        assert all(i.symbol == "BTCUSDT" for i in intents)

    def test_no_account_returns_empty(self):
        mod = RebalanceModule(target_weights={"BTCUSDT": 0.5})
        snap = SimpleNamespace(
            markets={"BTCUSDT": _market("40000")},
            positions={},
            account=None,
            ts=None,
        )
        assert list(mod.decide(snap)) == []

    def test_supports_rust_state_inputs(self):
        rust = pytest.importorskip("_quant_hotpath")
        mod = RebalanceModule(
            target_weights={"BTCUSDT": 0.5},
            drift_threshold=0.01,
            min_rebalance_interval=timedelta(0),
        )
        snap = SimpleNamespace(
            markets={
                "BTCUSDT": rust.RustMarketState(
                    symbol="BTCUSDT",
                    close=4_000_000_000_000,
                    last_price=4_000_000_000_000,
                )
            },
            positions={
                "BTCUSDT": rust.RustPositionState(
                    symbol="BTCUSDT",
                    qty=100_000_000,
                    avg_price=4_000_000_000_000,
                    last_price=4_000_000_000_000,
                )
            },
            account=rust.RustAccountState.initial(currency="USDT", balance=10_000_000_000_000),
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        intents = list(mod.decide(snap))

        assert len(intents) == 1
        assert intents[0].side == "buy"

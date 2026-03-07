"""Tests for decision sub-modules — rebalancing, sizing, registry, risk overlay."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.snapshot import StateSnapshot

from decision.rebalancing.schedule import (
    AlwaysRebalance,
    BarCountSchedule,
    TimeIntervalSchedule,
)
from decision.rebalancing.threshold import ThresholdRebalance
from decision.registry import Registry, signal_registry
from decision.sizing.base import VolatilityAdjustedSizer
from decision.risk_overlay.base import AlwaysAllow, CompositeOverlay


def _make_snapshot(
    *,
    last_price: Decimal = Decimal("50000"),
    balance: Decimal = Decimal("10000"),
    positions: dict | None = None,
    ts: datetime | None = None,
) -> StateSnapshot:
    market = MarketState(symbol="BTCUSDT", last_price=last_price, close=last_price)
    account = AccountState.initial(currency="USDT", balance=balance)
    return StateSnapshot.of(
        symbol="BTCUSDT",
        ts=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
        event_id="e1",
        event_type="bar",
        bar_index=0,
        markets={"BTCUSDT": market},
        positions=positions or {},
        account=account,
    )


# ── Rebalancing Schedule ────────────────────────────────


class TestAlwaysRebalance:
    def test_always_true(self) -> None:
        sched = AlwaysRebalance()
        snap = _make_snapshot()
        assert sched.should_rebalance(snap) is True
        assert sched.should_rebalance(snap) is True


class TestBarCountSchedule:
    def test_triggers_every_n_bars(self) -> None:
        sched = BarCountSchedule(interval=3)
        snap = _make_snapshot()
        results = [sched.should_rebalance(snap) for _ in range(6)]
        # Should trigger at bar 3 and bar 6
        assert results == [False, False, True, False, False, True]


class TestTimeIntervalSchedule:
    def test_triggers_after_interval(self) -> None:
        sched = TimeIntervalSchedule(interval=timedelta(hours=1))
        t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

        # First call always triggers
        assert sched.should_rebalance(_make_snapshot(ts=t0)) is True
        # 30 min later: no trigger
        assert sched.should_rebalance(_make_snapshot(ts=t0 + timedelta(minutes=30))) is False
        # 60 min later: trigger
        assert sched.should_rebalance(_make_snapshot(ts=t0 + timedelta(minutes=60))) is True


# ── Threshold Rebalance ──────────────────────────────


class TestThresholdRebalance:
    def test_empty_positions_triggers(self) -> None:
        tr = ThresholdRebalance(drift_pct=Decimal("0.05"))
        snap = _make_snapshot(positions={})
        assert tr.should_rebalance(snap) is True

    def test_equal_weights_no_trigger(self) -> None:
        tr = ThresholdRebalance(drift_pct=Decimal("0.10"))
        positions = {
            "BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("10")),
            "ETHUSDT": PositionState(symbol="ETHUSDT", qty=Decimal("10")),
        }
        snap = _make_snapshot(positions=positions)
        assert tr.should_rebalance(snap) is False

    def test_drifted_weights_trigger(self) -> None:
        tr = ThresholdRebalance(drift_pct=Decimal("0.05"))
        positions = {
            "BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("90")),
            "ETHUSDT": PositionState(symbol="ETHUSDT", qty=Decimal("10")),
        }
        snap = _make_snapshot(positions=positions)
        assert tr.should_rebalance(snap) is True


# ── Registry ──────────────────────────────────────────


class TestRegistry:
    def test_register_and_build(self) -> None:
        reg = Registry()
        reg.register("test_signal", lambda: "signal_instance", category="signal")
        assert reg.build("test_signal") == "signal_instance"

    def test_duplicate_raises(self) -> None:
        reg = Registry()
        reg.register("sig", lambda: 1)
        with pytest.raises(KeyError, match="Already registered"):
            reg.register("sig", lambda: 2)

    def test_overwrite_allowed(self) -> None:
        reg = Registry()
        reg.register("sig", lambda: 1)
        reg.register("sig", lambda: 2, overwrite=True)
        assert reg.build("sig") == 2

    def test_list_names_by_category(self) -> None:
        reg = Registry()
        reg.register("a", lambda: 1, category="signal")
        reg.register("b", lambda: 2, category="allocator")
        reg.register("c", lambda: 3, category="signal")
        assert reg.list_names(category="signal") == ["a", "c"]
        assert reg.list_names(category="allocator") == ["b"]

    def test_has_and_len(self) -> None:
        reg = Registry()
        assert reg.has("x") is False
        reg.register("x", lambda: 1)
        assert reg.has("x") is True
        assert len(reg) == 1

    def test_unknown_build_raises(self) -> None:
        reg = Registry()
        with pytest.raises(KeyError, match="Unknown component"):
            reg.build("nonexistent")


# ── VolatilityAdjustedSizer ──────────────────────────


class TestVolatilityAdjustedSizer:
    def test_uses_default_volatility(self) -> None:
        """When no features, uses default volatility to produce a nonzero qty."""
        sizer = VolatilityAdjustedSizer(
            risk_fraction=Decimal("0.02"),
            volatility_key="atr",
            default_volatility=Decimal("0.02"),
        )
        snap = _make_snapshot(balance=Decimal("10000"), last_price=Decimal("50000"))
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        # target_qty = (10000 * 0.02 * 1) / (0.02 * 50000) = 200 / 1000 = 0.200
        assert qty == Decimal("0.200")

    def test_zero_price_returns_zero(self) -> None:
        sizer = VolatilityAdjustedSizer()
        snap = _make_snapshot(last_price=Decimal("0"))
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        assert qty == 0


# ── Risk Overlay ─────────────────────────────────────


class TestAlwaysAllow:
    def test_allows(self) -> None:
        snap = _make_snapshot()
        ok, reasons = AlwaysAllow().allow(snap)
        assert ok is True
        assert len(reasons) == 0


class TestCompositeOverlay:
    def test_all_allow(self) -> None:
        comp = CompositeOverlay(overlays=(AlwaysAllow(), AlwaysAllow()))
        snap = _make_snapshot()
        ok, reasons = comp.allow(snap)
        assert ok is True

    def test_empty_allows(self) -> None:
        comp = CompositeOverlay()
        snap = _make_snapshot()
        ok, reasons = comp.allow(snap)
        assert ok is True

# tests/contract/test_state_immutability_contract.py
"""Contract: state immutability — snapshots, events, and state types are frozen."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType

import pytest

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.snapshot import StateSnapshot


_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_snapshot(**overrides) -> StateSnapshot:
    defaults = dict(
        symbol="BTCUSDT",
        ts=_TS,
        event_id="evt-1",
        event_type="market",
        bar_index=0,
        markets={"BTCUSDT": MarketState(
            symbol="BTCUSDT", last_price=Decimal("100"),
            close=Decimal("100"), last_ts=_TS,
        )},
        positions={"BTCUSDT": PositionState(
            symbol="BTCUSDT", qty=Decimal("0"),
        )},
        account=AccountState.initial(currency="USDT", balance=Decimal("10000")),
    )
    defaults.update(overrides)
    return StateSnapshot.of(**defaults)


class TestSnapshotImmutability:
    def test_snapshot_is_frozen(self):
        snap = _make_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.symbol = "ETHUSDT"  # type: ignore[misc]

    def test_snapshot_markets_immutable(self):
        snap = _make_snapshot()
        assert isinstance(snap.markets, MappingProxyType)
        with pytest.raises(TypeError):
            snap.markets["NEW"] = None  # type: ignore[index]

    def test_snapshot_positions_immutable(self):
        snap = _make_snapshot()
        assert isinstance(snap.positions, MappingProxyType)
        with pytest.raises(TypeError):
            snap.positions["NEW"] = None  # type: ignore[index]

    def test_snapshot_of_returns_new_instance(self):
        s1 = _make_snapshot(bar_index=1)
        s2 = _make_snapshot(bar_index=2)
        assert s1 is not s2
        assert s1.bar_index != s2.bar_index


class TestAccountImmutability:
    def test_account_initial_frozen(self):
        acct = AccountState.initial(currency="USDT", balance=Decimal("10000"))
        with pytest.raises((AttributeError, TypeError)):
            acct.balance = Decimal("999")  # type: ignore[misc]

    def test_account_with_update_returns_new(self):
        acct = AccountState.initial(currency="USDT", balance=Decimal("10000"))
        updated = acct.with_update(
            balance=Decimal("9000"),
            margin_used=Decimal("100"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("-50"),
            fees_paid=Decimal("2"),
            ts=_TS,
        )
        assert updated is not acct
        assert updated.balance == Decimal("9000")
        assert acct.balance == Decimal("10000")


class TestMarketStateImmutability:
    def test_market_state_frozen(self):
        mkt = MarketState(
            symbol="BTCUSDT", last_price=Decimal("100"),
            close=Decimal("100"), last_ts=_TS,
        )
        with pytest.raises((AttributeError, TypeError)):
            mkt.last_price = Decimal("200")  # type: ignore[misc]


class TestPositionImmutability:
    def test_position_frozen(self):
        pos = PositionState(symbol="BTCUSDT", qty=Decimal("1"))
        with pytest.raises((AttributeError, TypeError)):
            pos.qty = Decimal("2")  # type: ignore[misc]


class TestDecisionDoesNotMutateState:
    def test_decision_module_receives_frozen_snapshot(self):
        """Decision modules get frozen snapshots — cannot mutate."""
        snap = _make_snapshot()
        # Simulate what a decision module does: read from snapshot
        price = snap.markets["BTCUSDT"].last_price
        assert price == Decimal("100")
        # Verify it cannot be changed
        with pytest.raises(TypeError):
            snap.markets["BTCUSDT"] = None  # type: ignore[index]

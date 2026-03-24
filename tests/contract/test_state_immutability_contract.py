# tests/contract/test_state_immutability_contract.py
"""Contract: state immutability — snapshots, events, and state types are frozen."""
from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from state import AccountState, MarketState, PositionState
from state.snapshot import StateSnapshot

_SCALE = 100_000_000
_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_snapshot(**overrides) -> StateSnapshot:
    defaults = dict(
        symbol="BTCUSDT",
        ts=_TS,
        event_id="evt-1",
        event_type="market",
        bar_index=0,
        markets={"BTCUSDT": MarketState(
            symbol="BTCUSDT", last_price=100 * _SCALE,
            close=100 * _SCALE, last_ts=_TS.isoformat(),
        )},
        positions={"BTCUSDT": PositionState(
            symbol="BTCUSDT", qty=0,
        )},
        account=AccountState.initial(currency="USDT", balance=10000 * _SCALE),
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
        acct = AccountState.initial(currency="USDT", balance=10000 * _SCALE)
        with pytest.raises((AttributeError, TypeError)):
            acct.balance = 999 * _SCALE  # type: ignore[misc]

    def test_account_with_update_returns_new(self):
        acct = AccountState.initial(currency="USDT", balance=10000 * _SCALE)
        updated = acct.with_update(
            balance=9000 * _SCALE,
            margin_used=100 * _SCALE,
            realized_pnl=0,
            unrealized_pnl=-50 * _SCALE,
            fees_paid=2 * _SCALE,
            ts=_TS.isoformat(),
        )
        assert updated is not acct
        assert updated.balance == 9000 * _SCALE
        assert acct.balance == 10000 * _SCALE


class TestMarketStateImmutability:
    def test_market_state_frozen(self):
        mkt = MarketState(
            symbol="BTCUSDT", last_price=100 * _SCALE,
            close=100 * _SCALE, last_ts=_TS.isoformat(),
        )
        with pytest.raises((AttributeError, TypeError)):
            mkt.last_price = 200 * _SCALE  # type: ignore[misc]


class TestPositionImmutability:
    def test_position_frozen(self):
        pos = PositionState(symbol="BTCUSDT", qty=1 * _SCALE)
        with pytest.raises((AttributeError, TypeError)):
            pos.qty = 2 * _SCALE  # type: ignore[misc]


class TestDecisionDoesNotMutateState:
    def test_decision_module_receives_frozen_snapshot(self):
        """Decision modules get frozen snapshots — cannot mutate."""
        snap = _make_snapshot()
        # Simulate what a decision module does: read from snapshot
        price = snap.markets["BTCUSDT"].last_price
        assert price == 100 * _SCALE
        # Verify it cannot be changed
        with pytest.raises(TypeError):
            snap.markets["BTCUSDT"] = None  # type: ignore[index]

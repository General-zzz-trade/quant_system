# tests/unit/engine/test_state_restoration.py
"""Tests for EngineCoordinator.restore_from_snapshot()."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from engine.coordinator import CoordinatorConfig, EngineCoordinator, EnginePhase
from state.snapshot import StateSnapshot
from state.market import MarketState
from state.account import AccountState
from state.position import PositionState


def _make_snapshot(
    symbol: str = "BTCUSDT",
    bar_index: int = 42,
    event_id: str = "evt-001",
) -> StateSnapshot:
    """Create a minimal StateSnapshot for testing."""
    market = MarketState(
        symbol=symbol,
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("103"),
        volume=Decimal("1000"),
        last_ts=datetime.now(timezone.utc),
    )
    account = AccountState(currency="USDT", balance=Decimal("10000"))
    position = PositionState(
        symbol=symbol,
        qty=Decimal("1.5"),
        avg_price=Decimal("100"),
    )
    return StateSnapshot.of(
        symbol=symbol,
        ts=datetime.now(timezone.utc),
        event_id=event_id,
        event_type="market",
        bar_index=bar_index,
        markets={symbol: market},
        positions={symbol: position},
        account=account,
    )


class TestRestoreFromSnapshot:
    def test_restore_during_init(self):
        coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT"))
        assert coord.phase == EnginePhase.INIT

        snapshot = _make_snapshot(bar_index=42, event_id="evt-001")
        coord.restore_from_snapshot(snapshot)

        view = coord.get_state_view()
        assert view["event_index"] == 42
        assert view["last_event_id"] == "evt-001"
        assert "BTCUSDT" in view["positions"]

    def test_restore_during_running_raises(self):
        coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT"))
        coord.start()
        assert coord.phase == EnginePhase.RUNNING

        snapshot = _make_snapshot()
        with pytest.raises(RuntimeError, match="INIT"):
            coord.restore_from_snapshot(snapshot)

    def test_restore_updates_markets(self):
        coord = EngineCoordinator(
            cfg=CoordinatorConfig(
                symbol_default="BTCUSDT",
                symbols=("BTCUSDT",),
            )
        )

        snapshot = _make_snapshot(symbol="BTCUSDT")
        coord.restore_from_snapshot(snapshot)

        view = coord.get_state_view()
        restored_market = view["markets"]["BTCUSDT"]
        assert restored_market.close == Decimal("103")

    def test_state_view_consistent_after_restore(self):
        coord = EngineCoordinator(
            cfg=CoordinatorConfig(
                symbol_default="BTCUSDT",
                symbols=("BTCUSDT",),
            )
        )

        snapshot = _make_snapshot(bar_index=100, event_id="evt-100")
        coord.restore_from_snapshot(snapshot)

        view = coord.get_state_view()
        assert view["event_index"] == 100
        assert view["last_event_id"] == "evt-100"
        assert view["phase"] == "init"

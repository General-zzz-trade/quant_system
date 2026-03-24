# tests/unit/engine/test_state_restoration.py
"""Tests for EngineCoordinator.restore_from_snapshot()."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engine.coordinator import CoordinatorConfig, EngineCoordinator, EnginePhase
from state.snapshot import StateSnapshot
from state import MarketState, AccountState, PortfolioState, PositionState, RiskState
from _quant_hotpath import RustStateStore

_SCALE = 100_000_000


def _make_snapshot(
    symbol: str = "BTCUSDT",
    bar_index: int = 42,
    event_id: str = "evt-001",
) -> StateSnapshot:
    """Create a minimal StateSnapshot for testing."""
    ts_str = datetime.now(timezone.utc).isoformat()
    market = MarketState(
        symbol=symbol,
        open=100 * _SCALE,
        high=105 * _SCALE,
        low=99 * _SCALE,
        close=103 * _SCALE,
        volume=1000 * _SCALE,
        last_ts=ts_str,
    )
    account = AccountState(currency="USDT", balance=10000 * _SCALE)
    position = PositionState(
        symbol=symbol,
        qty=150000000,  # 1.5
        avg_price=100 * _SCALE,
    )
    portfolio = PortfolioState(
        total_equity="10004.5",
        cash_balance="10000",
        realized_pnl="0",
        unrealized_pnl="4.5",
        fees_paid="0",
        gross_exposure="154.5",
        net_exposure="154.5",
        leverage="0.01545",
        margin_used="0",
        margin_available="10000",
        margin_ratio=None,
        symbols=[symbol],
        last_ts=ts_str,
    )
    risk = RiskState(
        blocked=False,
        halted=False,
        level="ok",
        message=None,
        flags=[],
        equity_peak="10004.5",
        drawdown_pct="0",
        last_ts=ts_str,
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
        portfolio=portfolio,
        risk=risk,
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
        assert view["portfolio"] is not None
        assert view["risk"] is not None

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
        assert restored_market.close == 103 * _SCALE

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

    def test_restore_updates_store_when_present(self):
        store = RustStateStore(["BTCUSDT"], "USDT", 1)
        coord = EngineCoordinator(
            cfg=CoordinatorConfig(symbol_default="BTCUSDT", symbols=("BTCUSDT",)),
            store=store,
        )

        snapshot = _make_snapshot(bar_index=7, event_id="evt-store")
        coord.restore_from_snapshot(snapshot)

        bundle = dict(store.export_state())
        assert int(bundle["event_index"]) == 7
        assert bundle["last_event_id"] == "evt-store"
        assert dict(bundle["markets"])["BTCUSDT"].close_f == pytest.approx(103.0)

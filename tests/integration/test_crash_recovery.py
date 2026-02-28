"""Integration: coordinator-level crash recovery via SqliteStateStore.

Tests that a coordinator's state can be saved to SQLite and fully restored
in a new coordinator instance, preserving markets, positions, and bar_index.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from state.store import SqliteStateStore


def _market(symbol: str, close: float, idx: int) -> SimpleNamespace:
    ts = datetime(2024, 1, 1, idx // 60, idx % 60)
    return SimpleNamespace(
        event_type="MARKET",
        symbol=symbol,
        open=close, high=close + 1, low=close - 1, close=close,
        volume=50.0,
        ts=ts,
        header=SimpleNamespace(event_id=f"e{idx}", ts=ts),
    )


def _build_coordinator(symbols: tuple[str, ...] = ("BTCUSDT",)) -> EngineCoordinator:
    cfg = CoordinatorConfig(
        symbol_default=symbols[0], symbols=symbols, currency="USDT",
    )
    return EngineCoordinator(cfg=cfg)


class TestCrashRecovery:

    def test_save_and_restore_via_sqlite(self, tmp_path):
        """Coordinator → snapshot → SQLite save → new coordinator restore → state matches."""
        coord = _build_coordinator()
        coord.start()

        for i in range(5):
            coord.emit(_market("BTCUSDT", 40000.0 + i * 100, i), actor="test")

        view = coord.get_state_view()
        snap = view["last_snapshot"]
        assert snap is not None

        db_path = tmp_path / "state.db"
        store = SqliteStateStore(path=db_path)
        store.save(snap)

        # New coordinator restores from checkpoint
        coord2 = _build_coordinator()
        checkpoint = store.latest("BTCUSDT")
        assert checkpoint is not None
        coord2.restore_from_snapshot(checkpoint.snapshot)
        coord2.start()

        view2 = coord2.get_state_view()

        assert view2["event_index"] == view["event_index"]
        assert view2["markets"]["BTCUSDT"].close == view["markets"]["BTCUSDT"].close
        assert view2["account"].currency == view["account"].currency

        store.close()

    def test_multi_symbol_restore(self, tmp_path):
        """Multi-symbol save/restore preserves all symbol MarketState."""
        symbols = ("BTCUSDT", "ETHUSDT")
        coord = _build_coordinator(symbols)
        coord.start()

        for i in range(5):
            coord.emit(_market("BTCUSDT", 40000.0 + i * 100, i), actor="test")
            coord.emit(_market("ETHUSDT", 3000.0 + i * 10, 100 + i), actor="test")

        snap = coord.get_state_view()["last_snapshot"]
        assert snap is not None

        db_path = tmp_path / "state.db"
        store = SqliteStateStore(path=db_path)
        store.save(snap)

        coord2 = _build_coordinator(symbols)
        checkpoint = store.latest(snap.symbol)
        assert checkpoint is not None
        coord2.restore_from_snapshot(checkpoint.snapshot)
        coord2.start()

        view2 = coord2.get_state_view()
        assert "BTCUSDT" in view2["markets"]
        assert "ETHUSDT" in view2["markets"]
        assert view2["markets"]["BTCUSDT"].close == coord.get_state_view()["markets"]["BTCUSDT"].close
        assert view2["markets"]["ETHUSDT"].close == coord.get_state_view()["markets"]["ETHUSDT"].close

        store.close()

    def test_bar_index_continuity(self, tmp_path):
        """After restore, event_index continues from the saved value."""
        coord = _build_coordinator()
        coord.start()

        for i in range(7):
            coord.emit(_market("BTCUSDT", 40000.0 + i, i), actor="test")

        saved_index = coord.get_state_view()["event_index"]
        assert saved_index == 7

        snap = coord.get_state_view()["last_snapshot"]
        db_path = tmp_path / "state.db"
        store = SqliteStateStore(path=db_path)
        store.save(snap)

        coord2 = _build_coordinator()
        checkpoint = store.latest("BTCUSDT")
        coord2.restore_from_snapshot(checkpoint.snapshot)
        coord2.start()

        # Continue emitting — index should resume from saved value
        for i in range(3):
            coord2.emit(_market("BTCUSDT", 50000.0 + i, 100 + i), actor="test")

        assert coord2.get_state_view()["event_index"] == saved_index + 3

        store.close()

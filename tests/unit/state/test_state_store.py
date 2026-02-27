# tests/unit/state/test_state_store.py
"""Tests for InMemoryStateStore and SqliteStateStore."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.portfolio import PortfolioState
from state.risk import RiskState
from state.snapshot import StateSnapshot
from state.store import InMemoryStateStore, SqliteStateStore


def _make_snapshot(
    symbol: str = "BTCUSDT",
    bar_index: int = 42,
    balance: Decimal = Decimal("10000"),
    qty: Decimal = Decimal("0.5"),
) -> StateSnapshot:
    ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return StateSnapshot.of(
        symbol=symbol,
        ts=ts,
        event_id="evt-001",
        event_type="fill",
        bar_index=bar_index,
        markets={symbol: MarketState(
            symbol=symbol,
            last_price=Decimal("42000"),
            open=Decimal("41500"),
            high=Decimal("42500"),
            low=Decimal("41000"),
            close=Decimal("42000"),
            volume=Decimal("1234.56"),
            last_ts=ts,
        )},
        positions={
            symbol: PositionState(
                symbol=symbol,
                qty=qty,
                avg_price=Decimal("41000"),
                last_price=Decimal("42000"),
                last_ts=ts,
            ),
        },
        account=AccountState(
            currency="USDT",
            balance=balance,
            margin_used=Decimal("2100"),
            margin_available=balance - Decimal("2100"),
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("500"),
            fees_paid=Decimal("12.5"),
            last_ts=ts,
        ),
        portfolio=PortfolioState(
            total_equity=balance + Decimal("500"),
            cash_balance=balance,
            realized_pnl=Decimal("500"),
            unrealized_pnl=Decimal("500"),
            fees_paid=Decimal("12.5"),
            gross_exposure=Decimal("21000"),
            net_exposure=Decimal("21000"),
            leverage=Decimal("2"),
            margin_used=Decimal("2100"),
            margin_available=balance - Decimal("2100"),
            margin_ratio=None,
            symbols=(symbol,),
            last_ts=ts,
        ),
        risk=RiskState(
            blocked=False,
            halted=False,
            level="normal",
            message=None,
            flags=("within_limits",),
            equity_peak=Decimal("10500"),
            drawdown_pct=Decimal("0.0"),
            last_ts=ts,
        ),
    )


class TestInMemoryStateStore:
    def test_save_and_latest(self) -> None:
        store = InMemoryStateStore()
        snap = _make_snapshot()
        store.save(snap)
        cp = store.latest("BTCUSDT")
        assert cp is not None
        assert cp.symbol == "BTCUSDT"
        assert cp.bar_index == 42
        assert cp.snapshot.account.balance == Decimal("10000")

    def test_latest_returns_none_for_unknown(self) -> None:
        store = InMemoryStateStore()
        assert store.latest("ETHUSDT") is None

    def test_upsert_replaces_older(self) -> None:
        store = InMemoryStateStore()
        store.save(_make_snapshot(bar_index=1))
        store.save(_make_snapshot(bar_index=2))
        cp = store.latest("BTCUSDT")
        assert cp is not None
        assert cp.bar_index == 2


class TestSqliteStateStore:
    def test_save_and_latest(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        with SqliteStateStore(db) as store:
            snap = _make_snapshot()
            store.save(snap)
            cp = store.latest("BTCUSDT")
            assert cp is not None
            assert cp.symbol == "BTCUSDT"
            assert cp.bar_index == 42
            assert cp.snapshot.account.balance == Decimal("10000")
            assert cp.snapshot.market.last_price == Decimal("42000")

    def test_latest_returns_none_for_unknown(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        with SqliteStateStore(db) as store:
            assert store.latest("ETHUSDT") is None

    def test_upsert_replaces_older(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        with SqliteStateStore(db) as store:
            store.save(_make_snapshot(bar_index=1))
            store.save(_make_snapshot(bar_index=2))
            cp = store.latest("BTCUSDT")
            assert cp is not None
            assert cp.bar_index == 2

    def test_persistence_across_connections(self, tmp_path: Path) -> None:
        """Simulate crash recovery: write, close, reopen, verify."""
        db = tmp_path / "state.db"

        with SqliteStateStore(db) as store:
            store.save(_make_snapshot(balance=Decimal("25000")))

        with SqliteStateStore(db) as store:
            cp = store.latest("BTCUSDT")
            assert cp is not None
            assert cp.snapshot.account.balance == Decimal("25000")

    def test_multi_symbol(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        with SqliteStateStore(db) as store:
            store.save(_make_snapshot(symbol="BTCUSDT"))
            store.save(_make_snapshot(symbol="ETHUSDT", balance=Decimal("5000")))

            btc = store.latest("BTCUSDT")
            eth = store.latest("ETHUSDT")
            assert btc is not None and btc.snapshot.account.balance == Decimal("10000")
            assert eth is not None and eth.snapshot.account.balance == Decimal("5000")
            assert sorted(store.all_symbols()) == ["BTCUSDT", "ETHUSDT"]

    def test_position_roundtrip(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        snap = _make_snapshot(qty=Decimal("-1.23456789"))
        with SqliteStateStore(db) as store:
            store.save(snap)
            cp = store.latest("BTCUSDT")
            assert cp is not None
            pos = cp.snapshot.positions.get("BTCUSDT")
            assert pos is not None
            assert pos.qty == Decimal("-1.23456789")

    def test_portfolio_and_risk_roundtrip(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        snap = _make_snapshot()
        with SqliteStateStore(db) as store:
            store.save(snap)
            cp = store.latest("BTCUSDT")
            assert cp is not None
            assert cp.snapshot.portfolio is not None
            assert cp.snapshot.portfolio.leverage == Decimal("2")
            assert cp.snapshot.risk is not None
            assert cp.snapshot.risk.flags == ("within_limits",)

    def test_history_mode(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        with SqliteStateStore(db, keep_history=True) as store:
            store.save(_make_snapshot(bar_index=1))
            store.save(_make_snapshot(bar_index=2))
            store.save(_make_snapshot(bar_index=3))

            # Latest should be bar_index=3
            cp = store.latest("BTCUSDT")
            assert cp is not None
            assert cp.bar_index == 3

            # History should have all 3
            rows = store._conn.execute(
                "SELECT bar_index FROM checkpoint_history ORDER BY id"
            ).fetchall()
            assert [r[0] for r in rows] == [1, 2, 3]

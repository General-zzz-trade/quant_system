"""Tests for state.store — InMemoryStateStore and SqliteStateStore."""
from __future__ import annotations

from datetime import datetime, timezone

from state import MarketState, PositionState, AccountState
from state.snapshot import StateSnapshot
from state.store import InMemoryStateStore, SqliteStateStore

_SCALE = 100_000_000


def _make_snapshot(
    symbol: str = "ETHUSDT",
    bar_index: int = 0,
    balance: int = 10000 * _SCALE,
    event_type: str = "bar",
    ts: datetime | None = None,
) -> StateSnapshot:
    ts = ts or datetime(2024, 1, 1, tzinfo=timezone.utc)
    account = AccountState.initial(currency="USDT", balance=balance)
    market = MarketState(
        symbol=symbol, last_price=3500 * _SCALE,
        open=3400 * _SCALE, high=3600 * _SCALE,
        low=3300 * _SCALE, close=3500 * _SCALE,
        volume=1000 * _SCALE, last_ts=ts.isoformat(),
    )
    position = PositionState.empty(symbol)
    return StateSnapshot.of(
        symbol=symbol, ts=ts, event_id=f"ev-{bar_index}",
        event_type=event_type, bar_index=bar_index,
        markets={symbol: market},
        positions={symbol: position},
        account=account,
    )


class TestInMemoryStateStore:
    def test_latest_empty(self):
        store = InMemoryStateStore()
        assert store.latest("ETHUSDT") is None

    def test_save_and_latest(self):
        store = InMemoryStateStore()
        snap = _make_snapshot()
        store.save(snap)
        cp = store.latest("ETHUSDT")
        assert cp is not None
        assert cp.symbol == "ETHUSDT"
        assert cp.snapshot is snap

    def test_save_overwrites(self):
        store = InMemoryStateStore()
        store.save(_make_snapshot(bar_index=0))
        store.save(_make_snapshot(bar_index=5))
        cp = store.latest("ETHUSDT")
        assert cp is not None
        assert cp.bar_index == 5

    def test_multi_symbol_isolation(self):
        store = InMemoryStateStore()
        store.save(_make_snapshot(symbol="ETHUSDT"))
        store.save(_make_snapshot(symbol="BTCUSDT"))
        assert store.latest("ETHUSDT") is not None
        assert store.latest("BTCUSDT") is not None
        assert store.latest("SOLUSDT") is None

    def test_checkpoint_ts_and_event_id(self):
        store = InMemoryStateStore()
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        store.save(_make_snapshot(bar_index=42, ts=ts))
        cp = store.latest("ETHUSDT")
        assert cp is not None
        assert cp.ts == ts
        assert cp.event_id == "ev-42"


class TestSqliteStateStore:
    def test_save_and_latest(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            snap = _make_snapshot()
            store.save(snap)
            cp = store.latest("ETHUSDT")
            assert cp is not None
            assert cp.symbol == "ETHUSDT"
            assert cp.bar_index == 0

    def test_latest_empty(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            assert store.latest("ETHUSDT") is None

    def test_all_symbols(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            store.save(_make_snapshot(symbol="ETHUSDT"))
            store.save(_make_snapshot(symbol="BTCUSDT"))
            symbols = store.all_symbols()
            assert sorted(symbols) == ["BTCUSDT", "ETHUSDT"]

    def test_upsert_semantics(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            store.save(_make_snapshot(bar_index=1))
            store.save(_make_snapshot(bar_index=10))
            cp = store.latest("ETHUSDT")
            assert cp is not None
            assert cp.bar_index == 10

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            store.save(_make_snapshot())
        # After close, creating new store should see persisted data
        with SqliteStateStore(db_path) as store2:
            cp = store2.latest("ETHUSDT")
            assert cp is not None

    def test_round_trip_preserves_values(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            snap = _make_snapshot(balance=1234567800000)  # 12345.678 * _SCALE
            store.save(snap)
            cp = store.latest("ETHUSDT")
            assert cp is not None
            assert cp.snapshot.account.balance == 1234567800000

    def test_history_table_accumulation(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path, keep_history=True) as store:
            store.save(_make_snapshot(bar_index=1))
            store.save(_make_snapshot(bar_index=2))
            store.save(_make_snapshot(bar_index=3))
            # Latest should be bar_index=3
            cp = store.latest("ETHUSDT")
            assert cp is not None
            assert cp.bar_index == 3
            # History table should have 3 rows
            rows = store._conn.execute(
                "SELECT COUNT(*) FROM checkpoint_history WHERE symbol = ?",
                ("ETHUSDT",),
            ).fetchone()
            assert rows[0] == 3

    def test_multi_symbol_round_trip(self, tmp_path):
        db_path = tmp_path / "state.db"
        with SqliteStateStore(db_path) as store:
            store.save(_make_snapshot(symbol="ETHUSDT", bar_index=5))
            store.save(_make_snapshot(symbol="BTCUSDT", bar_index=10))
            eth = store.latest("ETHUSDT")
            btc = store.latest("BTCUSDT")
            assert eth is not None and eth.bar_index == 5
            assert btc is not None and btc.bar_index == 10

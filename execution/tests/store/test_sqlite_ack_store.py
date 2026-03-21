from __future__ import annotations

from types import SimpleNamespace

from execution.bridge.execution_bridge import ExecutionBridge
from execution.store.ack_store import SQLiteAckStore


class _Venue:
    def __init__(self) -> None:
        self.n = 0

    def submit_order(self, cmd):
        self.n += 1
        return {"ok": True, "n": self.n}

    def cancel_order(self, cmd):
        self.n += 1
        return {"ok": True, "n": self.n}


def test_sqlite_ack_store_persists_dedup_across_restart(tmp_path) -> None:
    db = tmp_path / "acks.sqlite"

    venue = _Venue()
    store1 = SQLiteAckStore(path=str(db))
    br1 = ExecutionBridge(venue_clients={"binance": venue}, ack_store=store1)

    cmd = SimpleNamespace(
        venue="binance",
        symbol="BTCUSDT",
        command_id="c1",
        idempotency_key="idem-1",
        action="submit",
    )

    a1 = br1.submit(cmd)
    assert a1.ok
    assert venue.n == 1

    # restart: new bridge + new venue instance (simulating new process)
    venue2 = _Venue()
    store2 = SQLiteAckStore(path=str(db))
    br2 = ExecutionBridge(venue_clients={"binance": venue2}, ack_store=store2)

    a2 = br2.submit(cmd)
    assert a2.deduped is True
    assert venue2.n == 0  # no resubmit


def test_sqlite_ack_store_context_manager_closes_connection(tmp_path) -> None:
    db = tmp_path / "acks.sqlite"

    with SQLiteAckStore(path=str(db)) as store:
        store.put("idem-1", {"ok": True})
        assert store._closed is False

    assert store._closed is True

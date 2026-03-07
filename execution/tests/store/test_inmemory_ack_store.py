from __future__ import annotations

import execution.store.ack_store as ack_store_module
from execution.store.ack_store import InMemoryAckStore


def test_inmemory_ack_store_round_trips_payload(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(ack_store_module.time, "time", lambda: now)

    store = InMemoryAckStore()
    store.put("idem-1", {"status": "ACCEPTED", "attempts": 1, "deduped": False})

    assert store.get("idem-1") == {
        "status": "ACCEPTED",
        "attempts": 1,
        "deduped": False,
    }


def test_inmemory_ack_store_honors_ttl(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(ack_store_module.time, "time", lambda: now)

    store = InMemoryAckStore(ttl_sec=5.0)
    store.put("idem-1", {"status": "ACCEPTED"})
    assert store.get("idem-1") == {"status": "ACCEPTED"}

    now = 1006.0
    assert store.get("idem-1") is None


def test_inmemory_ack_store_prune(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(ack_store_module.time, "time", lambda: now)

    store = InMemoryAckStore(ttl_sec=5.0)
    store.put("idem-1", {"status": "ACCEPTED"})
    store.put("idem-2", {"status": "FAILED"})

    now = 1006.0
    assert store.prune() == 2
    assert store.get("idem-1") is None
    assert store.get("idem-2") is None

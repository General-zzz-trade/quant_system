from __future__ import annotations

import execution.store.dedup_store as dedup_store_module
from execution.store.dedup_store import InMemoryDedupStore


def test_inmemory_dedup_store_round_trips_digest(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(dedup_store_module.time, "time", lambda: now)

    store = InMemoryDedupStore()
    store.put("fill-1", "digest-1")

    assert store.get("fill-1") == "digest-1"


def test_inmemory_dedup_store_overwrites_digest(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(dedup_store_module.time, "time", lambda: now)

    store = InMemoryDedupStore()
    store.put("fill-1", "digest-1")
    store.put("fill-1", "digest-2")

    assert store.get("fill-1") == "digest-2"


def test_inmemory_dedup_store_honors_ttl(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(dedup_store_module.time, "time", lambda: now)

    store = InMemoryDedupStore(ttl_sec=5.0)
    store.put("fill-1", "digest-1")
    assert store.get("fill-1") == "digest-1"

    now = 1006.0
    assert store.get("fill-1") is None


def test_inmemory_dedup_store_prune(monkeypatch) -> None:
    now = 1000.0
    monkeypatch.setattr(dedup_store_module.time, "time", lambda: now)

    store = InMemoryDedupStore(ttl_sec=5.0)
    store.put("fill-1", "digest-1")
    store.put("fill-2", "digest-2")

    now = 1006.0
    assert store.prune() == 2

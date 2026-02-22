from __future__ import annotations

from execution.store.dedup_store import SQLiteDedupStore


def test_sqlite_dedup_store_persists_across_restart(tmp_path):
    p = tmp_path / "dedup.db"
    s1 = SQLiteDedupStore(path=str(p))
    s1.put("k1", "d1")
    assert s1.get("k1") == "d1"
    s1.close()

    s2 = SQLiteDedupStore(path=str(p))
    assert s2.get("k1") == "d1"
    s2.close()


def test_sqlite_dedup_store_does_not_overwrite_digest(tmp_path):
    p = tmp_path / "dedup.db"
    s = SQLiteDedupStore(path=str(p))
    s.put("k1", "d1")
    s.put("k1", "d2")  # should not overwrite the original digest
    assert s.get("k1") == "d1"
    s.close()

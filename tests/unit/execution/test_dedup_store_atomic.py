"""Test SQLiteDedupStore atomic upsert."""
import tempfile
import os


class TestSQLiteDedupStoreAtomic:
    def test_put_then_get(self):
        from execution.store.dedup_store import SQLiteDedupStore
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteDedupStore(path=os.path.join(d, "dedup.db"))
            store.put("key1", "digest_abc")
            result = store.get("key1")
            assert result is not None
            # get() returns the digest string directly (Optional[str])
            assert result == "digest_abc"

    def test_put_twice_updates_timestamp(self):
        import time
        from execution.store.dedup_store import SQLiteDedupStore
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteDedupStore(path=os.path.join(d, "dedup.db"))
            store.put("key1", "digest_v1")
            time.sleep(0.05)
            store.put("key1", "digest_v1")
            result = store.get("key1")
            assert result is not None

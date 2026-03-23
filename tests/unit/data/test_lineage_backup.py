"""Tests for data lineage tracking and backup management."""
from __future__ import annotations



from data.lineage import LineageTracker
from data.backup import BackupManager


class TestLineageTracker:
    def test_record_and_get(self, tmp_path):
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        rec = tracker.record(
            artifact_id="BTCUSDT_1h",
            source="binance_api",
            row_count=1000,
            time_range_start="2024-01-01T00:00:00Z",
            time_range_end="2024-02-11T00:00:00Z",
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        assert rec.artifact_id == "BTCUSDT_1h"
        assert rec.source == "binance_api"
        assert rec.row_count == 1000
        assert len(rec.version) == 12
        assert len(rec.schema_hash) == 12

        fetched = tracker.get("BTCUSDT_1h")
        assert fetched is not None
        assert fetched.version == rec.version

    def test_persistence(self, tmp_path):
        path = tmp_path / "lineage.jsonl"
        tracker1 = LineageTracker(path)
        tracker1.record("BTCUSDT_1h", "binance", 100, "2024-01", "2024-02", ["ts", "close"])
        tracker1.record("ETHUSDT_1h", "binance", 200, "2024-01", "2024-02", ["ts", "close"])

        # Reload from disk
        tracker2 = LineageTracker(path)
        assert tracker2.get("BTCUSDT_1h") is not None
        assert tracker2.get("ETHUSDT_1h") is not None
        assert len(tracker2.list_all()) == 2

    def test_trace_lineage(self, tmp_path):
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        tracker.record("raw_btc", "binance", 1000, "2024-01", "2024-02", ["ts", "close"])
        tracker.record("clean_btc", "pipeline", 990, "2024-01", "2024-02",
                       ["ts", "close"], parent_artifacts=["raw_btc"])
        tracker.record("features_btc", "feature_eng", 990, "2024-01", "2024-02",
                       ["ts", "sma", "rsi"], parent_artifacts=["clean_btc"])

        chain = tracker.trace("features_btc")
        assert len(chain) == 3
        ids = {r.artifact_id for r in chain}
        assert ids == {"features_btc", "clean_btc", "raw_btc"}

    def test_processing_steps(self, tmp_path):
        tracker = LineageTracker(tmp_path / "lineage.jsonl")
        rec = tracker.record(
            "cleaned_data", "pipeline", 500, "2024-01", "2024-02",
            ["ts", "close"],
            processing_steps=["dedup", "gap_fill", "anomaly_remove"],
        )
        assert rec.processing_steps == ("dedup", "gap_fill", "anomaly_remove")


class TestBackupManager:
    def test_create_snapshot(self, tmp_path):
        source = tmp_path / "data"
        source.mkdir()
        (source / "BTC.parquet").write_text("data")
        (source / "ETH.parquet").write_text("data")
        (source / "notes.txt").write_text("not backed up")

        backup = BackupManager(source, tmp_path / "backups", file_patterns=["*.parquet"])
        snap = backup.create_snapshot("test")
        assert snap is not None
        assert snap.exists()
        assert (snap / "BTC.parquet").exists()
        assert (snap / "ETH.parquet").exists()
        assert not (snap / "notes.txt").exists()

    def test_list_snapshots(self, tmp_path):
        source = tmp_path / "data"
        source.mkdir()
        (source / "test.csv").write_text("data")

        backup = BackupManager(source, tmp_path / "backups", file_patterns=["*.csv"])
        backup.create_snapshot("a")
        backup.create_snapshot("b")
        snapshots = backup.list_snapshots()
        assert len(snapshots) == 2

    def test_cleanup_old(self, tmp_path):
        source = tmp_path / "data"
        source.mkdir()
        (source / "test.csv").write_text("data")

        backup = BackupManager(source, tmp_path / "backups", max_snapshots=2, file_patterns=["*.csv"])
        for i in range(5):
            backup.create_snapshot(f"snap{i}")
        removed = backup.cleanup_old()
        assert removed == 3
        assert len(backup.list_snapshots()) == 2

    def test_restore(self, tmp_path):
        source = tmp_path / "data"
        source.mkdir()
        (source / "BTC.csv").write_text("original")

        backup = BackupManager(source, tmp_path / "backups", file_patterns=["*.csv"])
        snap = backup.create_snapshot()

        # Corrupt source
        (source / "BTC.csv").write_text("corrupted")

        # Restore
        count = backup.restore_from(snap)
        assert count == 1
        assert (source / "BTC.csv").read_text() == "original"

    def test_empty_source(self, tmp_path):
        source = tmp_path / "data"
        source.mkdir()
        backup = BackupManager(source, tmp_path / "backups")
        assert backup.create_snapshot() is None


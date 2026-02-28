"""Data backup strategy — periodic parquet snapshots with retention policy."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages periodic data backups with configurable retention.

    Creates timestamped snapshot copies of data directories.
    Optionally syncs to S3/OSS (when boto3/oss2 are available).

    Usage:
        backup = BackupManager(
            source_dir="/app/data/live",
            backup_dir="/app/backups",
            max_snapshots=7,
        )
        backup.create_snapshot()
        backup.cleanup_old()
    """

    def __init__(
        self,
        source_dir: str | Path,
        backup_dir: str | Path,
        max_snapshots: int = 7,
        file_patterns: Sequence[str] = ("*.parquet", "*.csv"),
    ) -> None:
        self._source = Path(source_dir)
        self._backup_root = Path(backup_dir)
        self._max_snapshots = max_snapshots
        self._patterns = list(file_patterns)

    def create_snapshot(self, tag: str = "") -> Optional[Path]:
        """Create a timestamped snapshot of the source directory.

        Returns the snapshot directory path, or None if source is empty.
        """
        if not self._source.exists():
            logger.warning("Source directory does not exist: %s", self._source)
            return None

        # Collect files matching patterns
        files_to_backup: list[Path] = []
        for pattern in self._patterns:
            files_to_backup.extend(self._source.rglob(pattern))

        if not files_to_backup:
            logger.info("No files matching patterns in %s", self._source)
            return None

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        snapshot_dir = self._backup_root / f"snapshot_{ts_str}{suffix}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for f in files_to_backup:
            rel = f.relative_to(self._source)
            dest = snapshot_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

        logger.info(
            "Snapshot created: %s (%d files)", snapshot_dir, len(files_to_backup)
        )
        return snapshot_dir

    def list_snapshots(self) -> list[Path]:
        """List all existing snapshots sorted by creation time (newest first)."""
        if not self._backup_root.exists():
            return []
        snapshots = sorted(
            [d for d in self._backup_root.iterdir() if d.is_dir() and d.name.startswith("snapshot_")],
            key=lambda d: d.name,
            reverse=True,
        )
        return snapshots

    def cleanup_old(self) -> int:
        """Remove snapshots exceeding max_snapshots retention. Returns count removed."""
        snapshots = self.list_snapshots()
        to_remove = snapshots[self._max_snapshots:]
        for d in to_remove:
            shutil.rmtree(d)
            logger.info("Removed old snapshot: %s", d)
        return len(to_remove)

    def latest_snapshot(self) -> Optional[Path]:
        """Get the most recent snapshot directory."""
        snapshots = self.list_snapshots()
        return snapshots[0] if snapshots else None

    def restore_from(self, snapshot_dir: Path) -> int:
        """Restore files from a snapshot back to the source directory.

        Returns number of files restored.
        """
        if not snapshot_dir.exists():
            logger.error("Snapshot does not exist: %s", snapshot_dir)
            return 0

        count = 0
        for pattern in self._patterns:
            for f in snapshot_dir.rglob(pattern):
                rel = f.relative_to(snapshot_dir)
                dest = self._source / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                count += 1

        logger.info("Restored %d files from %s", count, snapshot_dir)
        return count

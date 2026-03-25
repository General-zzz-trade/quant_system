#!/usr/bin/env python3
"""Remote backup for critical quant system state.

Creates timestamped tar.gz archives of models, state, checkpoints,
and decision logs. Optionally pushes to S3 if AWS CLI is available.

Usage:
    python3 scripts/backup_remote.py             # Full backup
    python3 scripts/backup_remote.py --dry-run    # Show what would be backed up
    python3 scripts/backup_remote.py --no-s3      # Skip S3 upload even if available
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BACKUP_ITEMS = [
    "models_v8/",
    "data/runtime/state.db",
    "data/runtime/checkpoints/",
    "data/runtime/decision_audit.jsonl",
    "data/runtime/ic_health.json",
    ".env",
]

LOCAL_BACKUP_DIR = Path("/backup/quant_system")
MAX_LOCAL_BACKUPS = 30

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
log = logging.getLogger("backup")


def _resolve_items(root: Path) -> list[Path]:
    """Resolve backup items to existing paths."""
    found: list[Path] = []
    for item in BACKUP_ITEMS:
        p = root / item
        if p.exists():
            found.append(p)
        else:
            log.warning("Backup item missing (skipped): %s", p)
    return found


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def _dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def create_backup(root: Path, backup_dir: Path, dry_run: bool = False) -> Path | None:
    """Create a timestamped tar.gz backup archive.

    Returns the path to the created archive, or None on dry-run.
    """
    items = _resolve_items(root)
    if not items:
        log.error("No backup items found. Nothing to back up.")
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"backup_{ts}.tar.gz"

    total_size = sum(_dir_size(p) for p in items)
    log.info("Backup items (%d):", len(items))
    for p in items:
        rel = p.relative_to(root)
        sz = _dir_size(p)
        log.info("  %s  (%s)", rel, _human_size(sz))
    log.info("Total uncompressed: %s", _human_size(total_size))

    if dry_run:
        log.info("[DRY-RUN] Would create: %s/%s", backup_dir, archive_name)
        return None

    # Ensure backup dir exists
    backup_dir.mkdir(parents=True, exist_ok=True)

    archive_path = backup_dir / archive_name
    t0 = time.monotonic()

    with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
        for p in items:
            arcname = str(p.relative_to(root))
            tar.add(str(p), arcname=arcname)

    elapsed = time.monotonic() - t0
    archive_size = archive_path.stat().st_size
    ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0
    log.info(
        "Created %s (%s, %.0f%% compression) in %.1fs",
        archive_path,
        _human_size(archive_size),
        ratio,
        elapsed,
    )
    return archive_path


def cleanup_old_backups(backup_dir: Path, max_keep: int = MAX_LOCAL_BACKUPS) -> None:
    """Remove oldest backups beyond max_keep."""
    if not backup_dir.exists():
        return

    archives = sorted(backup_dir.glob("backup_*.tar.gz"))
    to_remove = archives[:-max_keep] if len(archives) > max_keep else []

    for old in to_remove:
        old.unlink()
        log.info("Removed old backup: %s", old.name)

    if to_remove:
        log.info("Cleaned up %d old backup(s), keeping %d", len(to_remove), max_keep)


def push_to_s3(archive_path: Path, dry_run: bool = False) -> bool:
    """Push archive to S3 if AWS CLI is available and bucket is configured.

    Returns True if upload succeeded or was skipped gracefully.
    """
    if not shutil.which("aws"):
        log.info("AWS CLI not found; skipping S3 upload")
        return True

    bucket = os.environ.get("QUANT_BACKUP_S3_BUCKET", "")
    if not bucket:
        log.info("QUANT_BACKUP_S3_BUCKET not set; skipping S3 upload")
        return True

    s3_key = f"s3://{bucket}/quant_system/{archive_path.name}"

    if dry_run:
        log.info("[DRY-RUN] Would upload to %s", s3_key)
        return True

    log.info("Uploading to %s ...", s3_key)
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", str(archive_path), s3_key],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            log.info("S3 upload complete")
            return True
        else:
            log.error("S3 upload failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        log.error("S3 upload timed out (600s)")
        return False
    except Exception as e:
        log.error("S3 upload error: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant system backup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be backed up")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=LOCAL_BACKUP_DIR,
        help=f"Local backup directory (default: {LOCAL_BACKUP_DIR})",
    )
    parser.add_argument(
        "--max-keep",
        type=int,
        default=MAX_LOCAL_BACKUPS,
        help=f"Max local backups to retain (default: {MAX_LOCAL_BACKUPS})",
    )
    args = parser.parse_args()

    log.info("=== Quant System Backup ===")
    log.info("Project root: %s", PROJECT_ROOT)

    # Create archive
    archive = create_backup(PROJECT_ROOT, args.backup_dir, dry_run=args.dry_run)

    # S3 upload
    if archive and not args.no_s3:
        push_to_s3(archive, dry_run=args.dry_run)

    # Cleanup old backups
    if not args.dry_run:
        cleanup_old_backups(args.backup_dir, max_keep=args.max_keep)

    log.info("=== Backup complete ===")


if __name__ == "__main__":
    main()

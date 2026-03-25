#!/usr/bin/env python3
"""Archive rotated logs for long-term retention.

Logrotate keeps 14 daily rotations (~2 weeks). This script provides
longer-term archival:

1. Move .gz files older than 7 days into archive/YYYY-MM/ directories
2. Pack completed monthly dirs into a single tar.gz
3. Purge monthly archives older than 6 months
4. Archive a compressed copy of decision_audit.jsonl

Designed to run weekly via systemd timer (log-archive.timer).
"""

import argparse
import gzip
import logging
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

LOG_DIR = Path("/quant_system/logs")
ARCHIVE_DIR = LOG_DIR / "archive"
AUDIT_LOG = Path("/quant_system/data/runtime/decision_audit.jsonl")

ARCHIVE_AFTER_DAYS = 7          # Move .gz to archive after 7 days
MONTHLY_PACK_AFTER_DAYS = 30    # Pack monthly dirs into tar.gz after 30 days
RETENTION_MONTHS = 6            # Keep 6 months of archives

logger = logging.getLogger("log_archiver")


def _file_age_days(path: Path) -> float:
    """Return file age in days based on mtime."""
    mtime = path.stat().st_mtime
    return (datetime.now().timestamp() - mtime) / 86400


def _month_key(path: Path) -> str:
    """Return YYYY-MM string from file mtime."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime.strftime("%Y-%m")


def _ensure_dir(path: Path, dry_run: bool) -> None:
    if not path.exists():
        if dry_run:
            logger.info("[DRY-RUN] mkdir %s", path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", path)


def move_old_gz_to_archive(dry_run: bool) -> int:
    """Move .gz files older than ARCHIVE_AFTER_DAYS into archive/YYYY-MM/."""
    if not LOG_DIR.exists():
        logger.warning("Log directory does not exist: %s", LOG_DIR)
        return 0

    moved = 0
    # Scan top-level .gz files produced by logrotate
    for gz_file in sorted(LOG_DIR.glob("*.gz")):
        if not gz_file.is_file():
            continue
        age = _file_age_days(gz_file)
        if age < ARCHIVE_AFTER_DAYS:
            continue

        month = _month_key(gz_file)
        dest_dir = ARCHIVE_DIR / month
        _ensure_dir(dest_dir, dry_run)
        dest = dest_dir / gz_file.name

        # Handle name collisions by appending a counter
        if dest.exists():
            stem = gz_file.stem  # e.g. bybit_alpha.log.1
            suffix = gz_file.suffix  # .gz
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}.dup{counter}{suffix}"
                counter += 1

        if dry_run:
            logger.info("[DRY-RUN] move %s -> %s (%.1f days old)", gz_file, dest, age)
        else:
            shutil.move(str(gz_file), str(dest))
            logger.info("Moved %s -> %s (%.1f days old)", gz_file, dest, age)
        moved += 1

    return moved


def archive_audit_log(dry_run: bool) -> bool:
    """Create a compressed snapshot of decision_audit.jsonl in the archive."""
    if not AUDIT_LOG.exists():
        logger.info("No audit log found at %s, skipping", AUDIT_LOG)
        return False

    if AUDIT_LOG.stat().st_size == 0:
        logger.info("Audit log is empty, skipping")
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    month = datetime.now().strftime("%Y-%m")
    dest_dir = ARCHIVE_DIR / month
    _ensure_dir(dest_dir, dry_run)

    dest = dest_dir / f"decision_audit_{today}.jsonl.gz"
    if dest.exists():
        logger.info("Audit archive already exists for today: %s", dest)
        return False

    if dry_run:
        size_mb = AUDIT_LOG.stat().st_size / (1024 * 1024)
        logger.info("[DRY-RUN] compress audit log (%.1f MB) -> %s", size_mb, dest)
        return True

    # Stream-compress to avoid loading entire file into memory
    with open(AUDIT_LOG, "rb") as f_in, gzip.open(dest, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)

    logger.info("Archived audit log -> %s", dest)
    return True


def pack_monthly_archives(dry_run: bool) -> int:
    """Pack completed monthly directories into tar.gz files."""
    if not ARCHIVE_DIR.exists():
        logger.info("No archive directory yet")
        return 0

    packed = 0
    now = datetime.now()
    current_month = now.strftime("%Y-%m")

    for month_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not month_dir.is_dir():
            continue

        dir_name = month_dir.name  # e.g. 2026-02

        # Skip current month (still accumulating)
        if dir_name == current_month:
            continue

        # Skip if already packed
        tar_path = ARCHIVE_DIR / f"{dir_name}.tar.gz"
        if tar_path.exists():
            continue

        # Check if directory has any files
        files = list(month_dir.iterdir())
        if not files:
            logger.info("Empty month directory, removing: %s", month_dir)
            if not dry_run:
                month_dir.rmdir()
            continue

        # Only pack if the month is old enough
        try:
            month_dt = datetime.strptime(dir_name, "%Y-%m")
        except ValueError:
            logger.warning("Skipping non-month directory: %s", month_dir)
            continue

        # Pack if month ended more than MONTHLY_PACK_AFTER_DAYS ago
        month_end = (month_dt.replace(day=28) + timedelta(days=4)).replace(day=1)  # first of next month
        days_since_end = (now - month_end).days
        if days_since_end < MONTHLY_PACK_AFTER_DAYS:
            continue

        if dry_run:
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            logger.info(
                "[DRY-RUN] pack %s -> %s (%d files, %.1f MB)",
                month_dir, tar_path, len(files), total_size / (1024 * 1024),
            )
        else:
            with tarfile.open(tar_path, "w:gz") as tar:
                for f in sorted(files):
                    tar.add(str(f), arcname=f.name)
            logger.info("Packed %s -> %s (%d files)", month_dir, tar_path, len(files))
            # Remove the directory after successful packing
            shutil.rmtree(month_dir)
            logger.info("Removed packed directory: %s", month_dir)

        packed += 1

    return packed


def purge_old_archives(dry_run: bool) -> int:
    """Remove archives older than RETENTION_MONTHS."""
    if not ARCHIVE_DIR.exists():
        return 0

    purged = 0
    now = datetime.now()
    cutoff = now - timedelta(days=RETENTION_MONTHS * 30)
    cutoff_month = cutoff.strftime("%Y-%m")

    for item in sorted(ARCHIVE_DIR.iterdir()):
        name = item.name

        # Extract month from name (YYYY-MM.tar.gz or YYYY-MM directory)
        month_str = name.replace(".tar.gz", "")
        try:
            datetime.strptime(month_str, "%Y-%m")
        except ValueError:
            continue

        if month_str >= cutoff_month:
            continue

        if dry_run:
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info("[DRY-RUN] purge %s (%.1f MB)", item, size_mb)
            else:
                logger.info("[DRY-RUN] purge directory %s", item)
        else:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
            logger.info("Purged old archive: %s", item)

        purged += 1

    return purged


def run(dry_run: bool = False) -> dict:
    """Run the full archive pipeline. Returns summary counts."""
    _ensure_dir(ARCHIVE_DIR, dry_run)

    results = {}

    logger.info("=== Step 1: Move old .gz files to archive ===")
    results["moved"] = move_old_gz_to_archive(dry_run)

    logger.info("=== Step 2: Archive decision_audit.jsonl ===")
    results["audit_archived"] = archive_audit_log(dry_run)

    logger.info("=== Step 3: Pack completed monthly archives ===")
    results["packed"] = pack_monthly_archives(dry_run)

    logger.info("=== Step 4: Purge archives older than %d months ===", RETENTION_MONTHS)
    results["purged"] = purge_old_archives(dry_run)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive rotated logs for long-term retention")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--retention-months", type=int, default=None, help="Override retention months")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.log_dir:
        global LOG_DIR, ARCHIVE_DIR
        LOG_DIR = Path(args.log_dir)
        ARCHIVE_DIR = LOG_DIR / "archive"

    if args.retention_months is not None:
        global RETENTION_MONTHS
        RETENTION_MONTHS = args.retention_months

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    logger.info("Log archiver starting (%s mode)", mode)
    logger.info("Log dir: %s | Archive dir: %s | Retention: %d months", LOG_DIR, ARCHIVE_DIR, RETENTION_MONTHS)

    results = run(dry_run=args.dry_run)

    logger.info("=== Summary ===")
    logger.info("  .gz files moved to archive: %d", results["moved"])
    logger.info("  Audit log archived: %s", results["audit_archived"])
    logger.info("  Monthly dirs packed: %d", results["packed"])
    logger.info("  Old archives purged: %d", results["purged"])
    logger.info("Done.")


if __name__ == "__main__":
    main()

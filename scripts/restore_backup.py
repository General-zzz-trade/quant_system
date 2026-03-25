#!/usr/bin/env python3
"""Restore from backup archive.

Usage:
    python3 scripts/restore_backup.py                        # List available backups
    python3 scripts/restore_backup.py backup_20260325.tar.gz # Restore specific backup
    python3 scripts/restore_backup.py --latest               # Restore most recent backup
    python3 scripts/restore_backup.py backup.tar.gz --yes    # Skip confirmation prompt
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_BACKUP_DIR = Path("/backup/quant_system")

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
log = logging.getLogger("restore")


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def list_backups(backup_dir: Path) -> list[Path]:
    """List available backups sorted by name (oldest first)."""
    if not backup_dir.exists():
        log.error("Backup directory does not exist: %s", backup_dir)
        return []

    archives = sorted(backup_dir.glob("backup_*.tar.gz"))
    if not archives:
        log.info("No backups found in %s", backup_dir)
        return []

    print(f"\nAvailable backups in {backup_dir}:\n")
    print(f"  {'#':>3}  {'Archive':<40} {'Size':>10}")
    print(f"  {'---':>3}  {'--------':<40} {'----':>10}")
    for i, arc in enumerate(archives, 1):
        sz = _human_size(arc.stat().st_size)
        print(f"  {i:>3}  {arc.name:<40} {sz:>10}")
    print(f"\n  Total: {len(archives)} backup(s)\n")
    return archives


def resolve_archive(arg: str, backup_dir: Path) -> Path | None:
    """Resolve archive argument to a path."""
    # Direct path
    p = Path(arg)
    if p.exists() and p.is_file():
        return p

    # Name within backup dir
    p = backup_dir / arg
    if p.exists() and p.is_file():
        return p

    # Try without .tar.gz suffix
    if not arg.endswith(".tar.gz"):
        p = backup_dir / f"{arg}.tar.gz"
        if p.exists() and p.is_file():
            return p

    log.error("Archive not found: %s", arg)
    return None


def preview_archive(archive_path: Path) -> list[str]:
    """Show contents of a backup archive."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()

    print(f"\nArchive: {archive_path.name}")
    print(f"Contents ({len(members)} entries):\n")

    top_level: dict[str, int] = {}
    for m in members:
        top = m.name.split("/")[0]
        if top not in top_level:
            top_level[top] = 0
        if m.isfile():
            top_level[top] += m.size

    for name, size in sorted(top_level.items()):
        print(f"  {name:<50} {_human_size(size):>10}")
    print()
    return [m.name for m in members]


def restore(archive_path: Path, project_root: Path, skip_confirm: bool = False) -> bool:
    """Extract archive to a temp dir, then copy files into project root."""
    preview_archive(archive_path)

    if not skip_confirm:
        print(f"This will overwrite files in: {project_root}")
        print("The running system should be STOPPED before restoring.")
        try:
            answer = input("\nProceed with restore? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        if answer not in ("y", "yes"):
            log.info("Restore cancelled.")
            return False

    with tempfile.TemporaryDirectory(prefix="quant_restore_") as tmpdir:
        tmp = Path(tmpdir)
        log.info("Extracting to temp dir: %s", tmp)

        with tarfile.open(archive_path, "r:gz") as tar:
            # Security: check for path traversal
            for member in tar.getmembers():
                member_path = (tmp / member.name).resolve()
                if not str(member_path).startswith(str(tmp)):
                    log.error("Archive contains unsafe path: %s", member.name)
                    return False
            tar.extractall(tmp)

        # Copy each item into project root
        restored = 0
        for item in tmp.iterdir():
            dest = project_root / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                log.info("Restored directory: %s", item.name)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
                log.info("Restored file: %s", item.name)
            restored += 1

        log.info("Restore complete: %d item(s) restored from %s", restored, archive_path.name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore quant system from backup")
    parser.add_argument("archive", nargs="?", help="Backup archive file (name or path)")
    parser.add_argument("--latest", action="store_true", help="Restore from most recent backup")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=LOCAL_BACKUP_DIR,
        help=f"Backup directory (default: {LOCAL_BACKUP_DIR})",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=f"Project root to restore into (default: {PROJECT_ROOT})",
    )
    args = parser.parse_args()

    # List mode
    if not args.archive and not args.latest:
        list_backups(args.backup_dir)
        return

    # Resolve archive
    if args.latest:
        archives = sorted(args.backup_dir.glob("backup_*.tar.gz"))
        if not archives:
            log.error("No backups found in %s", args.backup_dir)
            sys.exit(1)
        archive_path = archives[-1]
        log.info("Using latest backup: %s", archive_path.name)
    else:
        archive_path = resolve_archive(args.archive, args.backup_dir)
        if archive_path is None:
            sys.exit(1)

    if not restore(archive_path, args.project_root, skip_confirm=args.yes):
        sys.exit(1)


if __name__ == "__main__":
    main()

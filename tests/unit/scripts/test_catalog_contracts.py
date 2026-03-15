from __future__ import annotations

from pathlib import Path

from scripts.catalog import (
    ARCHIVE_CANDIDATE,
    EXPERIMENTAL,
    LEGACY,
    PRIMARY_ENTRYPOINTS,
    SUPPORTED,
    SUPPORTED_ENTRYPOINTS,
)

VALID_STATUSES = {SUPPORTED, EXPERIMENTAL, LEGACY, ARCHIVE_CANDIDATE}


def test_primary_entrypoints_have_valid_status_and_unique_names() -> None:
    names = [entry.name for entry in PRIMARY_ENTRYPOINTS]
    assert len(names) == len(set(names))
    assert {entry.status for entry in PRIMARY_ENTRYPOINTS} <= VALID_STATUSES


def test_primary_entrypoints_exist_on_disk() -> None:
    for entry in PRIMARY_ENTRYPOINTS:
        assert (Path("scripts") / entry.name).exists(), entry.name


def test_supported_entrypoints_have_valid_python_syntax() -> None:
    for entry in SUPPORTED_ENTRYPOINTS:
        path = Path("scripts") / entry.name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


def test_scripts_readme_mentions_all_supported_entrypoints() -> None:
    """README quick reference must mention all 'supported' scripts.
    Experimental/legacy/archive scripts may be omitted from README
    (catalog.py is the full source of truth, not README).
    """
    text = Path("scripts/README.md").read_text(encoding="utf-8")
    for entry in SUPPORTED_ENTRYPOINTS:
        assert entry.name in text, f"supported script {entry.name} missing from README"


def test_scripts_readme_declares_status_meanings() -> None:
    text = Path("scripts/README.md").read_text(encoding="utf-8")
    assert "`supported`" in text
    assert "`experimental`" in text
    assert "`legacy`" in text

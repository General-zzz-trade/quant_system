from __future__ import annotations

from unittest.mock import patch

from scripts.catalog import ARCHIVE_NOTE, PRIMARY_ENTRYPOINTS, SCRIPT_GROUPS, render_catalog
from scripts.cli import main


def test_render_catalog_includes_primary_entrypoints():
    text = render_catalog()
    for entry in PRIMARY_ENTRYPOINTS:
        assert entry.name in text
        assert entry.status in text
        assert entry.recommendation in text


def test_render_catalog_includes_groups_and_archive_note():
    text = render_catalog()
    for group in SCRIPT_GROUPS:
        assert group.name in text
        assert group.purpose in text
    assert ARCHIVE_NOTE in text


def test_cli_catalog_prints_catalog(capsys):
    with patch("sys.argv", ["quant", "catalog", "--scripts"]):
        main()
    captured = capsys.readouterr()
    assert "Scripts Catalog" in captured.out
    assert "Primary entrypoints:" in captured.out

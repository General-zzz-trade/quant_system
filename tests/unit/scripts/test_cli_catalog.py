from __future__ import annotations

import json
from unittest.mock import patch

from execution.store.event_log import SQLiteEventLog
from research.model_registry.registry import ModelRegistry
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


def test_cli_model_inspect_prints_loader_report(capsys):
    class _FakeLoader:
        def inspect_production_models(self, names):
            return [{"name": names[0], "available": True, "model_id": "m1"}]

    with (
        patch("research.model_registry.registry.ModelRegistry"),
        patch("research.model_registry.artifact.ArtifactStore"),
        patch("alpha.model_loader.ProductionModelLoader", return_value=_FakeLoader()),
    ):
        with patch("sys.argv", ["quant", "model-inspect", "--model", "alpha_btc"]):
            main()

    captured = capsys.readouterr()
    assert '"name": "alpha_btc"' in captured.out
    assert '"model_id": "m1"' in captured.out


def test_cli_model_promote_calls_registry(capsys):
    with patch("research.model_registry.registry.ModelRegistry") as registry_cls:
        registry = registry_cls.return_value
        registry.get.return_value = type(
            "MV", (), {"name": "alpha_btc", "version": 3})()
        with patch("sys.argv", [
            "quant", "model-promote", "--model-id", "m-prod",
            "--reason", "shadow_win", "--actor", "ops",
        ]):
            main()

    registry.promote.assert_called_once_with("m-prod", reason="shadow_win", actor="ops")
    captured = capsys.readouterr()
    assert '"action": "promote"' in captured.out
    assert '"model_id": "m-prod"' in captured.out
    assert '"model": "alpha_btc"' in captured.out
    assert '"version": 3' in captured.out
    assert '"reason": "shadow_win"' in captured.out
    assert '"actor": "ops"' in captured.out


def test_cli_model_rollback_calls_registry_and_prints_target(capsys):
    with patch("research.model_registry.registry.ModelRegistry") as registry_cls:
        registry = registry_cls.return_value
        registry.get_production.return_value = type("MV", (), {"model_id": "m-current", "version": 2})()
        registry.rollback_to_previous.return_value = type("MV", (), {"model_id": "m-prev", "version": 1})()
        with patch("sys.argv", [
            "quant", "model-rollback", "--model", "alpha_btc",
            "--reason", "rollback", "--actor", "ops",
        ]):
            main()

    registry.rollback_to_previous.assert_called_once_with(
        "alpha_btc",
        to_model_id=None,
        to_version=None,
        reason="rollback",
        actor="ops",
    )
    captured = capsys.readouterr()
    assert '"action": "rollback"' in captured.out
    assert '"from_model_id": "m-current"' in captured.out
    assert '"model_id": "m-prev"' in captured.out
    assert '"reason": "rollback"' in captured.out
    assert '"actor": "ops"' in captured.out


def test_cli_model_rollback_supports_explicit_target_args(capsys):
    with patch("research.model_registry.registry.ModelRegistry") as registry_cls:
        registry = registry_cls.return_value
        registry.get_production.return_value = type("MV", (), {"model_id": "m-current", "version": 4})()
        registry.rollback_to_previous.return_value = type("MV", (), {"model_id": "m-v2", "version": 2})()
        with patch("sys.argv", ["quant", "model-rollback", "--model", "alpha_btc", "--to-version", "2"]):
            main()

    registry.rollback_to_previous.assert_called_once_with(
        "alpha_btc",
        to_model_id=None,
        to_version=2,
        reason=None,
        actor="cli",
    )
    captured = capsys.readouterr()
    assert '"version": 2' in captured.out


def test_cli_model_history_prints_registry_actions(capsys):
    action = type(
        "Action",
        (),
        {
            "action_id": 7,
            "name": "alpha_btc",
            "action": "promote",
            "from_model_id": "m-old",
            "to_model_id": "m-new",
            "reason": "shadow_win",
            "actor": "ops",
            "created_at": type("Ts", (), {"isoformat": lambda self: "2026-03-13T00:00:00+00:00"})(),
            "metadata": {"ticket": "chg-1"},
        },
    )()
    with patch("research.model_registry.registry.ModelRegistry") as registry_cls:
        registry = registry_cls.return_value
        registry.list_actions.return_value = [action]
        with patch("sys.argv", ["quant", "model-history", "--model", "alpha_btc", "--limit", "5"]):
            main()

    registry.list_actions.assert_called_once_with("alpha_btc", limit=5)
    captured = capsys.readouterr()
    assert '"action": "promote"' in captured.out
    assert '"reason": "shadow_win"' in captured.out


def test_cli_ops_audit_prints_operator_controls_and_model_actions(tmp_path, capsys):
    event_log = SQLiteEventLog(str(tmp_path / "event_log.db"))
    event_log.append(
        event_type="operator_control",
        correlation_id="halt",
        payload={"command": "halt", "reason": "manual_halt", "source": "ops", "result": "hard_kill"},
    )
    event_log.append(
        event_type="model_reload",
        correlation_id="failed",
        payload={
            "outcome": "failed", "model_names": ["alpha_btc"],
            "error": "model_hot_reload_failed", "ts": "2026-03-13T00:00:01+00:00",
        },
    )
    event_log.append(
        event_type="execution_incident",
        correlation_id="binance:BTCUSDT:timeout",
        payload={
            "title": "execution-timeout", "category": "execution_timeout",
            "source": "execution:timeout", "ts": "2026-03-13T00:00:02+00:00",
        },
    )
    registry = ModelRegistry(tmp_path / "registry.db")
    mv1 = registry.register(name="alpha_btc", params={}, features=[], metrics={"sharpe": 1.0})
    registry.promote(mv1.model_id, reason="shadow_win", actor="ops")

    with patch(
        "sys.argv",
        [
            "quant",
            "ops-audit",
            "--registry-db",
            str(tmp_path / "registry.db"),
            "--event-log",
            str(tmp_path / "event_log.db"),
            "--model",
            "alpha_btc",
            "--limit",
            "5",
        ],
    ):
        main()

    captured = capsys.readouterr()
    assert '"operator_controls"' in captured.out
    assert '"execution_incidents"' in captured.out
    assert '"command": "halt"' in captured.out
    assert '"model_actions"' in captured.out
    assert '"reason": "shadow_win"' in captured.out
    assert '"timeline"' in captured.out
    assert '"model_reload_failed"' in captured.out
    assert '"execution-timeout"' in captured.out


def test_cli_ops_audit_timeline_is_sorted_and_limited(tmp_path, capsys):
    event_log = SQLiteEventLog(str(tmp_path / "event_log.db"))
    event_log.append(
        event_type="operator_control",
        correlation_id="halt",
        payload={
            "command": "halt",
            "reason": "manual_halt",
            "source": "ops",
            "result": "hard_kill",
            "ts": "2099-03-13T00:00:01+00:00",
        },
    )
    event_log.append(
        event_type="execution_incident",
        correlation_id="binance:BTCUSDT:timeout",
        payload={
            "title": "execution-timeout",
            "category": "execution_timeout",
            "source": "execution:timeout",
            "ts": "2099-03-13T00:00:02+00:00",
        },
    )
    event_log.append(
        event_type="model_reload",
        correlation_id="failed",
        payload={
            "outcome": "failed",
            "model_names": ["alpha_btc"],
            "error": "model_hot_reload_failed",
            "ts": "2099-03-13T00:00:03+00:00",
        },
    )
    registry = ModelRegistry(tmp_path / "registry.db")
    mv1 = registry.register(name="alpha_btc", params={}, features=[], metrics={"sharpe": 1.0})
    registry.promote(mv1.model_id, reason="shadow_win", actor="ops")

    with patch(
        "sys.argv",
        [
            "quant",
            "ops-audit",
            "--registry-db",
            str(tmp_path / "registry.db"),
            "--event-log",
            str(tmp_path / "event_log.db"),
            "--model",
            "alpha_btc",
            "--limit",
            "3",
        ],
    ):
        main()

    body = json.loads(capsys.readouterr().out)
    timeline = body["timeline"]

    assert len(timeline) == 3
    assert timeline == sorted(timeline, key=lambda row: row["ts"], reverse=True)
    kinds = {row["kind"] for row in timeline}
    assert timeline[0]["kind"] == "model_reload"
    assert timeline[1]["kind"] == "execution_incident"
    assert timeline[2]["kind"] == "control"
    assert "model_action" not in kinds

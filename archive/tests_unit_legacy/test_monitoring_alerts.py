"""Tests for monitoring.alerts — alert sinks and supporting types."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from monitoring.alerts.base import (
    Alert,
    CompositeAlertSink,
    DedupAlertSink,
    Severity,
)
from monitoring.alerts.console import ConsoleAlertSink
from monitoring.alerts.log_sink import LogAlertSink


def _make_alert(**kwargs) -> Alert:
    defaults = dict(
        title="Test Alert",
        message="Something happened",
        severity=Severity.WARNING,
        source="test",
        ts=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return Alert(**defaults)


class TestAlert:
    def test_to_dict(self) -> None:
        alert = _make_alert(meta={"key": "val"})
        d = alert.to_dict()
        assert d["title"] == "Test Alert"
        assert d["severity"] == "warning"
        assert d["meta"] == {"key": "val"}
        assert "ts" in d

    def test_default_severity_is_info(self) -> None:
        alert = Alert(title="t", message="m")
        assert alert.severity == Severity.INFO

    def test_severity_ordering(self) -> None:
        levels = list(Severity)
        assert levels.index(Severity.DEBUG) < levels.index(Severity.CRITICAL)


class TestConsoleAlertSink:
    def test_emit_prints(self, capsys) -> None:
        sink = ConsoleAlertSink()
        sink.emit(_make_alert())
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "Test Alert" in out
        assert "Something happened" in out


class TestLogAlertSink:
    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.jsonl"
            sink = LogAlertSink(path=path)
            sink.emit(_make_alert())
            sink.emit(_make_alert(title="Second"))

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            first = json.loads(lines[0])
            assert first["title"] == "Test Alert"
            second = json.loads(lines[1])
            assert second["title"] == "Second"


class TestCompositeAlertSink:
    def test_fanout(self) -> None:
        received: list[Alert] = []

        class Collector:
            def emit(self, alert: Alert) -> None:
                received.append(alert)

        comp = CompositeAlertSink(sinks=(Collector(), Collector()))
        comp.emit(_make_alert())
        assert len(received) == 2


class TestDedupAlertSink:
    def test_suppresses_duplicate(self) -> None:
        received: list[Alert] = []

        class Collector:
            def emit(self, alert: Alert) -> None:
                received.append(alert)

        dedup = DedupAlertSink(delegate=Collector(), window_seconds=60.0)
        alert = _make_alert()
        dedup.emit(alert)
        dedup.emit(alert)  # same title+severity within window
        assert len(received) == 1

    def test_different_title_not_suppressed(self) -> None:
        received: list[Alert] = []

        class Collector:
            def emit(self, alert: Alert) -> None:
                received.append(alert)

        dedup = DedupAlertSink(delegate=Collector(), window_seconds=60.0)
        dedup.emit(_make_alert(title="A"))
        dedup.emit(_make_alert(title="B"))
        assert len(received) == 2

"""Tests for monitoring.alerts.factory — sink construction from config dicts."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from monitoring.alerts.base import Alert, CompositeAlertSink, Severity
from monitoring.alerts.console import ConsoleAlertSink
from monitoring.alerts.factory import build_alert_sink
from monitoring.alerts.log_sink import LogAlertSink


def test_build_console_sink():
    sink = build_alert_sink({"type": "console"})
    assert isinstance(sink, ConsoleAlertSink)


def test_build_log_sink():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "alerts.jsonl")
        sink = build_alert_sink({"type": "log", "path": path})
        assert isinstance(sink, LogAlertSink)
        assert sink.path == Path(path)


def test_build_composite_sink():
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "type": "composite",
            "sinks": [
                {"type": "console"},
                {"type": "log", "path": str(Path(tmp) / "a.jsonl")},
            ],
        }
        sink = build_alert_sink(config)
        assert isinstance(sink, CompositeAlertSink)
        assert len(sink.sinks) == 2
        assert isinstance(sink.sinks[0], ConsoleAlertSink)
        assert isinstance(sink.sinks[1], LogAlertSink)


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown alert sink type"):
        build_alert_sink({"type": "carrier_pigeon"})


def test_build_webhook_sink():
    from monitoring.alerts.webhook import WebhookAlertSink
    sink = build_alert_sink({"type": "webhook", "url": "https://example.com/hook"})
    assert isinstance(sink, WebhookAlertSink)
    assert sink.url == "https://example.com/hook"
    assert sink.min_severity == Severity.WARNING

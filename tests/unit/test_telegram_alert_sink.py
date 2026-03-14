# tests/unit/test_telegram_alert_sink.py
"""Tests for TelegramAlertSink and factory integration."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from monitoring.alerts.base import Alert, Severity
from monitoring.alerts.channels import TelegramAlertSink, TelegramConfig
from monitoring.alerts.factory import build_alert_sink


def _make_alert(**kwargs) -> Alert:
    defaults = dict(
        title="Test Alert",
        message="Something happened",
        severity=Severity.WARNING,
        source="test",
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return Alert(**defaults)


def test_telegram_sends_correct_payload():
    cfg = TelegramConfig(bot_token="123:ABC", chat_id="456")
    sink = TelegramAlertSink(config=cfg)
    alert = _make_alert()

    with patch("monitoring.alerts.channels.urllib.request.urlopen") as mock_open:
        sink.emit(alert)
        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        assert "bot123:ABC/sendMessage" in req.full_url
        body = req.data.decode()
        assert "chat_id=456" in body
        assert "WARNING" in body
        assert "Test+Alert" in body or "Test%20Alert" in body or "Test Alert" in body


def test_telegram_http_error_no_raise():
    cfg = TelegramConfig(bot_token="tok", chat_id="cid")
    sink = TelegramAlertSink(config=cfg)

    with patch("monitoring.alerts.channels.urllib.request.urlopen", side_effect=OSError("network")):
        sink.emit(_make_alert())  # should not raise


def test_factory_builds_telegram_sink():
    sink = build_alert_sink({"type": "telegram", "bot_token": "t", "chat_id": "c"})
    assert isinstance(sink, TelegramAlertSink)
    assert sink._config.bot_token == "t"
    assert sink._config.chat_id == "c"

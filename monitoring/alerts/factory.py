"""Sink factory — build AlertSink instances from configuration dicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from monitoring.alerts.base import AlertSink, CompositeAlertSink, Severity
from monitoring.alerts.console import ConsoleAlertSink
from monitoring.alerts.log_sink import LogAlertSink
from monitoring.alerts.webhook import WebhookAlertSink


def build_alert_sink(config: Dict[str, Any]) -> AlertSink:
    """Build an AlertSink from a config dict.

    Supported types:
        console  — prints to stdout
        log      — appends JSONL to a file (requires "path")
        webhook  — POSTs JSON to a URL (requires "url", optional "headers")
        telegram — sends via Telegram bot (requires "bot_token", "chat_id")
        composite — fans out to multiple sinks (requires "sinks" list)
    """
    sink_type = config.get("type", "")

    if sink_type == "console":
        return ConsoleAlertSink()

    if sink_type == "log":
        return LogAlertSink(path=Path(config["path"]))

    if sink_type == "webhook":
        min_sev = Severity(config["min_severity"]) if "min_severity" in config else Severity.WARNING
        return WebhookAlertSink(
            url=config["url"],
            headers=config.get("headers", {}),
            min_severity=min_sev,
        )

    if sink_type == "telegram":
        from monitoring.alerts.channels import TelegramAlertSink, TelegramConfig
        return TelegramAlertSink(
            config=TelegramConfig(
                bot_token=config["bot_token"],
                chat_id=config["chat_id"],
            ),
        )

    if sink_type == "composite":
        children = [build_alert_sink(c) for c in config["sinks"]]
        return CompositeAlertSink(sinks=children)

    raise ValueError(f"Unknown alert sink type: {sink_type!r}")

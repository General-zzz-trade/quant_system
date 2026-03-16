# runner/builders/core_infra_builder.py
"""Phase 1: core infrastructure — structured logging, latency tracker, fill recorder, kill switch.

Extracted from LiveRunner._build_core_infra().
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_core_infra(
    config: Any,
    on_fill: Optional[Callable[[Any], None]],
    alert_sink: Optional[Any],
    fills: List[Dict[str, Any]],
) -> tuple:
    """Phase 1: structured logging, latency tracker, fill recorder, kill switch, alert sink.

    Returns (latency_tracker, _record_fill, kill_switch, alert_sink).
    """
    from execution.latency.tracker import LatencyTracker
    from risk.kill_switch import KillSwitch

    # ── Auto-wire Telegram alerts from env vars ──────────
    if alert_sink is None:
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat:
            from monitoring.alerts.telegram import TelegramAlertSink
            from monitoring.alerts.base import CompositeAlertSink
            from monitoring.alerts.console import ConsoleAlertSink
            alert_sink = CompositeAlertSink(sinks=[
                ConsoleAlertSink(),
                TelegramAlertSink(tg_token, tg_chat),
            ])
            logger.info("Telegram alerts auto-wired (chat_id=%s)", tg_chat)

    # ── 0) Structured logging ─────────────────────────────
    if config.enable_structured_logging:
        from infra.logging.structured import setup_structured_logging
        setup_structured_logging(
            level=config.log_level,
            log_file=config.log_file,
        )

    # ── 1) LatencyTracker ─────────────────────────────────
    latency_tracker = LatencyTracker()

    def _record_fill(fill: Any) -> None:
        from execution.models.fills import fill_to_record
        fills.append(fill_to_record(fill))
        order_id = getattr(fill, "order_id", None)
        if order_id:
            latency_tracker.record_fill(str(order_id))
        if on_fill is not None:
            on_fill(fill)

    # ── 2) KillSwitch ────────────────────────────────────
    kill_switch = KillSwitch()

    return latency_tracker, _record_fill, kill_switch, alert_sink

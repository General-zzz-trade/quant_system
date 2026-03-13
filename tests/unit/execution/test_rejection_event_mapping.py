from __future__ import annotations

from execution.observability.rejections import rejection_event_to_alert
from execution.models.rejection_events import rejection_to_event
from execution.models.rejections import CanonicalRejection
from monitoring.alerts.base import Severity


def test_rejection_to_event_maps_canonical_rejection() -> None:
    rejection = CanonicalRejection(
        status="REJECTED",
        command_id="cmd-1",
        venue="binance",
        symbol="BTCUSDT",
        reason="insufficient_balance",
        reason_family="balance",
        retryable=False,
        deduped=False,
    )

    event = rejection_to_event(rejection)

    assert event.event_type == "EXECUTION_REJECT"
    assert event.header.event_type == "EXECUTION_REJECT"
    assert event.status == "REJECTED"
    assert event.command_id == "cmd-1"
    assert event.venue == "binance"
    assert event.symbol == "BTCUSDT"
    assert event.reason == "insufficient_balance"
    assert event.reason_family == "balance"
    assert event.routing_key == "binance:BTCUSDT:rejected:balance"
    assert event.retryable is False


def test_rejection_event_to_alert_maps_non_retryable_to_error() -> None:
    rejection = CanonicalRejection(
        status="REJECTED",
        command_id="cmd-1",
        venue="binance",
        symbol="BTCUSDT",
        reason="insufficient_balance",
        reason_family="balance",
        retryable=False,
        deduped=False,
    )

    alert = rejection_event_to_alert(rejection_to_event(rejection))

    assert alert.title == "execution-rejected"
    assert alert.severity == Severity.ERROR
    assert alert.source == "execution:rejection"
    assert alert.meta is not None
    assert alert.meta["category"] == "execution_rejection"
    assert alert.meta["routing_key"] == "binance:BTCUSDT:rejected:balance"
    assert alert.meta["status"] == "REJECTED"
    assert alert.meta["reason_family"] == "balance"
    assert alert.meta["command_id"] == "cmd-1"
    assert alert.meta["retryable"] is False


def test_rejection_event_to_alert_maps_retryable_failed_to_warning() -> None:
    rejection = CanonicalRejection(
        status="FAILED",
        command_id="cmd-2",
        venue="binance",
        symbol="ETHUSDT",
        reason="retryable:TimeoutError:timeout",
        reason_family="timeout",
        retryable=True,
        deduped=False,
    )

    alert = rejection_event_to_alert(rejection_to_event(rejection))

    assert alert.title == "execution-failed"
    assert alert.severity == Severity.WARNING
    assert alert.meta is not None
    assert alert.meta["status"] == "FAILED"
    assert alert.meta["routing_key"] == "binance:ETHUSDT:failed:timeout"
    assert alert.meta["reason_family"] == "timeout"
    assert alert.meta["retryable"] is True

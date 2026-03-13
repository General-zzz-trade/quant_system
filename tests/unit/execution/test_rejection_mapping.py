from __future__ import annotations

from execution.bridge.execution_bridge import Ack
from execution.models.rejections import ack_to_rejection


def test_ack_to_rejection_maps_rejected_ack() -> None:
    ack = Ack(
        status="REJECTED",
        command_id="cmd-1",
        idempotency_key="idem-1",
        venue="binance",
        symbol="BTCUSDT",
        attempts=1,
        deduped=False,
        result=None,
        error="non_retryable:invalid quantity",
    )

    rejection = ack_to_rejection(ack)

    assert rejection is not None
    assert rejection.status == "REJECTED"
    assert rejection.command_id == "cmd-1"
    assert rejection.venue == "binance"
    assert rejection.symbol == "BTCUSDT"
    assert rejection.reason == "non_retryable:invalid quantity"
    assert rejection.reason_family == "validation"
    assert rejection.retryable is False


def test_ack_to_rejection_maps_failed_ack_as_retryable() -> None:
    class _Ack:
        status = "FAILED"
        ok = False
        venue = "binance"
        command_id = "cmd-2"
        error = "retryable:TimeoutError:timeout"

    rejection = ack_to_rejection(_Ack(), default_symbol="ETHUSDT")

    assert rejection is not None
    assert rejection.status == "FAILED"
    assert rejection.command_id == "cmd-2"
    assert rejection.symbol == "ETHUSDT"
    assert rejection.reason_family == "timeout"
    assert rejection.retryable is True


def test_ack_to_rejection_marks_deduped_rejects_with_deduped_family() -> None:
    ack = Ack(
        status="REJECTED",
        command_id="cmd-4",
        idempotency_key="idem-4",
        venue="binance",
        symbol="BTCUSDT",
        attempts=1,
        deduped=True,
        result=None,
        error="duplicate request",
    )

    rejection = ack_to_rejection(ack)

    assert rejection is not None
    assert rejection.reason_family == "deduped"
    assert rejection.deduped is True


def test_ack_to_rejection_returns_none_for_accepted_ack() -> None:
    ack = Ack(
        status="ACCEPTED",
        command_id="cmd-3",
        idempotency_key="idem-3",
        venue="binance",
        symbol="BTCUSDT",
        attempts=1,
        result={"order_id": "v-1"},
        error=None,
    )

    assert ack_to_rejection(ack) is None

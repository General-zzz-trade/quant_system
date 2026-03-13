from __future__ import annotations

from execution.bridge.execution_bridge import Ack
from execution.models.acks import normalize_ack


def test_normalize_ack_from_execution_bridge_ack() -> None:
    ack = Ack(
        status="ACCEPTED",
        command_id="cmd-1",
        idempotency_key="idem-1",
        venue="binance",
        symbol="BTCUSDT",
        attempts=2,
        deduped=False,
        result={"price": "40000", "qty": "1.0"},
        error=None,
    )

    normalized = normalize_ack(ack)

    assert normalized.status == "ACCEPTED"
    assert normalized.ok is True
    assert normalized.command_id == "cmd-1"
    assert normalized.venue == "binance"
    assert normalized.symbol == "BTCUSDT"
    assert normalized.attempts == 2
    assert normalized.result == {"price": "40000", "qty": "1.0"}


def test_normalize_ack_from_simple_test_double() -> None:
    class _Ack:
        ok = False
        status = "REJECTED"
        venue = "binance"
        command_id = "cmd-2"
        error = "insufficient_balance"

    normalized = normalize_ack(_Ack(), default_symbol="ETHUSDT")

    assert normalized.status == "REJECTED"
    assert normalized.ok is False
    assert normalized.command_id == "cmd-2"
    assert normalized.venue == "binance"
    assert normalized.symbol == "ETHUSDT"
    assert normalized.error == "insufficient_balance"
    assert normalized.result is None

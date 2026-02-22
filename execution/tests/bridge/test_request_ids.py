from __future__ import annotations

from execution.bridge.request_ids import RequestIdFactory, make_idempotency_key


def test_client_order_id_deterministic_same_logical_id_same_output() -> None:
    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=True, max_len=36)

    a = rid.client_order_id(strategy="ema_pullback", symbol="BTCUSDT", logical_id="sig-1")
    b = rid.client_order_id(strategy="ema_pullback", symbol="BTCUSDT", logical_id="sig-1")

    assert a == b
    assert len(a) <= 36


def test_client_order_id_non_deterministic_without_logical_id_unique() -> None:
    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=False, max_len=36)

    a = rid.client_order_id(strategy="ema", symbol="BTCUSDT")
    b = rid.client_order_id(strategy="ema", symbol="BTCUSDT")

    assert a != b
    assert len(a) <= 36 and len(b) <= 36


def test_idempotency_key_stable_and_sensitive_to_action_or_key() -> None:
    k1 = make_idempotency_key(venue="binance", action="submit", key="abc")
    k2 = make_idempotency_key(venue="binance", action="submit", key="abc")
    k3 = make_idempotency_key(venue="binance", action="cancel", key="abc")
    k4 = make_idempotency_key(venue="binance", action="submit", key="xyz")

    assert k1 == k2
    assert k1 != k3
    assert k1 != k4

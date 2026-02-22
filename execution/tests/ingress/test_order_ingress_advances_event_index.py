from __future__ import annotations

from types import SimpleNamespace
import pytest

from engine.coordinator import EngineCoordinator, CoordinatorConfig
from execution.ingress.order_router import OrderIngressRouter


def _mk_order(*, order_id: str, qty: float, digest: str, ts_ms: int = 1704067200000) -> SimpleNamespace:
    return SimpleNamespace(
        venue="binance",
        symbol="BTCUSDT",
        order_id=order_id,
        client_order_id="c-1",
        status="NEW",
        side="buy",
        order_type="LIMIT",
        tif="GTC",
        qty=qty,
        price=100.0,
        filled_qty=0.0,
        avg_price=None,
        ts_ms=ts_ms,
        order_key=f"binance:BTCUSDT:order:{order_id}",
        payload_digest=digest,
    )


def test_order_update_ingress_advances_event_index_and_is_idempotent() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = OrderIngressRouter(coordinator=coord, default_actor="venue:binance")

    assert coord.get_state_view()["event_index"] == 0

    o1 = _mk_order(order_id="o-1", qty=1.0, digest="d1")
    assert router.ingest_canonical_order(o1) is True
    assert coord.get_state_view()["event_index"] == 1

    # same key + same digest => drop, event_index 不再推进
    o1_dup = _mk_order(order_id="o-1", qty=1.0, digest="d1")
    assert router.ingest_canonical_order(o1_dup) is False
    assert coord.get_state_view()["event_index"] == 1


def test_order_update_ingress_payload_mismatch_must_fail_fast() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = OrderIngressRouter(coordinator=coord, default_actor="venue:binance")

    assert router.ingest_canonical_order(_mk_order(order_id="o-9", qty=1.0, digest="ok")) is True
    assert coord.get_state_view()["event_index"] == 1

    # same order_key but different digest => 数据损坏，必须 fail-fast
    with pytest.raises(ValueError):
        router.ingest_canonical_order(_mk_order(order_id="o-9", qty=9.0, digest="BAD"))

    # fail-fast 不应推进 event_index
    assert coord.get_state_view()["event_index"] == 1

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from execution.ingress.order_router import OrderIngressRouter
from execution.models.orders import CanonicalOrder, ingress_order_dedup_identity


def _canonical_order(*, payload_digest: str = "", status: str = "new") -> CanonicalOrder:
    return CanonicalOrder(
        venue="binance",
        symbol="BTCUSDT",
        order_id="ord-1",
        client_order_id="cid-1",
        status=status,
        side="buy",
        order_type="limit",
        tif="gtc",
        qty=Decimal("1.0"),
        price=Decimal("42000"),
        filled_qty=Decimal("0"),
        avg_price=None,
        ts_ms=1704067200000,
        order_key="",
        payload_digest=payload_digest,
    )


def _router() -> OrderIngressRouter:
    return OrderIngressRouter(coordinator=SimpleNamespace(emit=lambda *_args, **_kwargs: None))



def test_order_router_exposes_same_dedup_identity_as_helper() -> None:
    router = _router()
    order = _canonical_order()

    assert router._dedup_key_and_digest(order) == ingress_order_dedup_identity(order)


def test_order_router_deduplicates_equivalent_orders_without_precomputed_digest() -> None:
    emitted: list[object] = []
    router = OrderIngressRouter(coordinator=SimpleNamespace(emit=lambda event, actor=None: emitted.append((event,
        actor))))

    assert router.ingest_canonical_order(_canonical_order(), actor="venue:test") is True
    assert router.ingest_canonical_order(_canonical_order(), actor="venue:test") is False
    assert len(emitted) == 1


def test_order_router_rejects_payload_mismatch_for_same_order_identity() -> None:
    router = _router()

    assert router.ingest_canonical_order(_canonical_order()) is True
    with pytest.raises(ValueError, match="payload mismatch"):
        router.ingest_canonical_order(_canonical_order(status="filled"))

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from execution.ingress.router import FillIngressRouter
from execution.models.fills import CanonicalFill


def _canonical_fill(*, digest: str = "digest-1", qty: str = "0.25") -> CanonicalFill:
    return CanonicalFill(
        venue="binance",
        symbol="BTCUSDT",
        order_id="ord-1",
        trade_id="trade-1",
        fill_id="fill-1",
        side="buy",
        qty=Decimal(qty),
        price=Decimal("42500"),
        fee=Decimal("0.10"),
        ts_ms=int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
        payload_digest=digest,
    )


def _legacy_fill(*, digest: str = "digest-1", qty: float = 0.25) -> dict[str, object]:
    return {
        "venue": "binance",
        "symbol": "BTCUSDT",
        "order_id": "ord-1",
        "trade_id": "trade-1",
        "fill_id": "fill-1",
        "side": "buy",
        "qty": qty,
        "price": 42500.0,
        "fee": 0.10,
        "ts_ms": int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
        "payload_digest": digest,
    }


def _router() -> FillIngressRouter:
    return FillIngressRouter(coordinator=SimpleNamespace(emit=lambda *_args, **_kwargs: None))


def test_canonical_and_legacy_fill_share_same_dedup_identity() -> None:
    router = _router()

    canonical = _canonical_fill()
    legacy = _legacy_fill()

    canonical_event = router._to_fill_event(canonical)
    legacy_event = router._to_fill_event(legacy)

    assert router._dedup_key_and_digest(canonical_event) == router._dedup_key_and_digest(legacy_event)


def test_router_deduplicates_equivalent_canonical_and_legacy_fill() -> None:
    emitted: list[object] = []
    router = FillIngressRouter(coordinator=SimpleNamespace(emit=lambda event, actor=None: emitted.append((event,
        actor))))

    assert router.ingest_canonical_fill(_canonical_fill(), actor="venue:test") is True
    assert router.ingest_canonical_fill(_legacy_fill(), actor="venue:test") is False
    assert len(emitted) == 1


def test_router_rejects_payload_mismatch_across_canonical_and_legacy_fill() -> None:
    router = _router()

    assert router.ingest_canonical_fill(_canonical_fill()) is True
    with pytest.raises(ValueError, match="payload mismatch"):
        router.ingest_canonical_fill(_legacy_fill(digest="digest-2", qty=9.0))

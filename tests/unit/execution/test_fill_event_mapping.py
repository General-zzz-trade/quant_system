from __future__ import annotations

from decimal import Decimal

from event.types import EventType
from execution.models.fill_events import (
    build_synthetic_ingress_fill_event,
    canonical_fill_to_ingress_event,
    canonical_fill_to_public_event,
    ingress_fill_dedup_identity,
)
from execution.models.fills import CanonicalFill


def _canonical_fill() -> CanonicalFill:
    return CanonicalFill(
        venue="binance",
        symbol="BTCUSDT",
        order_id="ord-1",
        trade_id="trade-1",
        fill_id="binance:BTCUSDT:trade-1",
        side="buy",
        qty=Decimal("0.25"),
        price=Decimal("42500"),
        fee=Decimal("0.10"),
        fee_asset="USDT",
        liquidity="taker",
        ts_ms=1_704_067_200_000,
        payload_digest="digest-1",
    )


def test_canonical_fill_to_public_event_maps_minimal_fact_view() -> None:
    event = canonical_fill_to_public_event(_canonical_fill(), source="execution:test")

    assert event.event_type == EventType.FILL
    assert event.fill_id == "binance:BTCUSDT:trade-1"
    assert event.order_id == "ord-1"
    assert event.symbol == "BTCUSDT"
    assert event.qty == Decimal("0.25")
    assert event.price == Decimal("42500")
    assert event.header.event_type == EventType.FILL
    assert event.header.source == "execution:test"


def test_canonical_fill_to_ingress_event_preserves_execution_fields() -> None:
    event = canonical_fill_to_ingress_event(_canonical_fill())

    assert event.event_type == "FILL"
    assert event.symbol == "BTCUSDT"
    assert event.side == "buy"
    assert event.qty == 0.25
    assert event.price == 42500.0
    assert event.fee == 0.1
    assert event.venue == "binance"
    assert event.order_id == "ord-1"
    assert event.fill_id == "binance:BTCUSDT:trade-1"
    assert event.trade_id == "trade-1"
    assert event.payload_digest == "digest-1"
    assert event.header.event_type == "FILL"
    assert event.header.event_id is None


def test_ingress_fill_dedup_identity_prefers_payload_digest_and_fill_id() -> None:
    event = canonical_fill_to_ingress_event(_canonical_fill())

    key, digest = ingress_fill_dedup_identity(event)

    assert key == ("binance", "BTCUSDT", "binance:BTCUSDT:trade-1")
    assert digest == "digest-1"


def test_build_synthetic_ingress_fill_event_produces_stable_identity_and_digest() -> None:
    event1 = build_synthetic_ingress_fill_event(
        source="bridge",
        venue="binance",
        symbol="BTCUSDT",
        order_id="ord-1",
        identity_seed="cmd-1",
        fill_seq=1,
        side="buy",
        qty="0.25",
        price="42500",
        fee="0.1",
    )
    event2 = build_synthetic_ingress_fill_event(
        source="bridge",
        venue="binance",
        symbol="BTCUSDT",
        order_id="ord-1",
        identity_seed="cmd-1",
        fill_seq=1,
        side="buy",
        qty="0.25",
        price="42500",
        fee="0.1",
    )
    event3 = build_synthetic_ingress_fill_event(
        source="bridge",
        venue="binance",
        symbol="BTCUSDT",
        order_id="ord-1",
        identity_seed="cmd-1",
        fill_seq=2,
        side="buy",
        qty="0.25",
        price="42500",
        fee="0.1",
    )

    assert event1.fill_id == event2.fill_id
    assert event1.payload_digest == event2.payload_digest
    assert event3.fill_id != event1.fill_id

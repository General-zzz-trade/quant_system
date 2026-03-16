"""Round-trip parity tests for fill model conversions.

Verifies that all fill conversion paths preserve semantic identity:
  CanonicalFill → FillEvent (public) → no data loss on required fields
  CanonicalFill → IngressFillEvent (rich) → no data loss
  CanonicalFill → record dict → all fields preserved

This ensures the three fill representations stay aligned.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.models.fills import CanonicalFill, fill_to_record
from execution.models.fill_events import (
    canonical_fill_to_public_event,
    canonical_fill_to_ingress_event,
)


@pytest.fixture
def sample_fill() -> CanonicalFill:
    return CanonicalFill(
        venue="bybit",
        symbol="ETHUSDT",
        order_id="ord-001",
        trade_id="trd-001",
        fill_id="bybit:ETHUSDT:trd-001",
        side="sell",
        qty=Decimal("0.5"),
        price=Decimal("2100.50"),
        fee=Decimal("0.42"),
        fee_asset="USDT",
        liquidity="taker",
        ts_ms=1700000000000,
        payload_digest="abc123",
    )


class TestCanonicalFillToPublicEvent:
    """CanonicalFill → FillEvent (minimal public contract)."""

    def test_required_fields_preserved(self, sample_fill: CanonicalFill):
        event = canonical_fill_to_public_event(sample_fill)
        assert event.fill_id == sample_fill.fill_id
        assert event.order_id == sample_fill.order_id
        assert event.symbol == sample_fill.symbol
        assert event.qty == sample_fill.qty
        assert event.price == sample_fill.price
        assert event.side == sample_fill.side  # side now flows through

    def test_event_type_is_fill(self, sample_fill: CanonicalFill):
        event = canonical_fill_to_public_event(sample_fill)
        assert str(event.event_type.value).lower() == "fill"

    def test_header_has_event_id(self, sample_fill: CanonicalFill):
        event = canonical_fill_to_public_event(sample_fill)
        assert event.header.event_id is not None


class TestCanonicalFillToIngressEvent:
    """CanonicalFill → CanonicalFillIngressEvent (rich pipeline input)."""

    def test_all_fields_carried(self, sample_fill: CanonicalFill):
        ingress = canonical_fill_to_ingress_event(sample_fill)
        assert ingress.symbol == sample_fill.symbol
        assert ingress.qty == float(sample_fill.qty)
        assert ingress.price == float(sample_fill.price)
        assert ingress.fee == float(sample_fill.fee)
        assert ingress.venue == sample_fill.venue
        assert ingress.order_id == sample_fill.order_id
        assert ingress.fill_id == sample_fill.fill_id
        assert ingress.trade_id == sample_fill.trade_id
        assert ingress.payload_digest == sample_fill.payload_digest

    def test_side_preserved(self, sample_fill: CanonicalFill):
        ingress = canonical_fill_to_ingress_event(sample_fill)
        assert ingress.side == "sell"

    def test_event_type_is_fill(self, sample_fill: CanonicalFill):
        ingress = canonical_fill_to_ingress_event(sample_fill)
        assert ingress.event_type == "FILL"


class TestCanonicalFillToRecord:
    """CanonicalFill → record dict (lossless serialization)."""

    def test_all_fields_in_record(self, sample_fill: CanonicalFill):
        rec = sample_fill.to_record()
        assert rec["venue"] == "bybit"
        assert rec["symbol"] == "ETHUSDT"
        assert rec["order_id"] == "ord-001"
        assert rec["trade_id"] == "trd-001"
        assert rec["fill_id"] == "bybit:ETHUSDT:trd-001"
        assert rec["side"] == "sell"
        assert rec["qty"] == "0.5"
        assert rec["price"] == "2100.50"
        assert rec["fee"] == "0.42"
        assert rec["fee_asset"] == "USDT"
        assert rec["liquidity"] == "taker"
        assert rec["ts_ms"] == "1700000000000"
        assert rec["payload_digest"] == "abc123"

    def test_fill_to_record_helper_matches_all_fields(self, sample_fill: CanonicalFill):
        """fill_to_record() must produce identical output to to_record()."""
        rec1 = sample_fill.to_record()
        rec2 = fill_to_record(sample_fill)
        assert rec1 == rec2, f"Field mismatch: {set(rec1.items()) ^ set(rec2.items())}"

    def test_fill_to_record_duck_type_has_all_fields(self):
        """fill_to_record() on duck-typed fill should have same keys as CanonicalFill."""
        from types import SimpleNamespace
        duck = SimpleNamespace(
            venue="test", symbol="X", order_id="o1", trade_id="t1",
            fill_id="f1", side="buy", qty="1.0", price="100",
            fee="0.04", fee_asset="USDT", liquidity="taker",
            ts_ms=12345, payload_digest="abc",
        )
        rec = fill_to_record(duck)
        expected_keys = {"venue", "symbol", "order_id", "trade_id", "fill_id",
                         "side", "qty", "price", "fee", "fee_asset", "liquidity",
                         "ts_ms", "payload_digest"}
        assert set(rec.keys()) == expected_keys


class TestFillConversionConsistency:
    """Cross-path consistency: public event and ingress event agree on core fields."""

    def test_public_and_ingress_agree(self, sample_fill: CanonicalFill):
        public = canonical_fill_to_public_event(sample_fill)
        ingress = canonical_fill_to_ingress_event(sample_fill)

        assert public.symbol == ingress.symbol
        assert float(public.qty) == ingress.qty
        assert float(public.price) == ingress.price
        assert public.order_id == ingress.order_id
        assert public.fill_id == ingress.fill_id
        assert public.side == ingress.side  # side now consistent across tiers

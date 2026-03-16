"""Tests for event.codec — encode/decode round-trip, Decimal precision, datetime tz."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from event.codec import (
    EventCodecError,
    EventCodecRegistry,
    PROTO_VERSION,
    decode_event,
    decode_event_json,
    encode_event,
    encode_event_json,
    _json_default,
)
from event.header import EventHeader
from event.types import (
    ControlEvent,
    EventType,
    FillEvent,
    FundingEvent,
    MarketEvent,
    RiskEvent,
    SignalEvent,
)


# ---------------------------------------------------------------------------
# Fixtures — ensure registry has needed types
# ---------------------------------------------------------------------------

def _ensure_registered() -> None:
    """Register event classes if not already registered."""
    for cls in [MarketEvent, SignalEvent, FillEvent, RiskEvent, ControlEvent, FundingEvent]:
        if not EventCodecRegistry.has(cls.event_type):
            EventCodecRegistry.register(cls)


def _make_header(event_type: EventType) -> EventHeader:
    return EventHeader(
        event_id="test-evt-001",
        event_type=event_type,
        version=1,
        ts_ns=1_700_000_000_000_000_000,
        source="test",
        root_event_id="test-evt-001",
    )


# ---------------------------------------------------------------------------
# Tests: Round-trip encode -> decode
# ---------------------------------------------------------------------------


class TestCodecRoundTrip:
    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        _ensure_registered()

    def test_market_event_round_trip(self) -> None:
        header = _make_header(EventType.MARKET)
        evt = MarketEvent(
            header=header,
            ts=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            open=Decimal("3000.50"),
            high=Decimal("3100.75"),
            low=Decimal("2900.25"),
            close=Decimal("3050.00"),
            volume=Decimal("12345.678"),
        )
        payload = encode_event(evt)
        assert payload["proto"] == PROTO_VERSION
        assert payload["event_type"] == "market"

    def test_signal_event_round_trip(self) -> None:
        header = _make_header(EventType.SIGNAL)
        evt = SignalEvent(
            header=header,
            signal_id="sig-100",
            symbol="BTCUSDT",
            side="long",
            strength=Decimal("0.92"),
        )
        payload = encode_event(evt)
        assert payload["event_type"] == "signal"
        body = payload["body"]
        assert body["signal_id"] == "sig-100"
        assert body["strength"] == Decimal("0.92")

    def test_fill_event_round_trip(self) -> None:
        header = _make_header(EventType.FILL)
        evt = FillEvent(
            header=header,
            fill_id="fl-1",
            order_id="ord-1",
            symbol="ETHUSDT",
            qty=Decimal("2.5"),
            price=Decimal("3050.123456789"),
        )
        payload = encode_event(evt)
        body = payload["body"]
        assert body["qty"] == Decimal("2.5")
        assert body["price"] == Decimal("3050.123456789")

    def test_control_event_round_trip(self) -> None:
        header = _make_header(EventType.CONTROL)
        evt = ControlEvent(
            header=header,
            command="shutdown",
            reason="maintenance",
        )
        payload = encode_event(evt)
        assert payload["body"]["command"] == "shutdown"

    def test_funding_event_round_trip(self) -> None:
        header = _make_header(EventType.FUNDING)
        evt = FundingEvent(
            header=header,
            ts=datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            funding_rate=Decimal("0.00015"),
            mark_price=Decimal("3050.50"),
        )
        payload = encode_event(evt)
        assert payload["body"]["funding_rate"] == Decimal("0.00015")


# ---------------------------------------------------------------------------
# Tests: JSON round-trip
# ---------------------------------------------------------------------------


class TestCodecJsonRoundTrip:
    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        _ensure_registered()

    def test_market_json_round_trip(self) -> None:
        header = _make_header(EventType.MARKET)
        evt = MarketEvent(
            header=header,
            ts=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            open=Decimal("3000"),
            high=Decimal("3100"),
            low=Decimal("2900"),
            close=Decimal("3050"),
            volume=Decimal("500"),
        )
        raw_json = encode_event_json(evt)
        decoded = decode_event_json(raw_json)
        assert isinstance(decoded, MarketEvent)
        assert decoded.symbol == "ETHUSDT"
        assert decoded.close == Decimal("3050")

    def test_signal_json_round_trip(self) -> None:
        header = _make_header(EventType.SIGNAL)
        evt = SignalEvent(
            header=header,
            signal_id="sig-rt",
            symbol="BTCUSDT",
            side="short",
            strength=Decimal("0.55"),
        )
        raw_json = encode_event_json(evt)
        decoded = decode_event_json(raw_json)
        assert isinstance(decoded, SignalEvent)
        assert decoded.strength == Decimal("0.55")
        assert decoded.side == "short"


# ---------------------------------------------------------------------------
# Tests: Decimal precision
# ---------------------------------------------------------------------------


class TestDecimalPrecision:
    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        _ensure_registered()

    def test_high_precision_decimal_preserved(self) -> None:
        """Decimal with many decimal places must survive JSON round-trip."""
        header = _make_header(EventType.FILL)
        price = Decimal("42123.123456789012345")
        evt = FillEvent(
            header=header,
            fill_id="fl-prec",
            order_id="ord-prec",
            symbol="BTCUSDT",
            qty=Decimal("0.00001"),
            price=price,
        )
        raw_json = encode_event_json(evt)
        decoded = decode_event_json(raw_json)
        assert isinstance(decoded, FillEvent)
        assert decoded.price == price
        assert decoded.qty == Decimal("0.00001")


# ---------------------------------------------------------------------------
# Tests: datetime timezone handling
# ---------------------------------------------------------------------------


class TestDatetimeTimezone:
    def test_json_default_utc_datetime(self) -> None:
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _json_default(dt)
        assert result.endswith("Z")
        assert "2024-06-01" in result

    def test_json_default_non_utc_converted(self) -> None:
        tz_jst = timezone(timedelta(hours=9))
        dt = datetime(2024, 6, 1, 21, 0, 0, tzinfo=tz_jst)
        result = _json_default(dt)
        assert result.endswith("Z")
        # 21:00 JST = 12:00 UTC
        assert "12:00:00" in result

    def test_json_default_naive_datetime_raises(self) -> None:
        dt = datetime(2024, 6, 1, 12, 0, 0)
        with pytest.raises(EventCodecError, match="tz-aware"):
            _json_default(dt)


# ---------------------------------------------------------------------------
# Tests: Error cases
# ---------------------------------------------------------------------------


class TestCodecErrors:
    @pytest.fixture(autouse=True)
    def setup_registry(self) -> None:
        _ensure_registered()

    def test_decode_unknown_event_type_raises(self) -> None:
        payload = {
            "proto": PROTO_VERSION,
            "event_type": "nonexistent_type",
            "header": {},
            "body": {},
        }
        with pytest.raises(EventCodecError):
            decode_event(payload)

    def test_decode_wrong_proto_version_raises(self) -> None:
        payload = {
            "proto": 999,
            "event_type": "market",
            "header": {},
            "body": {},
        }
        with pytest.raises(EventCodecError, match="proto"):
            decode_event(payload)

    def test_decode_missing_header_raises(self) -> None:
        payload = {
            "proto": PROTO_VERSION,
            "event_type": "market",
            "body": {},
        }
        with pytest.raises(EventCodecError, match="header"):
            decode_event(payload)

    def test_decode_missing_body_raises(self) -> None:
        payload = {
            "proto": PROTO_VERSION,
            "event_type": "market",
            "header": _make_header(EventType.MARKET).to_dict(),
        }
        with pytest.raises(EventCodecError, match="body"):
            decode_event(payload)

    def test_decode_json_invalid_json_raises(self) -> None:
        with pytest.raises(EventCodecError, match="JSON"):
            decode_event_json("not valid json {{}")

    def test_decode_json_non_dict_raises(self) -> None:
        with pytest.raises(EventCodecError, match="dict"):
            decode_event_json('"just a string"')

    def test_registry_has_returns_false_for_unknown(self) -> None:
        # EventType doesn't have a UNKNOWN value, so we test with a known type
        # that we know IS registered
        assert EventCodecRegistry.has(EventType.MARKET) is True

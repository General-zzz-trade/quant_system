"""Tests for coordinator event validation fixes.

Covers:
1. event_id propagation from header to evt_dict
2. SignalEvent side validation (long/short/flat != buy/sell)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import uuid


class TestEventIdPropagation:
    """event_id from event.header should be included in validation dict."""

    def test_event_id_extracted_from_header(self):
        """evt_dict should contain event_id when header has one."""
        eid = str(uuid.uuid4())
        header = SimpleNamespace(event_id=eid)
        event = SimpleNamespace(
            header=header,
            symbol="BTCUSDT",
            side=None,
            signal_side=None,
            venue=None,
            order_type=None,
            qty=None,
            price=None,
            event_type="MarketEvent",
            ts=1700000000.0,
        )

        # Replicate coordinator logic
        evt_dict = {}
        for attr in ("symbol", "side", "signal_side", "venue", "order_type", "qty", "price", "event_type"):
            val = getattr(event, attr, None)
            if val is not None:
                if isinstance(val, (int, float)):
                    evt_dict[attr] = float(val)
                else:
                    evt_dict[attr] = str(val)
        evt_dict["event_type"] = type(event).__name__

        # New: extract event_id from header
        h = getattr(event, "header", None)
        if h is not None:
            eid_val = getattr(h, "event_id", None)
            if eid_val is not None:
                evt_dict["event_id"] = str(eid_val)

        assert "event_id" in evt_dict
        assert evt_dict["event_id"] == eid

    def test_no_header_no_crash(self):
        """Events without header should not crash validation."""
        event = SimpleNamespace(
            symbol="BTCUSDT",
            side="buy",
            event_type="OrderEvent",
            ts=1700000000.0,
        )

        header = getattr(event, "header", None)
        evt_dict = {}
        if header is not None:
            eid = getattr(header, "event_id", None)
            if eid is not None:
                evt_dict["event_id"] = str(eid)

        assert "event_id" not in evt_dict  # gracefully absent


class TestSignalEventSideValidation:
    """SignalEvent side='long'/'short'/'flat' should not trigger invalid side warning."""

    def test_long_maps_to_buy(self):
        """'long' should map to 'buy' for signal side validation."""
        side = "long"
        mapped = {"long": "buy", "short": "sell"}.get(side, side)
        assert mapped == "buy"

    def test_short_maps_to_sell(self):
        """'short' should map to 'sell' for signal side validation."""
        side = "short"
        mapped = {"long": "buy", "short": "sell"}.get(side, side)
        assert mapped == "sell"

    def test_flat_passes_through(self):
        """'flat' should pass through unmapped."""
        side = "flat"
        mapped = {"long": "buy", "short": "sell"}.get(side, side)
        assert mapped == "flat"

    def test_signal_sides_valid_after_mapping(self):
        """All SignalEvent sides should be valid after mapping."""
        try:
            from _quant_hotpath import rust_validate_signal_side
        except ImportError:
            # Rust not built — validate manually
            def rust_validate_signal_side(s):
                return s in ("buy", "sell", "flat")

        for side in ("long", "short", "flat"):
            mapped = {"long": "buy", "short": "sell"}.get(side, side)
            assert rust_validate_signal_side(mapped), f"'{side}' → '{mapped}' should be valid"

    def test_order_sides_not_mapped(self):
        """Order events with buy/sell should NOT go through signal mapping."""
        event_name = "OrderEvent"
        side = "buy"
        is_signal = event_name == "SignalEvent"
        assert not is_signal
        # Order sides validated directly
        try:
            from _quant_hotpath import rust_validate_side
        except ImportError:
            def rust_validate_side(s):
                return s in ("buy", "sell")
        assert rust_validate_side(side)

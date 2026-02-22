"""Execution adapter contract — adapters must implement required methods."""
from __future__ import annotations

from execution.adapters.base import VenueAdapter


def test_venue_adapter_protocol_exists():
    assert hasattr(VenueAdapter, "list_instruments")
    assert hasattr(VenueAdapter, "get_balances")
    assert hasattr(VenueAdapter, "get_positions")

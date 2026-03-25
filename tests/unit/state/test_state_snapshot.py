"""Tests for state snapshot — immutability, mapping proxies, properties."""
from __future__ import annotations

from datetime import datetime
from types import MappingProxyType

import pytest

from state.snapshot import StateSnapshot, _freeze_mapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(**overrides):
    """Create a minimal StateSnapshot with sensible defaults."""
    defaults = dict(
        symbol="BTCUSDT",
        ts=datetime(2025, 1, 1),
        event_id="evt-001",
        event_type="MARKET",
        bar_index=42,
        markets={"BTCUSDT": {"close": 50000}},
        positions={"BTCUSDT": {"qty": 1}, "ETHUSDT": {"qty": 2}},
        account={"balance": 100000},
    )
    defaults.update(overrides)
    return StateSnapshot.of(**defaults)


# ---------------------------------------------------------------------------
# Frozen / immutable
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_cannot_set_attribute(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.symbol = "ETHUSDT"  # type: ignore[misc]

    def test_cannot_set_bar_index(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.bar_index = 99  # type: ignore[misc]

    def test_cannot_delete_attribute(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            del snap.symbol  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MappingProxyType — prevents dict mutation
# ---------------------------------------------------------------------------

class TestMappingProxy:
    def test_markets_is_mapping_proxy(self):
        snap = _make_snapshot()
        assert isinstance(snap.markets, MappingProxyType)

    def test_positions_is_mapping_proxy(self):
        snap = _make_snapshot()
        assert isinstance(snap.positions, MappingProxyType)

    def test_markets_mutation_raises(self):
        snap = _make_snapshot()
        with pytest.raises(TypeError):
            snap.markets["NEW"] = {}  # type: ignore[index]

    def test_positions_mutation_raises(self):
        snap = _make_snapshot()
        with pytest.raises(TypeError):
            snap.positions["NEW"] = {}  # type: ignore[index]

    def test_already_frozen_mapping_not_rewrapped(self):
        proxy = MappingProxyType({"A": 1})
        result = _freeze_mapping(proxy)
        assert result is proxy  # same object, not re-wrapped


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_market_returns_own_symbol(self):
        snap = _make_snapshot(
            markets={"BTCUSDT": {"close": 50000}, "ETHUSDT": {"close": 3000}},
            symbol="BTCUSDT",
        )
        assert snap.market == {"close": 50000}

    def test_market_fallback_to_first(self):
        """If symbol not in markets, returns first value."""
        snap = _make_snapshot(
            markets={"ETHUSDT": {"close": 3000}},
            symbol="MISSING",
        )
        assert snap.market == {"close": 3000}

    def test_symbols_returns_sorted_tuple(self):
        snap = _make_snapshot(
            positions={"ETHUSDT": {}, "BTCUSDT": {}, "AAVEUSD": {}},
        )
        assert snap.symbols == ("AAVEUSD", "BTCUSDT", "ETHUSDT")

    def test_symbols_empty_positions(self):
        snap = _make_snapshot(positions={})
        assert snap.symbols == ()


# ---------------------------------------------------------------------------
# Factory (.of) + optional fields
# ---------------------------------------------------------------------------

class TestFactory:
    def test_of_sets_all_fields(self):
        snap = _make_snapshot()
        assert snap.symbol == "BTCUSDT"
        assert snap.bar_index == 42
        assert snap.event_type == "MARKET"
        assert snap.event_id == "evt-001"

    def test_optional_defaults_are_none(self):
        snap = _make_snapshot()
        assert snap.portfolio is None
        assert snap.risk is None
        assert snap.features is None

    def test_optional_fields_accepted(self):
        snap = _make_snapshot(
            portfolio={"leverage": 3.0},
            risk={"max_dd": 0.05},
            features={"rsi_14": 55.0},
        )
        assert snap.portfolio == {"leverage": 3.0}
        assert snap.risk == {"max_dd": 0.05}
        assert snap.features == {"rsi_14": 55.0}

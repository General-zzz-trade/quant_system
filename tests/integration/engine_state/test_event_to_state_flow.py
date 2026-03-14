"""Integration: market event -> pipeline -> snapshot -> state update full flow."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from engine.coordinator import CoordinatorConfig, EngineCoordinator


def _market_event(symbol: str, close: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="MARKET",
        symbol=symbol,
        open=close - 1,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=100.0,
        ts=datetime(2024, 1, 1, 0, idx),
        header=SimpleNamespace(event_id=f"e{idx}", ts=datetime(2024, 1, 1, 0, idx)),
    )


def _fill_event(symbol: str, side: str, qty: float, price: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="FILL",
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=0.0,
        realized_pnl=0.0,
        margin_change=0.0,
        ts=datetime(2024, 1, 1, 0, idx),
        header=SimpleNamespace(event_id=f"f{idx}", ts=datetime(2024, 1, 1, 0, idx)),
    )


def test_market_events_advance_state():
    """Market events flow through pipeline and update market state."""
    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        starting_balance=10000.0,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    # Emit market events
    for i in range(5):
        coord.emit(_market_event("BTCUSDT", 40000.0 + i * 100, i), actor="test")

    view = coord.get_state_view()
    assert view["event_index"] == 5
    assert view["last_snapshot"] is not None
    snap = view["last_snapshot"]
    assert snap.markets["BTCUSDT"].close == 40400.0


def test_fill_updates_position():
    """Fill events update position state through the pipeline."""
    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        starting_balance=10000.0,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    # Market event first (to establish price)
    coord.emit(_market_event("BTCUSDT", 40000.0, 0), actor="test")

    # Fill event
    coord.emit(_fill_event("BTCUSDT", "buy", 0.5, 40000.0, 1), actor="test")

    view = coord.get_state_view()
    assert view["event_index"] == 2
    pos = view["positions"].get("BTCUSDT")
    assert pos is not None
    assert float(pos.qty) > 0


def test_multi_symbol_state_isolation():
    """Events for different symbols update isolated state."""
    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT", "ETHUSDT"),
        currency="USDT",
        starting_balance=10000.0,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    coord.emit(_market_event("BTCUSDT", 40000.0, 0), actor="test")
    coord.emit(_market_event("ETHUSDT", 2500.0, 1), actor="test")

    view = coord.get_state_view()
    assert view["markets"]["BTCUSDT"].close == 40000.0
    assert view["markets"]["ETHUSDT"].close == 2500.0

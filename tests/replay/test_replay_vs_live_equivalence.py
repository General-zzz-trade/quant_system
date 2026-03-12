"""Replay vs live path equivalence."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.replay import EventReplay, ReplayConfig
from event.header import EventHeader
from event.types import EventType, MarketEvent


def _events() -> list[MarketEvent]:
    return [
        MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("10"),
        ),
        MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("101"),
            volume=Decimal("12"),
        ),
    ]


def _run_live(events: list[MarketEvent]) -> dict:
    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(symbol_default="BTCUSDT", currency="USDT")
    )
    coordinator.start()
    for event in events:
        coordinator.emit(event, actor="live")
    view = coordinator.get_state_view()
    coordinator.stop()
    return view


def _run_replay(events: list[MarketEvent]) -> dict:
    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(symbol_default="BTCUSDT", currency="USDT")
    )
    coordinator.start()
    replay = EventReplay(
        dispatcher=coordinator.dispatcher,
        source=events,
        config=ReplayConfig(strict_order=False, actor="replay"),
    )
    replay.run()
    view = coordinator.get_state_view()
    coordinator.stop()
    return view


def test_replay_vs_live_equivalence() -> None:
    events = _events()
    live_view = _run_live(events)
    replay_view = _run_replay(events)

    assert live_view["event_index"] == replay_view["event_index"] == 2
    assert live_view["market"].close == replay_view["market"].close == Decimal("101")
    assert live_view["market"].last_price == replay_view["market"].last_price == Decimal("101")

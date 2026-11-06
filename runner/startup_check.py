"""Offline native market-to-state smoke check, without models or exchange clients.

This validates installation and engine wiring. It does not validate online data,
model loading, trading decisions, order execution or business health.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import socket
from unittest.mock import patch


@contextmanager
def no_network():
    """Fail immediately on Python socket connection or DNS attempts."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network is forbidden during the offline startup check")

    with (
        patch.object(socket.socket, "connect", blocked),
        patch.object(socket.socket, "connect_ex", blocked),
        patch.object(socket, "create_connection", blocked),
        patch.object(socket, "getaddrinfo", blocked),
    ):
        yield


def check_startup() -> dict[str, object]:
    """Exercise the installed Rust extension through the real coordinator."""
    with no_network():
        from engine.coordinator import CoordinatorConfig, EngineCoordinator, EnginePhase
        from event.header import EventHeader
        from event.types import EventType, MarketEvent

        coordinator = EngineCoordinator(cfg=CoordinatorConfig(
            symbol_default="BTCUSDT", symbols=("BTCUSDT",), starting_balance=10000,
        ))
        coordinator.start()
        try:
            ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
            header = EventHeader(
                event_id="startup-check-market", event_type=EventType.MARKET,
                version=1, ts_ns=int(ts.timestamp()) * 1_000_000_000,
                source="offline-startup-check",
            )
            coordinator.emit(MarketEvent(
                header=header, ts=ts, symbol="BTCUSDT", open=Decimal("100"),
                high=Decimal("102"), low=Decimal("99"), close=Decimal("101"),
                volume=Decimal("10"),
            ), actor="startup-check")
            view = coordinator.get_state_view()
            if view["event_index"] != 1:
                raise RuntimeError("Startup check failed: synthetic market event did not advance native state")
            # RustMarketState.close is an Fd8 raw integer; close_f is its public
            # unscaled accessor. Do not compare the raw storage value to a price.
            close = Decimal(str(view["markets"]["BTCUSDT"].close_f))
            if close != Decimal("101"):
                raise RuntimeError("Startup check failed: native market state has an unexpected close")
        finally:
            coordinator.stop()
        if coordinator.phase != EnginePhase.STOPPED:
            raise RuntimeError("Startup check failed: coordinator did not stop")
        return {
            "status": "passed", "scope": "offline-native-market-to-state",
            "event_index": view["event_index"], "close": str(close),
            "models_checked": False, "exchange_checked": False,
        }

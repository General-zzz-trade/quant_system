from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar, Mapping

import pytest

from event.bootstrap import bootstrap_event_layer
from event.errors import EventFatalError, EventSecurityError
from event.security import RunMode, make_actor
from event.types import BaseEvent, MarketEvent


@dataclass(frozen=True)
class _Header:
    event_type: str
    event_id: str = "evt-1"


@dataclass(frozen=True, slots=True)
class _UnknownEvent(BaseEvent):
    event_type: ClassVar[str] = "unknown"
    payload: str = "x"

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.payload}

    @classmethod
    def from_dict(cls, *, header: Any, body: Mapping[str, Any]) -> "_UnknownEvent":
        return cls(header=header, payload=str(body.get("payload", "x")))


def test_runtime_security_allows_registered_event_types() -> None:
    runtime, store = bootstrap_event_layer()
    actor = make_actor(module="test", roles={"market"}, mode=RunMode.LIVE, source="test")
    ev = MarketEvent(
        header=_Header(event_type="market", event_id="mkt-1"),
        ts=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("1"),
    )
    runtime.emit(ev, actor=actor)
    assert store.size() == 1


def test_runtime_security_blocks_unknown_event_types() -> None:
    runtime, _ = bootstrap_event_layer()
    actor = make_actor(module="test", roles={"market"}, mode=RunMode.LIVE, source="test")
    ev = _UnknownEvent(header=_Header(event_type="unknown", event_id="u-1"))
    with pytest.raises(EventSecurityError):
        runtime.emit(ev, actor=actor)


def test_runtime_security_requires_actor_when_enabled() -> None:
    runtime, _ = bootstrap_event_layer()
    ev = MarketEvent(
        header=_Header(event_type="market", event_id="mkt-2"),
        ts=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("1"),
    )
    with pytest.raises(EventFatalError, match="actor"):
        runtime.emit(ev)

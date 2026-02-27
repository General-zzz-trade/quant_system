# event/bootstrap.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Tuple

from event.codec import EventCodecRegistry
from event.errors import EventValidationError
from event.runtime import EventRuntime
from event.schema import EventSchema, FieldSpec, SchemaRegistry
from event.store import EventStore, InMemoryEventStore
from event.types import (
    ControlEvent,
    EventType,
    FillEvent,
    IntentEvent,
    MarketEvent,
    OrderEvent,
    RiskEvent,
    SignalEvent,
)


# ============================================================
# Validators (institution-grade)
# ============================================================

def _dec_ge0(v: Decimal) -> None:
    if v < Decimal("0"):
        raise EventValidationError(f"decimal must be >= 0, got {v}")


def _dec_gt0(v: Decimal) -> None:
    if v <= Decimal("0"):
        raise EventValidationError(f"decimal must be > 0, got {v}")


def _ohlc_invariants(body, _event) -> None:
    """OHLC 合法性（制度级）"""
    o = body["open"]
    h = body["high"]
    l = body["low"]
    c = body["close"]
    v = body["volume"]

    if not all(isinstance(x, Decimal) for x in (o, h, l, c, v)):
        raise EventValidationError("OHLC invariant expects Decimal fields")

    if h < max(o, c, l):
        raise EventValidationError(f"invalid OHLC: high({h}) < max(open,close,low)")
    if l > min(o, c, h):
        raise EventValidationError(f"invalid OHLC: low({l}) > min(open,close,high)")
    if v < Decimal("0"):
        raise EventValidationError(f"invalid volume: {v} < 0")


def _order_price_invariants(body, _event) -> None:
    price = body.get("price")
    if price is None:
        return
    if not isinstance(price, Decimal):
        raise EventValidationError("order.price must be Decimal or None")
    if price <= Decimal("0"):
        raise EventValidationError(f"order.price must be > 0, got {price}")


# ============================================================
# Bootstrap
# ============================================================

def bootstrap_event_layer(store: "EventStore | None" = None) -> Tuple[EventRuntime, "EventStore"]:
    """Bootstrap a minimal event layer.

    说明：这是最小可用启动器（适用于回测/开发）。
    实盘部署时建议替换为 durable EventStore（SQLite/WAL/Kafka 等）。

    Args:
        store: Optional EventStore override. Defaults to InMemoryEventStore.
    """

    schemas = SchemaRegistry()

    # -------- MARKET v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.MARKET,
            version=MarketEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("ts", types=(datetime,), tz_aware=True),
                FieldSpec("symbol", types=(str,), min_len=1, max_len=64),
                FieldSpec("open", types=(Decimal,), validator=_dec_ge0),
                FieldSpec("high", types=(Decimal,), validator=_dec_ge0),
                FieldSpec("low", types=(Decimal,), validator=_dec_ge0),
                FieldSpec("close", types=(Decimal,), validator=_dec_ge0),
                FieldSpec("volume", types=(Decimal,), validator=_dec_ge0),
            ),
            post_validators=(_ohlc_invariants,),
        )
    )

    # -------- SIGNAL v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.SIGNAL,
            version=SignalEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("signal_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("symbol", types=(str,), min_len=1, max_len=64),
                FieldSpec("side", types=(str,), enum=("long", "short")),
                FieldSpec("strength", types=(Decimal,), validator=_dec_ge0),
            ),
        )
    )

    # -------- INTENT v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.INTENT,
            version=IntentEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("intent_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("symbol", types=(str,), min_len=1, max_len=64),
                FieldSpec("side", types=(str,), enum=("buy", "sell")),
                FieldSpec("target_qty", types=(Decimal,), validator=_dec_gt0),
                FieldSpec("reason_code", types=(str,), enum=("signal", "rebalance", "risk", "manual")),
                FieldSpec("origin", types=(str,), min_len=1, max_len=256),
            ),
        )
    )

    # -------- ORDER v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.ORDER,
            version=OrderEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("order_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("intent_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("symbol", types=(str,), min_len=1, max_len=64),
                FieldSpec("side", types=(str,), enum=("buy", "sell")),
                FieldSpec("qty", types=(Decimal,), validator=_dec_gt0),
                FieldSpec("price", types=(Decimal,), allow_none=True),
            ),
            post_validators=(_order_price_invariants,),
        )
    )

    # -------- FILL v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.FILL,
            version=FillEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("fill_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("order_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("symbol", types=(str,), min_len=1, max_len=64),
                FieldSpec("qty", types=(Decimal,), validator=_dec_gt0),
                FieldSpec("price", types=(Decimal,), validator=_dec_gt0),
            ),
        )
    )

    # -------- RISK v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.RISK,
            version=RiskEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("rule_id", types=(str,), min_len=1, max_len=128),
                FieldSpec("level", types=(str,), enum=("info", "warn", "block")),
                FieldSpec("message", types=(str,), min_len=1, max_len=2048),
            ),
        )
    )

    # -------- CONTROL v1 --------
    schemas.register(
        EventSchema(
            event_type=EventType.CONTROL,
            version=ControlEvent.VERSION,
            allow_extra_fields=False,
            fields=(
                FieldSpec("command", types=(str,), enum=("halt", "resume", "flush", "shutdown")),
                FieldSpec("reason", types=(str,), min_len=1, max_len=1024),
            ),
        )
    )

    # -------- codec registry (for durable logs / replay) --------
    for cls in (MarketEvent, SignalEvent, IntentEvent, OrderEvent, FillEvent, RiskEvent, ControlEvent):
        if not EventCodecRegistry.has(cls.event_type):
            EventCodecRegistry.register(cls)

    if store is None:
        store = InMemoryEventStore()

    runtime = EventRuntime(
        schema_registry=schemas,
        store=store,
        tracer=None,
        metrics=None,
        policy=None,
    )

    return runtime, store

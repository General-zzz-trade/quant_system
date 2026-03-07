"""Shared Hypothesis strategies for quant system property-based tests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from hypothesis import strategies as st
from hypothesis.strategies import composite

from engine.saga import SagaState, _TRANSITIONS


# ── Primitives ────────────────────────────────────────────

prices = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("200000"), allow_nan=False, allow_infinity=False)
quantities = st.decimals(min_value=Decimal("0.001"), max_value=Decimal("10000"), allow_nan=False, allow_infinity=False)
sides = st.sampled_from(["buy", "sell"])
symbols = st.sampled_from(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
utc_timestamps = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 1, 1),
    timezones=st.just(timezone.utc),
)


# ── Event stubs (duck-typed for reducers) ─────────────────

@dataclass
class FillEventStub:
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    event_type: str = "fill"
    ts: Optional[datetime] = None
    fee: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    cash_delta: Optional[Decimal] = None
    margin_change: Optional[Decimal] = None


@dataclass
class MarketEventStub:
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    event_type: str = "market"
    ts: Optional[datetime] = None


@composite
def fill_events(draw, symbol="BTCUSDT"):
    """Generate valid fill event stubs for reducers."""
    return FillEventStub(
        symbol=symbol,
        side=draw(sides),
        qty=draw(quantities),
        price=draw(prices),
        ts=draw(utc_timestamps),
        fee=draw(st.one_of(st.none(), st.decimals(
            min_value=Decimal("0"), max_value=Decimal("100"),
            allow_nan=False, allow_infinity=False,
        ))),
        realized_pnl=draw(st.one_of(st.none(), st.decimals(
            min_value=Decimal("-10000"), max_value=Decimal("10000"),
            allow_nan=False, allow_infinity=False,
        ))),
    )


@composite
def market_events(draw, symbol="BTCUSDT"):
    """Generate valid market bar event stubs for reducers."""
    close = draw(prices)
    low = draw(st.decimals(
        min_value=Decimal("0.01"), max_value=close,
        allow_nan=False, allow_infinity=False,
    ))
    high = draw(st.decimals(
        min_value=close, max_value=Decimal("200000"),
        allow_nan=False, allow_infinity=False,
    ))
    open_p = draw(st.decimals(
        min_value=low, max_value=high,
        allow_nan=False, allow_infinity=False,
    ))
    return MarketEventStub(
        symbol=symbol,
        open=open_p,
        high=high,
        low=low,
        close=close,
        volume=draw(st.decimals(
            min_value=Decimal("0"), max_value=Decimal("1000000"),
            allow_nan=False, allow_infinity=False,
        )),
        ts=draw(utc_timestamps),
    )


# ── Envelope strategies ──────────────────────────────────

@composite
def envelopes(draw):
    """Generate valid Envelope objects for bus testing."""
    from core.types import Envelope, EventKind, EventMetadata, Priority, TraceContext

    kind = draw(st.sampled_from(list(EventKind)))
    priority = draw(st.sampled_from(list(Priority)))

    return Envelope(
        event={"type": kind.name, "data": "test"},
        metadata=EventMetadata.create(source="test"),
        kind=kind,
        priority=priority,
    )


# ── Saga transition sequences ────────────────────────────

@composite
def valid_saga_transition_sequences(draw, max_length=10):
    """Generate valid sequences of SagaState transitions starting from PENDING."""
    state = SagaState.PENDING
    sequence = []
    for _ in range(max_length):
        allowed = _TRANSITIONS.get(state, frozenset())
        if not allowed:
            break
        next_state = draw(st.sampled_from(sorted(allowed, key=lambda s: s.value)))
        sequence.append(next_state)
        state = next_state
    return sequence

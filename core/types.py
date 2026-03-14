"""Algebraic type system for the quant system.

Goals:
  1. Make illegal states unrepresentable at the type level.
  2. Replace ``Any`` in hot paths with concrete generic types.
  3. Provide a typed Envelope[E] that carries metadata + trace context.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Generic,
    Optional,
    Tuple,
    TypeVar,
)

# ── Event type discriminator ─────────────────────────────

class EventKind(Enum):
    """Exhaustive set of event categories — used for routing."""
    MARKET = auto()
    SIGNAL = auto()
    INTENT = auto()
    ORDER = auto()
    FILL = auto()
    RISK = auto()
    CONTROL = auto()


class Route(Enum):
    """Dispatcher routing destinations."""
    PIPELINE = "pipeline"
    DECISION = "decision"
    EXECUTION = "execution"
    DROP = "drop"


class Priority(Enum):
    """Event priority for bounded bus ordering."""
    CRITICAL = 0   # kill-switch, emergency reduce
    HIGH = 1       # fills, order updates
    NORMAL = 2     # market data, signals
    LOW = 3        # monitoring, audit


# ── Trace context (W3C-style) ────────────────────────────

@dataclass(frozen=True, slots=True)
class TraceContext:
    """Distributed trace context carried through the full event chain.

    Every event inherits the ``trace_id`` of its root cause, enabling
    full-chain forensics: MarketEvent → Signal → Decision → Order → Fill.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def new_root(cls) -> TraceContext:
        tid = uuid.uuid4().hex
        sid = uuid.uuid4().hex[:16]
        return cls(trace_id=tid, span_id=sid)

    def child_span(self) -> TraceContext:
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            baggage=self.baggage,
        )


# ── Event metadata ───────────────────────────────────────

@dataclass(frozen=True, slots=True)
class EventMetadata:
    """Immutable metadata attached to every event in the system."""
    event_id: str
    timestamp: datetime
    trace: TraceContext
    source: str                          # e.g. "binance_ws", "backtest", "decision"
    causation_id: Optional[str] = None   # event_id of the direct cause
    correlation_id: Optional[str] = None # shared id across a logical operation
    sequence: int = 0                    # monotonic within a source

    @classmethod
    def create(
        cls,
        *,
        source: str,
        trace: Optional[TraceContext] = None,
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        sequence: int = 0,
    ) -> EventMetadata:
        return cls(
            event_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            trace=trace or TraceContext.new_root(),
            source=source,
            causation_id=causation_id,
            correlation_id=correlation_id,
            sequence=sequence,
        )


# ── Generic Envelope ─────────────────────────────────────

E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Envelope(Generic[E]):
    """Type-safe event wrapper — the only object that flows through the bus.

    ``E`` is the concrete event type (``MarketEvent``, ``OrderEvent``, etc.).
    The ``kind`` field enables O(1) routing without isinstance checks.
    """
    event: E
    metadata: EventMetadata
    kind: EventKind
    priority: Priority = Priority.NORMAL

    @property
    def event_id(self) -> str:
        return self.metadata.event_id

    @property
    def trace_id(self) -> str:
        return self.metadata.trace.trace_id

    @property
    def timestamp(self) -> datetime:
        return self.metadata.timestamp


# ── Typed identifiers (prevent mixing str IDs) ──────────

@dataclass(frozen=True, slots=True)
class Symbol:
    """Normalized trading symbol — venue-independent."""
    base: str     # e.g. "BTC"
    quote: str    # e.g. "USDT"

    @property
    def canonical(self) -> str:
        return f"{self.base}{self.quote}"

    def __str__(self) -> str:
        return self.canonical

    @classmethod
    def parse(cls, raw: str) -> Symbol:
        """Parse common formats: BTCUSDT, BTC/USDT, BTC-USDT."""
        clean = raw.upper().replace("/", "").replace("-", "")
        # Try common quote currencies
        for q in ("USDT", "BUSD", "USD", "USDC", "BTC", "ETH"):
            if clean.endswith(q) and len(clean) > len(q):
                return cls(base=clean[:-len(q)], quote=q)
        raise ValueError(f"Cannot parse symbol: {raw!r}")


@dataclass(frozen=True, slots=True)
class VenueSymbol:
    """Venue-specific symbol — used at the execution boundary."""
    venue: str
    symbol: Symbol
    raw: str       # venue's native format, e.g. "BTCUSDT" for Binance

    def __str__(self) -> str:
        return f"{self.venue}:{self.raw}"

"""Event classes — thin Python wrappers around Rust-backed implementations.

Each event class delegates storage to the corresponding Rust type from
_quant_hotpath while maintaining full backward compatibility with existing
Python callers (Decimal fields, header, VERSION, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, Optional
from enum import Enum
from datetime import datetime

from _quant_hotpath import (
    RustMarketEvent as _RustMarketEvent,
    RustFillEvent as _RustFillEvent,
    RustFundingEvent as _RustFundingEvent,
    RustSignalEvent as _RustSignalEvent,
    RustIntentEvent as _RustIntentEvent,
    RustOrderEvent as _RustOrderEvent,
    RustRiskEvent as _RustRiskEvent,
    RustControlEvent as _RustControlEvent,
    RustEventHeader as _RustEventHeader,
)


# ============================================================
# EventType
# ============================================================

class EventType(Enum):
    MARKET = "market"
    SIGNAL = "signal"
    INTENT = "intent"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    CONTROL = "control"
    FUNDING = "funding"


# ============================================================
# BaseEvent — base class for all event types
# ============================================================

@dataclass(frozen=True, slots=True)
class BaseEvent:
    """BaseEvent — base for all event types.

    Concrete subclasses delegate data to Rust PyO3 objects stored in
    ``_rust``.  The Python wrapper exposes identical attribute names so
    all existing callers work unchanged.
    """

    event_type: ClassVar[EventType]
    header: Any

    VERSION: ClassVar[int] = 1

    @property
    def version(self) -> int:
        return int(self.__class__.VERSION)

    def to_rust(self) -> Any:
        """Return the underlying Rust event object (if available)."""
        return getattr(self, "_rust", None)


# ============================================================
# MarketEvent — wraps _RustMarketEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class MarketEvent(BaseEvent):
    """MarketEvent — kline/bar event, Rust-backed."""

    event_type: ClassVar[EventType] = EventType.MARKET
    VERSION: ClassVar[int] = 1

    ts: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    _rust: Any = None  # _RustMarketEvent cache

    def __post_init__(self) -> None:
        if self._rust is None:
            ts_str = self.ts.isoformat() if self.ts else None
            rust = _RustMarketEvent(
                symbol=str(self.symbol),
                open=float(self.open),
                high=float(self.high),
                low=float(self.low),
                close=float(self.close),
                volume=float(self.volume),
                ts=ts_str,
            )
            object.__setattr__(self, "_rust", rust)


# ============================================================
# SignalEvent — wraps _RustSignalEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class SignalEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.SIGNAL

    signal_id: str
    symbol: str
    side: str  # "long" | "short"
    strength: Decimal

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            rust_header = _to_rust_header(self.header)
            if rust_header is not None:
                rust = _RustSignalEvent(
                    header=rust_header,
                    signal_id=str(self.signal_id),
                    symbol=str(self.symbol),
                    side=str(self.side),
                    strength=float(self.strength),
                )
                object.__setattr__(self, "_rust", rust)


# ============================================================
# IntentEvent — wraps _RustIntentEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class IntentEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.INTENT

    intent_id: str
    symbol: str
    side: str          # "buy" | "sell"
    target_qty: Decimal
    reason_code: str   # signal | rebalance | risk | manual
    origin: str        # strategy_id / model_version

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            rust_header = _to_rust_header(self.header)
            if rust_header is not None:
                rust = _RustIntentEvent(
                    header=rust_header,
                    intent_id=str(self.intent_id),
                    symbol=str(self.symbol),
                    side=str(self.side),
                    target_qty=float(self.target_qty),
                    reason_code=str(self.reason_code),
                    origin=str(self.origin),
                )
                object.__setattr__(self, "_rust", rust)


# ============================================================
# OrderEvent — wraps _RustOrderEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.ORDER

    order_id: str
    intent_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal | None

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            rust_header = _to_rust_header(self.header)
            if rust_header is not None:
                rust = _RustOrderEvent(
                    header=rust_header,
                    order_id=str(self.order_id),
                    intent_id=str(self.intent_id),
                    symbol=str(self.symbol),
                    side=str(self.side),
                    qty=float(self.qty),
                    price=float(self.price) if self.price is not None else None,
                )
                object.__setattr__(self, "_rust", rust)


# ============================================================
# FillEvent — wraps _RustFillEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class FillEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.FILL

    fill_id: str
    order_id: str
    symbol: str
    qty: Decimal
    price: Decimal
    side: Optional[str] = None  # "buy"/"sell" — optional for backward compat

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            rust = _RustFillEvent(
                symbol=str(self.symbol),
                side=str(self.side) if self.side else "buy",
                qty=float(self.qty),
                price=float(self.price),
            )
            object.__setattr__(self, "_rust", rust)


# ============================================================
# RiskEvent — wraps _RustRiskEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class RiskEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.RISK

    rule_id: str
    level: str          # info | warn | block
    message: str

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            rust_header = _to_rust_header(self.header)
            if rust_header is not None:
                rust = _RustRiskEvent(
                    header=rust_header,
                    rule_id=str(self.rule_id),
                    level=str(self.level),
                    message=str(self.message),
                )
                object.__setattr__(self, "_rust", rust)


# ============================================================
# ControlEvent — wraps _RustControlEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class ControlEvent(BaseEvent):
    event_type: ClassVar[EventType] = EventType.CONTROL

    command: str    # halt / reduce_only / resume / flush / shutdown
    reason: str

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            rust_header = _to_rust_header(self.header)
            if rust_header is not None:
                rust = _RustControlEvent(
                    header=rust_header,
                    command=str(self.command),
                    reason=str(self.reason),
                )
                object.__setattr__(self, "_rust", rust)


# ============================================================
# FundingEvent — wraps _RustFundingEvent
# ============================================================

@dataclass(frozen=True, slots=True)
class FundingEvent(BaseEvent):
    """Funding rate settlement event for perpetual futures.

    Injected at 00:00/08:00/16:00 UTC for Binance perpetuals.
    Settlement amount = position_qty * mark_price * funding_rate
    Positive rate: longs pay shorts. Negative rate: shorts pay longs.
    """

    event_type: ClassVar[EventType] = EventType.FUNDING
    VERSION: ClassVar[int] = 1

    ts: datetime
    symbol: str
    funding_rate: Decimal       # e.g. Decimal("0.0001") = 1 bps
    mark_price: Decimal         # mark price at settlement time

    _rust: Any = None

    def __post_init__(self) -> None:
        if self._rust is None:
            ts_str = self.ts.isoformat() if self.ts else None
            rust = _RustFundingEvent(
                symbol=str(self.symbol),
                funding_rate=float(self.funding_rate),
                mark_price=float(self.mark_price),
                position_qty=0.0,
                ts=ts_str,
            )
            object.__setattr__(self, "_rust", rust)


# ============================================================
# Helper: convert Python EventHeader to RustEventHeader
# ============================================================

def _to_rust_header(header: Any) -> Optional[_RustEventHeader]:
    """Convert a Python EventHeader (or compatible) to RustEventHeader.

    If the header is already a RustEventHeader, return it directly.
    Returns None if the header lacks required fields (e.g. test mocks).
    """
    if isinstance(header, _RustEventHeader):
        return header
    # Duck-type from Python EventHeader -- require minimum fields
    try:
        event_id = str(header.event_id)
        et = header.event_type
        event_type_str = et.value if isinstance(et, EventType) else str(et)
        version = int(getattr(header, "version", 1))
        ts_ns = int(getattr(header, "ts_ns", 0))
        source = str(getattr(header, "source", "unknown"))
        return _RustEventHeader(
            event_id=event_id,
            event_type=event_type_str,
            version=version,
            ts_ns=ts_ns,
            source=source,
            parent_event_id=getattr(header, "parent_event_id", None),
            root_event_id=getattr(header, "root_event_id", None),
            run_id=getattr(header, "run_id", None),
            seq=getattr(header, "seq", None),
            correlation_id=getattr(header, "correlation_id", None),
        )
    except (AttributeError, TypeError):
        return None

"""Domain value types used across the event system and risk rules."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class Side(str, Enum):
    """Trading side: BUY (long) or SELL (short)."""
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Symbol:
    """Normalized trading symbol (venue-independent)."""
    value: str  # e.g. "BTCUSDT"

    @property
    def normalized(self) -> str:
        return self.value.upper()

    def __str__(self) -> str:
        return self.normalized


class Venue(str, Enum):
    """Supported trading venues."""
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    SIM = "SIM"


@dataclass(frozen=True, slots=True)
class Qty:
    """Quantity wrapper with Decimal value."""
    value: Decimal

    @classmethod
    def of(cls, v: Any) -> "Qty":
        return cls(value=Decimal(str(v)))


@dataclass(frozen=True, slots=True)
class Price:
    """Price wrapper with Decimal value."""
    value: Decimal

    @classmethod
    def of(cls, v: Any) -> "Price":
        return cls(value=Decimal(str(v)))


@dataclass(frozen=True, slots=True)
class Money:
    """Monetary amount wrapper."""
    amount: Decimal
    currency: str = "USDT"

    @classmethod
    def of(cls, v: Any, currency: str = "USDT") -> "Money":
        return cls(amount=Decimal(str(v)), currency=currency)


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Time in force."""
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"

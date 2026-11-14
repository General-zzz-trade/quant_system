"""What the exchange reports: fills, positions and balances, as immutable Decimal value objects.

Moved unchanged from execution/models (migration step 8); the old modules re-export these classes, so
legacy isinstance checks see the same class objects.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, TypedDict


@dataclass(frozen=True, slots=True)
class CanonicalFill:
    """CanonicalFill — execution layer's canonical fill fact (Tier 1).

    Three-tier fill hierarchy:
    - Tier 1: CanonicalFill (this) — Decimal precision, 12 fields, internal truth
    - Tier 2: CanonicalFillIngressEvent (fill_events.py) — float, pipeline input
    - Tier 3: FillEvent (event/types.py) — Decimal, 5 fields, public event bus

    Mapping: CanonicalFill → Tier 2 via canonical_fill_to_ingress_event()
             CanonicalFill → Tier 3 via canonical_fill_to_public_event()
    Round-trip parity locked by tests/unit/execution/test_fill_roundtrip.py.

    Key invariants:
    - fill_id MUST be stable (same real fill → same fill_id across duplicates)
    - payload_digest detects "same fill_id, different payload" (data corruption)
    """
    venue: str
    symbol: str

    order_id: str
    trade_id: str
    fill_id: str

    side: str              # "buy" / "sell"
    qty: Decimal           # base qty, >0
    price: Decimal         # >0

    fee: Decimal = Decimal("0")
    fee_asset: Optional[str] = None
    liquidity: Optional[str] = None  # "maker" / "taker" / None

    ts_ms: int = 0         # epoch ms
    payload_digest: str = ""

    raw: Optional[Mapping[str, Any]] = None

    def to_record(self) -> dict[str, str]:
        """Lossless string-serialized record for fill history tracking.

        All fields preserved as strings. Use this instead of ad-hoc dicts.
        """
        return {
            "venue": self.venue,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "trade_id": self.trade_id,
            "fill_id": self.fill_id,
            "side": self.side,
            "qty": str(self.qty),
            "price": str(self.price),
            "fee": str(self.fee),
            "fee_asset": self.fee_asset or "",
            "liquidity": self.liquidity or "",
            "ts_ms": str(self.ts_ms),
            "payload_digest": self.payload_digest,
        }


@dataclass(frozen=True, slots=True)
class VenuePosition:
    """
    交易所报告的仓位快照。

    与 state/position.py 的 PositionState（内部SSOT）不同，
    VenuePosition 是交易所原始上报数据，用于对账。
    """
    venue: str
    symbol: str

    qty: Decimal                   # 有符号：正=多，负=空
    entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None

    unrealized_pnl: Decimal = Decimal("0")
    leverage: Optional[int] = None
    margin_type: Optional[str] = None   # "cross" / "isolated"

    ts_ms: int = 0
    raw: Optional[Mapping[str, Any]] = None

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def abs_qty(self) -> Decimal:
        return abs(self.qty)

    @property
    def side(self) -> str:
        if self.qty > 0:
            return "buy"
        elif self.qty < 0:
            return "sell"
        return "flat"


@dataclass(frozen=True, slots=True)
class VenueOrder:
    """An order as the exchange reports it. ``state`` is one of the order lifecycle's words."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str                      # "buy" / "sell"
    state: str                     # new | partially_filled | filled | canceled | expired | rejected
    filled_qty: Decimal
    avg_price: Decimal


@dataclass(frozen=True, slots=True)
class OpenOrder:
    """A working order in the exchange's open-order snapshot."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    type: str


class Placement(TypedDict, total=False):
    """What placing an order returned. ``status`` is always present; anything but ``submitted`` or
    ``rejected`` means the outcome is unknown and must be looked up, never resent."""

    status: str                    # submitted | rejected | error
    orderId: str
    clientOrderId: str
    code: int | None               # the exchange's refusal code, for logs only
    msg: str


class StopPlacement(TypedDict, total=False):
    """What placing a protective stop returned."""

    status: str                    # submitted | rejected | error
    algoId: str
    code: int | None               # for logs only
    msg: str
    breached: bool                 # refused because price is already beyond the trigger


class Ticker(TypedDict, total=False):
    symbol: str
    lastPrice: float
    bid1Price: float
    ask1Price: float


# The three states of an exchange-side protective stop that the application acts on.
STOP_ACTIVE = "active"             # working at the exchange
STOP_TRIGGERED = "triggered"       # fired; ``actual_order_id`` names the order it became, once known
STOP_GONE = "gone"                 # canceled, expired, rejected or in a state this code does not know


@dataclass(frozen=True, slots=True)
class StopOrder:
    """A protective stop as the exchange reports it."""

    algo_id: str
    client_algo_id: str
    symbol: str
    side: str
    status: str                    # the exchange's own word, for logs only
    state: str                     # STOP_ACTIVE | STOP_TRIGGERED | STOP_GONE
    trigger_price: Decimal
    actual_order_id: str


@dataclass(frozen=True, slots=True)
class CanonicalBalance:
    """
    标准化余额快照 — 交易所返回的资产余额。

    free:   可用于下单的数量
    locked: 被挂单或仓位占用的数量
    total:  free + locked
    """
    venue: str
    asset: str

    free: Decimal
    locked: Decimal
    total: Decimal

    ts_ms: int = 0
    raw: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_free_locked(
        cls,
        *,
        venue: str,
        asset: str,
        free: Decimal,
        locked: Decimal,
        ts_ms: int = 0,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> CanonicalBalance:
        return cls(
            venue=venue,
            asset=asset.upper(),
            free=free,
            locked=locked,
            total=free + locked,
            ts_ms=ts_ms,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class BalanceSnapshot:
    """一次完整的账户余额快照（多个资产）。"""
    venue: str
    balances: tuple[CanonicalBalance, ...]
    ts_ms: int = 0

    def get(self, asset: str) -> Optional[CanonicalBalance]:
        asset_upper = asset.upper()
        for b in self.balances:
            if b.asset == asset_upper:
                return b
        return None

    @property
    def assets(self) -> Sequence[str]:
        return [b.asset for b in self.balances]

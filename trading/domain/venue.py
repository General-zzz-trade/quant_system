"""What the exchange reports: fills, positions and balances, as immutable Decimal value objects.

Moved unchanged from execution/models (migration step 8); the old modules re-export these classes, so
legacy isinstance checks see the same class objects.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional


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

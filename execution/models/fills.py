# execution/models/fills.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional


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


def fill_to_record(fill: Any) -> dict[str, str]:
    """Convert any fill-like object to a standard record dict.

    Works with CanonicalFill, SimpleNamespace, or any duck-typed fill.
    Fields aligned with CanonicalFill.to_record() — same 12 fields.
    Use this to replace all ad-hoc fill dict construction.
    """
    if isinstance(fill, CanonicalFill):
        return fill.to_record()
    return {
        "venue": str(getattr(fill, "venue", "")),
        "symbol": str(getattr(fill, "symbol", "")),
        "order_id": str(getattr(fill, "order_id", "")),
        "trade_id": str(getattr(fill, "trade_id", "")),
        "fill_id": str(getattr(fill, "fill_id", "")),
        "side": str(getattr(fill, "side", "")),
        "qty": str(getattr(fill, "qty", "")),
        "price": str(getattr(fill, "price", "")),
        "fee": str(getattr(fill, "fee", "")),
        "fee_asset": str(getattr(fill, "fee_asset", "") or ""),
        "liquidity": str(getattr(fill, "liquidity", "") or ""),
        "ts_ms": str(getattr(fill, "ts_ms", "")),
        "payload_digest": str(getattr(fill, "payload_digest", "")),
    }

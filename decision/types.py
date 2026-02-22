from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence, Literal

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
SignalSide = Literal["buy", "sell", "flat"]


@dataclass(frozen=True, slots=True)
class SignalResult:
    symbol: str
    side: SignalSide
    score: Decimal  # signed strength; >0 buy, <0 sell
    confidence: Decimal = Decimal("1")
    meta: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    score: Decimal
    side: Side
    meta: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True, slots=True)
class TargetPosition:
    symbol: str
    target_qty: Decimal  # signed target position qty
    reason_code: str = "signal"
    origin: str = "decision"


@dataclass(frozen=True, slots=True)
class OrderSpec:
    """Execution-neutral order specification."""
    order_id: str
    intent_id: str
    symbol: str
    side: Side
    qty: Decimal
    order_type: OrderType = "limit"
    price: Optional[Decimal] = None
    tif: str = "GTC"
    meta: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True, slots=True)
class DecisionExplain:
    """Stable explain schema (must remain JSON-serializable)."""
    ts: datetime
    strategy_id: str
    gates: Dict[str, Any]
    universe: Sequence[str]
    signals: Sequence[Mapping[str, Any]]
    candidates: Sequence[Mapping[str, Any]]
    targets: Sequence[Mapping[str, Any]]
    orders: Sequence[Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # datetime -> iso for stable json
        d["ts"] = self.ts.isoformat()
        return d


@dataclass(frozen=True, slots=True)
class DecisionOutput:
    ts: datetime
    strategy_id: str
    targets: Sequence[TargetPosition]
    orders: Sequence[OrderSpec]
    explain: DecisionExplain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "strategy_id": self.strategy_id,
            "targets": [asdict(t) for t in self.targets],
            "orders": [
                {
                    **{k: v for k, v in asdict(o).items() if k != "meta"},
                    "meta": dict(o.meta) if o.meta is not None else None,
                }
                for o in self.orders
            ],
            "explain": self.explain.to_dict(),
        }

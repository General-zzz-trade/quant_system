"""Confirmed position projection, independent of signals and exchange requests."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class StrategyPosition:
    qty: Decimal = Decimal(0)
    entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    fees: dict[str, Decimal] = field(default_factory=dict)

    def apply(self, fact: Any) -> None:
        qty, price = Decimal(str(fact.qty)), Decimal(str(fact.price))
        if fact.side not in ("buy", "sell") or not qty.is_finite() or qty <= 0 or not price.is_finite() or price <= 0:
            raise ValueError("Invalid confirmed fill")
        fee = Decimal(str(getattr(fact, "fee", 0) or 0))
        if not fee.is_finite():
            raise ValueError("Invalid fee")
        signed = qty if fact.side == "buy" else -qty
        old, new = self.qty, self.qty + signed
        if old == 0 or old * signed > 0:
            self.entry_price = (abs(old) * self.entry_price + qty * price) / abs(new)
        else:
            self.realized_pnl += min(abs(old), qty) * (price - self.entry_price) * (1 if old > 0 else -1)
            if new == 0:
                self.entry_price = Decimal(0)
            elif new * old < 0:
                self.entry_price = price
        self.qty = new
        # Preserve exchange sign and currency; do not mix USDT and coin fees.
        asset = str(getattr(fact, "fee_asset", None) or "unknown")
        self.fees[asset] = self.fees.get(asset, Decimal(0)) + fee

    def to_dict(self) -> dict[str, Any]:
        return {"qty": str(self.qty), "entry_price": str(self.entry_price),
                "realized_pnl": str(self.realized_pnl), "fees": {k: str(v) for k, v in self.fees.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyPosition:
        return cls(Decimal(data["qty"]), Decimal(data["entry_price"]), Decimal(data["realized_pnl"]),
                   {k: Decimal(v) for k, v in data.get("fees", {}).items()})

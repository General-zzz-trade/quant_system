from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from _quant_hotpath import rust_limit_price_f64 as _rust_limit_price_f64

from decision.market_access import get_decimal_attr
from state.snapshot import StateSnapshot
from decision.types import OrderSpec


@dataclass(frozen=True, slots=True)
class MarketableLimitPolicy:
    slippage_bps: Decimal = Decimal("10")  # 10 bps
    name: str = "marketable_limit"

    def apply(self, snapshot: StateSnapshot, order: OrderSpec) -> OrderSpec:
        m = snapshot.market
        px = get_decimal_attr(m, "close", "last_price")
        if px is None:
            raise ValueError("No market price available for marketable limit policy")
        price_f = _rust_limit_price_f64(order.side, float(px), float(self.slippage_bps), True)
        price = Decimal(price_f)
        return replace(order, price=price, order_type="limit")

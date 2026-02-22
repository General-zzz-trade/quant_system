from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence, Mapping


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    # universe
    symbols: Optional[Sequence[str]] = None  # None => infer from snapshot (symbol + positions)

    # sizing / allocation
    max_positions: int = 1
    risk_fraction: Decimal = Decimal("0.02")  # fraction of equity to allocate per rebalance
    min_notional: Decimal = Decimal("5")      # avoid dust orders
    min_qty: Decimal = Decimal("0")           # optional venue-specific min qty

    # leverage / exposure guard (soft guard; hard guard should remain in risk layer)
    max_leverage: Decimal = Decimal("5")
    allow_short: bool = True

    # execution preferences
    execution_policy: str = "marketable_limit"  # marketable_limit | passive
    price_slippage_bps: Decimal = Decimal("10") # 10 bps
    tif: str = "GTC"

    # metadata
    strategy_id: str = "decision.default"
    origin: str = "decision"

    # feature flags
    feature_flags: Mapping[str, bool] = field(default_factory=dict)

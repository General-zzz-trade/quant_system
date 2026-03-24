"""VolTargetAllocator -- volatility target allocation strategy.

Extracted from allocator.py to keep it under 500 lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from portfolio.allocator import (
    AllocatorError,
    AllocationPlan,
    PortfolioConstraints,
    AccountSnapshot,
    PriceProvider,
    TargetWeightAllocator,
    _d,
    _abs,
)


@dataclass(frozen=True, slots=True)
class VolTargetAllocator:
    """
    volatility target allocation (risk budget prototype)

    inputs:
      - target_vol: target portfolio annualized vol (e.g. 0.15)
      - vols: Mapping[str, float|Decimal] per-symbol annualized vol (e.g. 0.60)
      - base_weights: optional, if provided use these first, then scale by target vol
        otherwise default to 1/vol equal-risk weights (approximate risk parity)

    output:
      - target_weights (relative), then constrained by max_gross_leverage
    """
    name: str = "vol_target_allocator"

    def allocate(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        account: AccountSnapshot,
        prices: PriceProvider,
        constraints: PortfolioConstraints,
        inputs: Mapping[str, Any] | None = None,
        tags: Tuple[str, ...] = (),
    ) -> AllocationPlan:
        if inputs is None:
            raise AllocatorError("VolTargetAllocator needs inputs")
        if "target_vol" not in inputs or "vols" not in inputs:
            raise AllocatorError("VolTargetAllocator needs inputs['target_vol'] and inputs['vols']")

        target_vol = _d(inputs["target_vol"])
        vols: Mapping[str, Any] = inputs["vols"]
        base_weights: Optional[Mapping[str, Any]] = inputs.get("base_weights")

        if target_vol <= 0:
            raise AllocatorError("target_vol must be positive")

        # 1) Generate relative weights (unscaled)
        rel_w: Dict[str, Decimal] = {}
        if base_weights is not None:
            for s in symbols:
                rel_w[s] = _d(base_weights.get(s, 0))
        else:
            # Approximate: w ~ 1/vol (higher vol -> lower weight)
            for s in symbols:
                v = _d(vols.get(s))
                if v is None or v <= 0:
                    raise AllocatorError(f"missing or invalid vol: {s} vol={v}")
                rel_w[s] = Decimal("1") / v

        # 2) Normalize to sum(abs)=1
        denom = sum(_abs(x) for x in rel_w.values())
        if denom == 0:
            raise AllocatorError("all weights are 0")
        for s in list(rel_w.keys()):
            rel_w[s] = rel_w[s] / denom

        # 3) Estimate portfolio vol (simplified: assume correlation=0)
        #    port_vol ~ sqrt(sum(w^2 * vol^2))
        port_var = Decimal("0")
        for s in symbols:
            v = _d(vols.get(s))
            port_var += (rel_w[s] * v) * (rel_w[s] * v)
        port_vol = port_var.sqrt() if port_var > 0 else Decimal("0")
        if port_vol <= 0:
            raise AllocatorError("portfolio vol estimate is 0, check vols/base_weights")

        # 4) Scale factor: scale = target_vol / port_vol
        scale = target_vol / port_vol

        # 5) Apply scale to weights (will be constrained by max_gross_leverage)
        scaled_w = {s: rel_w[s] * scale for s in symbols}

        return TargetWeightAllocator().allocate(
            ts=ts,
            symbols=symbols,
            account=account,
            prices=prices,
            constraints=constraints,
            inputs={"target_weights": scaled_w, "weight_residual_to_cash": True},
            tags=tags + (self.name,),
        )

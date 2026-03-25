# portfolio/allocator_constraints.py
"""Portfolio constraint functions and utility helpers.

Extracted from allocator.py to reduce file size.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, Mapping, Optional, Sequence, Tuple

from portfolio.allocator import (
    AllocationPlan,
    PriceProvider,
    PortfolioConstraints,
    TargetPosition,
    _d,
    _abs,
    _sign,
    _safe_div,
)


# ============================================================
# Constraint functions
# ============================================================

def _gross_notional(positions_qty: Mapping[str, Decimal], prices: PriceProvider) -> Decimal:
    total = Decimal("0")
    for s, q in positions_qty.items():
        px = _d(prices.price(s))
        total += _abs(_d(q)) * px
    return total


def _apply_gross_leverage_cap(
    targets: Mapping[str, TargetPosition],
    *,
    equity: Decimal,
    max_gross_leverage: Optional[Decimal],
) -> Tuple[Dict[str, TargetPosition], Optional[str]]:
    if max_gross_leverage is None:
        return dict(targets), None

    cap = _d(max_gross_leverage)
    if cap <= 0:
        return dict(targets), "max_gross_leverage<=0"

    gross = sum(t.target_notional for t in targets.values())
    lev = _safe_div(gross, equity)
    if lev <= cap or gross == 0:
        return dict(targets), None

    scale = cap / lev
    out: Dict[str, TargetPosition] = {}
    for s, t in targets.items():
        out[s] = TargetPosition(
            symbol=s,
            target_qty=t.target_qty * scale,
            target_notional=t.target_notional * scale,
            target_weight=t.target_weight * scale,
            meta=dict(t.meta) | {"scaled_by_gross_leverage": str(scale)},
        )
    return out, f"gross_leverage_cap: leverage {lev} -> {cap}, scaled_by={scale}"


def _apply_max_notional_per_symbol(
    targets: Mapping[str, TargetPosition],
    *,
    equity: Decimal,
    max_notional_per_symbol: Optional[Decimal],
    prices: PriceProvider,
) -> Tuple[Dict[str, TargetPosition], Tuple[str, ...]]:
    if max_notional_per_symbol is None:
        return dict(targets), ()

    cap = _d(max_notional_per_symbol)
    if cap <= 0:
        return dict(targets), ("max_notional_per_symbol<=0",)

    notes: list[str] = []
    out: Dict[str, TargetPosition] = {}
    for s, t in targets.items():
        if t.target_notional <= cap:
            out[s] = t
            continue

        px = _d(prices.price(s))
        if px <= 0:
            out[s] = t
            continue

        signed_notional = _sign(t.target_weight) * cap
        new_qty = signed_notional / px
        new_w = signed_notional / equity

        notes.append(f"max_notional_per_symbol: {s} {t.target_notional} -> {cap}")
        out[s] = TargetPosition(
            symbol=s,
            target_qty=new_qty,
            target_notional=cap,
            target_weight=new_w,
            meta=dict(t.meta) | {"capped_notional": str(cap)},
        )

    return out, tuple(notes)


def _apply_turnover_cap(
    targets: Mapping[str, TargetPosition],
    *,
    equity: Decimal,
    current_qty: Mapping[str, Decimal],
    prices: PriceProvider,
    turnover_cap: Optional[Decimal],
) -> Tuple[Dict[str, TargetPosition], Decimal, Tuple[str, ...]]:
    if turnover_cap is None:
        return dict(targets), Decimal("0"), ()

    cap = _d(turnover_cap)
    if cap <= 0:
        return dict(targets), Decimal("0"), ("turnover_cap<=0",)

    total_delta = Decimal("0")
    cur_notional: Dict[str, Decimal] = {}
    for s, t in targets.items():
        px = _d(prices.price(s))
        cq = _d(current_qty.get(s, 0))
        cn = _abs(cq) * px
        cur_notional[s] = cn
        total_delta += _abs(t.target_notional - cn)

    turnover = _safe_div(total_delta, equity)
    if turnover <= cap or total_delta == 0:
        return dict(targets), turnover, ()

    scale = cap / turnover

    out: Dict[str, TargetPosition] = {}
    for s, t in targets.items():
        px = _d(prices.price(s))
        cq = _d(current_qty.get(s, 0))

        cur_signed_notional = cq * px
        tgt_signed_notional = t.target_qty * px

        delta_signed = tgt_signed_notional - cur_signed_notional
        new_signed_notional = cur_signed_notional + delta_signed * scale

        new_qty = _safe_div(new_signed_notional, px)
        new_notional_abs = _abs(new_signed_notional)
        new_weight = _safe_div(new_signed_notional, equity)

        out[s] = TargetPosition(
            symbol=s,
            target_qty=new_qty,
            target_notional=new_notional_abs,
            target_weight=new_weight,
            meta=dict(t.meta) | {"turnover_scaled_by": str(scale)},
        )

    return out, cap, (f"turnover_cap: turnover {turnover} -> {cap}, scaled_by={scale}",)


# ============================================================
# Utility functions
# ============================================================

def plan_to_target_qty(plan: AllocationPlan) -> Dict[str, Decimal]:
    return {t.symbol: t.target_qty for t in plan.targets}


def plan_to_target_weight(plan: AllocationPlan) -> Dict[str, Decimal]:
    return {t.symbol: t.target_weight for t in plan.targets}


def plan_to_target_notional(plan: AllocationPlan) -> Dict[str, Decimal]:
    return {t.symbol: t.target_notional for t in plan.targets}


# --- Rust acceleration ---
try:
    from _quant_hotpath import RustPortfolioAllocator  # noqa: F401
    from _quant_hotpath import (
        cpp_compute_exposures,
        cpp_factor_model_covariance,
        cpp_estimate_specific_risk,
        cpp_black_litterman_posterior,
        rust_allocate_portfolio,
        rust_strategy_weights,
    )
    _RUST_ALLOCATOR_AVAILABLE = True

    compute_exposures = cpp_compute_exposures
    factor_model_covariance = cpp_factor_model_covariance
    estimate_specific_risk = cpp_estimate_specific_risk
    black_litterman_posterior = cpp_black_litterman_posterior
    allocate_portfolio = rust_allocate_portfolio
    strategy_weights = rust_strategy_weights
except ImportError:
    _RUST_ALLOCATOR_AVAILABLE = False


# ============================================================
# Rust-delegated allocation
# ============================================================

def _rust_allocate_targets(
    symbols: Sequence[str],
    target_weights: Dict[str, Decimal],
    equity: Decimal,
    prices: PriceProvider,
    current_qty: Mapping[str, Decimal],
    constraints: PortfolioConstraints,
) -> Dict[str, TargetPosition]:
    """Delegate weight->constraint->qty math to rust_allocate_portfolio."""
    w_f = {s: float(target_weights.get(s, 0)) for s in symbols}
    px_f = {s: float(prices.price(s)) for s in symbols}
    eq_f = float(equity)
    cq_f = {s: float(current_qty.get(s, 0)) for s in symbols}

    result = rust_allocate_portfolio(
        target_weights=w_f,
        prices=px_f,
        equity=eq_f,
        current_qty=cq_f,
        max_weight=(
            float(constraints.max_weight) if constraints.max_weight is not None else None
        ),
        max_notional_per_symbol=(
            float(constraints.max_notional_per_symbol)
            if constraints.max_notional_per_symbol is not None else None
        ),
        max_gross_leverage=(
            float(constraints.max_gross_leverage)
            if constraints.max_gross_leverage is not None else None
        ),
        turnover_cap=(
            float(constraints.turnover_cap)
            if constraints.turnover_cap is not None else None
        ),
        allow_short=constraints.allow_short,
    )

    targets: Dict[str, TargetPosition] = {}
    for s in symbols:
        if s not in result:
            targets[s] = TargetPosition(
                symbol=s,
                target_qty=Decimal("0"),
                target_notional=Decimal("0"),
                target_weight=Decimal("0"),
            )
            continue
        r = result[s]
        targets[s] = TargetPosition(
            symbol=s,
            target_qty=Decimal(str(r["qty"])),
            target_notional=Decimal(str(r["notional"])),
            target_weight=Decimal(str(r["weight"])),
            meta={"price": str(px_f[s]), "rust_delegated": "true"},
        )
    return targets

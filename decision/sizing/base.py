"""Position sizing protocols and implementations.

Sizers convert allocation weights into concrete position quantities,
accounting for account equity, volatility, and risk constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Protocol

from state.snapshot import StateSnapshot

from _quant_hotpath import rust_volatility_adjusted_qty as _rust_va_qty


class PositionSizer(Protocol):
    """Protocol for position sizing."""
    def target_qty(self, snapshot: StateSnapshot, symbol: str, weight: Decimal) -> Decimal:
        """Compute target position quantity for a given weight allocation."""
        ...


@dataclass(frozen=True, slots=True)
class VolatilityAdjustedSizer:
    """Size positions inversely proportional to volatility.

    Uses ATR (Average True Range) or rolling volatility from features.

    target_qty = (equity * risk_fraction * weight) / (volatility * price)

    Parameters
    ----------
    risk_fraction : Decimal
        Fraction of equity to risk per position.
    volatility_key : str
        Key in snapshot features for the volatility measure.
    default_volatility : Decimal
        Fallback volatility if feature is missing.
    """
    risk_fraction: Decimal = Decimal("0.02")
    volatility_key: str = "atr"
    default_volatility: Decimal = Decimal("0.02")

    def target_qty(self, snapshot: StateSnapshot, symbol: str, weight: Decimal) -> Decimal:
        equity = _get_equity(snapshot)
        price = _get_price(snapshot)
        vol = _get_feature(snapshot, self.volatility_key, self.default_volatility)

        if price <= 0 or vol <= 0:
            return Decimal("0")

        qty_f = _rust_va_qty(
            float(equity),
            float(price),
            float(vol),
            float(self.risk_fraction),
            float(weight),
        )
        return Decimal(str(qty_f))


# ── Helpers ──────────────────────────────────────────────

def _get_equity(snapshot: StateSnapshot) -> Decimal:
    acct = getattr(snapshot, "account", None)
    if acct is None:
        return Decimal("0")
    bf = getattr(acct, "balance_f", None)
    if bf is not None:
        return Decimal(str(bf))
    return getattr(acct, "balance", Decimal("0"))


def _get_price(snapshot: StateSnapshot) -> Decimal:
    market = getattr(snapshot, "market", None)
    if market is None:
        return Decimal("0")
    cf = getattr(market, "close_f", None)
    if cf is not None:
        return Decimal(str(cf))
    p = getattr(market, "close", None) or getattr(market, "last_price", None)
    if p is None:
        return Decimal("0")
    return Decimal(str(p))


def _get_feature(snapshot: StateSnapshot, key: str, default: Decimal) -> Decimal:
    feats = getattr(snapshot, "features", None)
    if isinstance(feats, Mapping):
        raw = feats.get(key)
        if raw is not None:
            try:
                return Decimal(str(raw))
            except (ValueError, TypeError):
                pass
    return default

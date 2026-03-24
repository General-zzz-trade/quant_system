from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

# State fields are Rust PyO3 types (RustMarketState, RustPositionState, etc.)
# imported via ``from state import MarketState, ...`` which re-exports from
# _quant_hotpath.  Snapshot is duck-typed so no direct import needed here.

def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    # If already a lazy-convert mapping or MappingProxyType, return as-is
    if hasattr(mapping, "_converter"):
        return mapping
    if isinstance(mapping, MappingProxyType):
        return mapping
    # For small dicts (1-3 symbols), skip sort overhead
    if len(mapping) <= 3:
        return MappingProxyType(dict(mapping))
    ordered = {k: mapping[k] for k in sorted(mapping.keys())}
    return MappingProxyType(ordered)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable snapshot at an event boundary.

    State fields are Rust PyO3 types (RustMarketState etc.) — duck-typed.
    """

    symbol: str
    ts: Optional[datetime]
    event_id: Optional[str]
    event_type: str
    bar_index: int

    markets: Mapping[str, Any]
    positions: Mapping[str, Any]
    account: Any

    portfolio: Any = None
    risk: Any = None
    features: Optional[Mapping[str, Any]] = None

    @property
    def market(self) -> Any:
        """Backward compat: return MarketState for self.symbol."""
        if self.symbol in self.markets:
            return self.markets[self.symbol]
        return next(iter(self.markets.values()))

    @property
    def symbols(self) -> Tuple[str, ...]:
        return tuple(sorted(self.positions.keys()))

    @classmethod
    def of(
        cls,
        *,
        symbol: str,
        ts: Optional[datetime],
        event_id: Optional[str],
        event_type: str,
        bar_index: int,
        markets: Mapping[str, Any],
        positions: Mapping[str, Any],
        account: Any,
        portfolio: Optional[Any] = None,
        risk: Optional[Any] = None,
        features: Optional[Mapping[str, Any]] = None,
    ) -> "StateSnapshot":
        return cls(
            symbol=symbol,
            ts=ts,
            event_id=event_id,
            event_type=event_type,
            bar_index=bar_index,
            markets=_freeze_mapping(markets),
            positions=_freeze_mapping(positions),
            account=account,
            portfolio=portfolio,
            risk=risk,
            features=features,
        )

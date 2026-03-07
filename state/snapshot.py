from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from state.portfolio import PortfolioState
from state.risk import RiskState


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    ordered = {k: mapping[k] for k in sorted(mapping.keys())}
    return MappingProxyType(ordered)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable snapshot at an event boundary.

    State fields accept both Python dataclass types (MarketState etc.)
    and Rust PyO3 types (RustMarketState etc.) — duck-typed.
    """

    symbol: str
    ts: Optional[datetime]
    event_id: Optional[str]
    event_type: str
    bar_index: int

    markets: Mapping[str, Any]
    positions: Mapping[str, Any]
    account: Any

    portfolio: Optional[PortfolioState] = None
    risk: Optional[RiskState] = None
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

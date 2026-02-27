from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.portfolio import PortfolioState
from state.risk import RiskState


def _freeze_positions(positions: Mapping[str, PositionState]) -> Mapping[str, PositionState]:
    # stable order for diffs/tests
    ordered = {k: positions[k] for k in sorted(positions.keys())}
    return MappingProxyType(ordered)


def _freeze_markets(markets: Mapping[str, MarketState]) -> Mapping[str, MarketState]:
    ordered = {k: markets[k] for k in sorted(markets.keys())}
    return MappingProxyType(ordered)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable snapshot at an event boundary."""

    symbol: str
    ts: Optional[datetime]
    event_id: Optional[str]
    event_type: str
    bar_index: int

    markets: Mapping[str, MarketState]
    positions: Mapping[str, PositionState]
    account: AccountState

    portfolio: Optional[PortfolioState] = None
    risk: Optional[RiskState] = None
    features: Optional[Mapping[str, Any]] = None

    @property
    def market(self) -> MarketState:
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
        markets: Mapping[str, MarketState],
        positions: Mapping[str, PositionState],
        account: AccountState,
        portfolio: Optional[PortfolioState] = None,
        risk: Optional[RiskState] = None,
        features: Optional[Mapping[str, Any]] = None,
    ) -> "StateSnapshot":
        return cls(
            symbol=symbol,
            ts=ts,
            event_id=event_id,
            event_type=event_type,
            bar_index=bar_index,
            markets=_freeze_markets(markets),
            positions=_freeze_positions(positions),
            account=account,
            portfolio=portfolio,
            risk=risk,
            features=features,
        )

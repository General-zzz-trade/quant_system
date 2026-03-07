# DEPRECATED: Superseded by Rust pipeline. Retained for parity tests.
from __future__ import annotations

from typing import Any, Callable, Mapping

from state.portfolio import PortfolioState
from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.reducers.base import ReducerResult
from state._util import get_event_ts


class PortfolioReducer:
    """Recompute PortfolioState deterministically.

    Portfolio is derived from (account, positions, market). We recompute on every
    event boundary to keep the models simple and deterministic.
    """

    def __init__(
        self,
        *,
        get_account: Callable[[], AccountState],
        get_positions: Callable[[], Mapping[str, PositionState]],
        get_market: Callable[[], MarketState],
    ) -> None:
        self._get_account = get_account
        self._get_positions = get_positions
        self._get_market = get_market

    def reduce(self, state: PortfolioState, event: Any) -> ReducerResult[PortfolioState]:
        account = self._get_account()
        positions = self._get_positions()
        market = self._get_market()
        ts = get_event_ts(event)

        new_state = PortfolioState.compute(account=account, positions=positions, market=market, ts=ts)

        changed = new_state != state
        return ReducerResult(state=new_state, changed=changed, note="portfolio_recompute")

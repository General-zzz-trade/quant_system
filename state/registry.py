from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from decimal import Decimal

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.portfolio import PortfolioState
from state.risk import RiskLimits, RiskState
from state.snapshot import StateSnapshot
from state._util import get_event_id, get_event_ts, get_event_type, get_symbol
from state.reducers.market import MarketReducer
from state.reducers.position import PositionReducer
from state.reducers.account import AccountReducer
from state.reducers.portfolio import PortfolioReducer
from state.reducers.risk import RiskReducer
from state.reducers.base import apply_one


@dataclass
class StateProjector:
    """State projector (Route A -> Route B).

    Route A (v1) produced market/position/account snapshots deterministically.
    Route B adds:
      - PortfolioState recomputation (MTM aggregation)
      - RiskState evaluation (limits + manual RISK/CONTROL events)

    This projector is intentionally single-symbol for now. Extend by replacing
    positions dict and market state with per-symbol maps.
    """

    symbol: str
    currency: str

    market: MarketState
    account: AccountState
    positions: Dict[str, PositionState]
    bar_index: int = 0

    portfolio: Optional[PortfolioState] = None
    risk: Optional[RiskState] = None

    # reducers
    _market_reducer: MarketReducer = MarketReducer()
    _position_reducer: PositionReducer = PositionReducer()
    _account_reducer: AccountReducer = AccountReducer()

    _portfolio_reducer: Optional[PortfolioReducer] = None
    _risk_reducer: Optional[RiskReducer] = None

    @classmethod
    def initial(
        cls,
        *,
        symbol: str,
        currency: str = "USDT",
        initial_equity: str | Decimal = "0",
        risk_limits: Optional[RiskLimits] = None,
    ) -> "StateProjector":
        bal = Decimal(str(initial_equity)) if not isinstance(initial_equity, Decimal) else initial_equity

        proj = cls(
            symbol=symbol,
            currency=currency,
            market=MarketState.empty(symbol),
            account=AccountState.initial(currency=currency, balance=bal),
            positions={symbol: PositionState.empty(symbol)},
            bar_index=0,
            portfolio=None,
            risk=RiskState(),  # default empty risk
        )

        # Wire derived reducers using closures over projector state
        proj._portfolio_reducer = PortfolioReducer(
            get_account=lambda: proj.account,
            get_positions=lambda: proj.positions,
            get_market=lambda: proj.market,
        )

        proj.portfolio = PortfolioState.compute(
            account=proj.account,
            positions=proj.positions,
            market=proj.market,
            ts=None,
        )

        lim = risk_limits or RiskLimits()
        proj._risk_reducer = RiskReducer(
            limits=lim,
            get_portfolio=lambda: proj.portfolio if proj.portfolio is not None else PortfolioState.compute(account=proj.account, positions=proj.positions, market=proj.market, ts=None),
            get_positions=lambda: proj.positions,
        )

        return proj

    def apply(self, event: Any) -> StateSnapshot:
        # v1 is single symbol: ignore events for other symbols
        sym = get_symbol(event, self.symbol)
        if sym != self.symbol:
            return self.snapshot(event)

        # Apply market projection first (MTM derived state depends on it)
        m_res = apply_one(self.market, self._market_reducer, event)
        self.market = m_res.state

        # bar_index increments on bar events (deterministic)
        et = get_event_type(event)
        if et in ("market", "market_bar", "bar", "marketbar"):
            self.bar_index += 1

        # Apply fill-driven projections
        pos = self.positions.get(self.symbol) or PositionState.empty(self.symbol)
        p_res = apply_one(pos, self._position_reducer, event)
        self.positions[self.symbol] = p_res.state

        a_res = apply_one(self.account, self._account_reducer, event)
        self.account = a_res.state

        # Route B: recompute derived portfolio + risk
        if self._portfolio_reducer is not None:
            if self.portfolio is None:
                # bootstrap
                self.portfolio = PortfolioState.compute(account=self.account, positions=self.positions, market=self.market, ts=get_event_ts(event))
            else:
                pr = apply_one(self.portfolio, self._portfolio_reducer, event)
                self.portfolio = pr.state

        if self._risk_reducer is not None:
            if self.risk is None:
                self.risk = RiskState()
            rr = apply_one(self.risk, self._risk_reducer, event)
            self.risk = rr.state

        return self.snapshot(event)

    def snapshot(self, event: Any) -> StateSnapshot:
        return StateSnapshot.of(
            symbol=self.symbol,
            ts=get_event_ts(event),
            event_id=get_event_id(event),
            event_type=get_event_type(event),
            bar_index=self.bar_index,
            market=self.market,
            positions=self.positions,
            account=self.account,
            portfolio=self.portfolio,
            risk=self.risk,
        )

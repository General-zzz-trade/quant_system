from __future__ import annotations

from typing import Any, Optional

import logging
import math

from state.errors import ReducerError
from state.market import MarketState
from state.reducers.base import ReducerResult
from state._util import get_event_type, get_event_ts, get_symbol, to_decimal

_logger = logging.getLogger(__name__)


class MarketReducer:
    """Project market events into MarketState.

    Supports both:
    - EventType.MARKET v1 (OHLCV fields: open/high/low/close/volume)
    - Legacy market_bar / market_tick forms
    """

    def reduce(self, state: MarketState, event: Any) -> ReducerResult[MarketState]:
        et = get_event_type(event)
        sym = get_symbol(event, state.symbol)

        if sym != state.symbol:
            # v1 is single-symbol state; ignore other symbols
            return ReducerResult(state=state, changed=False)

        ts = get_event_ts(event)

        # Bar-like events
        if et in ("market", "market_bar", "bar", "marketbar"):
            o = getattr(event, "open", None)
            h = getattr(event, "high", None)
            l = getattr(event, "low", None)
            c = getattr(event, "close", None)
            v = getattr(event, "volume", None)

            # Support nested bar object
            bar = getattr(event, "bar", None)
            if c is None and bar is not None:
                o = getattr(bar, "open", o)
                h = getattr(bar, "high", h)
                l = getattr(bar, "low", l)
                c = getattr(bar, "close", c)
                v = getattr(bar, "volume", v)

            if c is not None:
                try:
                    c_f = float(c)
                    if c_f <= 0 or not math.isfinite(c_f):
                        _logger.warning("Invalid close price for %s: %s, skipping", sym, c)
                        return ReducerResult(state=state, changed=False)
                except (TypeError, ValueError):
                    pass

            # Validate high >= low and volume >= 0
            if h is not None and l is not None:
                try:
                    h_f, l_f = float(h), float(l)
                    if h_f < l_f:
                        _logger.warning("Invalid candle for %s: high=%.4f < low=%.4f, skipping", sym, h_f, l_f)
                        return ReducerResult(state=state, changed=False)
                except (TypeError, ValueError):
                    pass
            if v is not None:
                try:
                    if float(v) < 0:
                        _logger.warning("Invalid candle for %s: volume < 0, skipping", sym)
                        return ReducerResult(state=state, changed=False)
                except (TypeError, ValueError):
                    pass

            if c is None:
                # If only tick price exists, treat as tick
                price = getattr(event, "price", None)
                if price is not None:
                    p = to_decimal(price)
                    return ReducerResult(state=state.with_tick(price=p, ts=ts), changed=True, note="market_tick")
                raise ReducerError("market event missing close/price")

            o_d = to_decimal(o, allow_none=True) if o is not None else to_decimal(c)
            h_d = to_decimal(h, allow_none=True) if h is not None else to_decimal(c)
            l_d = to_decimal(l, allow_none=True) if l is not None else to_decimal(c)
            c_d = to_decimal(c)
            v_d = to_decimal(v, allow_none=True) if v is not None else None

            new_state = state.with_bar(o=o_d, h=h_d, l=l_d, c=c_d, v=v_d, ts=ts)
            return ReducerResult(state=new_state, changed=True, note="market_bar")

        # Tick-like events
        if et in ("market_tick", "tick"):
            price = getattr(event, "price", None)
            if price is None:
                raise ReducerError("tick event missing price")
            p = to_decimal(price)
            return ReducerResult(state=state.with_tick(price=p, ts=ts), changed=True, note="market_tick")

        return ReducerResult(state=state, changed=False)

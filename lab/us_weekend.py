"""The weekend gap of Binance's US-stock perpetuals.

Owner 2026-11-18: "先开始开发策略", "两个都做" (the weekend gap first).

The stock market is closed from Friday's close to Monday's open (longer over a holiday); the perpetual trades all the
time, on a few per cent of its weekday volume. If thin weekend trading pushes it off, the stock's real open should
pull it back. One event per trading-calendar break of three calendar days or more:

- ``reference``: the hourly close at the last trading day's 16:00 New York (daylight saving included);
- ``decision``: the last hourly close at least 30 minutes before the next trading day's 09:30 New York open. The move
  is that close over the reference; a move of at least the threshold is faded (a market order at the next bar's open);
- ``exit``: the first hourly close at least an hour after the open ("open+1h"), or the hourly close at that day's
  16:00 ("close"); the position is closed at the next bar's open;
- the trade goes through the shared broker (``replay``): its fees and slippage, the funding settled while the
  position is held (a long pays a positive rate, a short receives it).

An event uses only bars closed by its decision to decide; the prices after it only settle the trade.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal as D
from zoneinfo import ZoneInfo

import pandas as pd

from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec
from lab.um_lab import _fill
from trading.domain.bars import bar_from_kline

H = 3_600_000
DAY = 24 * H
NEW_YORK = ZoneInfo("America/New_York")
EXIT_RULES = ("open+1h", "close")
# The fade's target: under the broker's 1x exposure cap, which refuses a fill whose notional exceeds it (a full 1
# was refused whenever the next open moved the notional over it); the lab's ideas hold 0.95 for the same reason.
EXPOSURE = 0.9


def _ms(day: date, at: time) -> int:
    return int(datetime.combine(day, at, NEW_YORK).astimezone(timezone.utc).timestamp() * 1000)


def session(day: date) -> tuple[int, int]:
    """(open, close) of a US trading day in UTC ms: 09:30 and 16:00 New York."""
    return _ms(day, time(9, 30)), _ms(day, time(16, 0))


def breaks(days: list[date]) -> list[tuple[date, date]]:
    """Consecutive trading days three or more calendar days apart (weekends, long weekends)."""
    days = sorted(days)
    return [(a, b) for a, b in zip(days, days[1:]) if (b - a).days >= 3]


def times(d0: date, d1: date, *, exit_rule: str) -> dict[str, int]:
    """The event's hourly closes (UTC ms): reference, decision and exit."""
    if exit_rule not in EXIT_RULES:
        raise ValueError(f"exit_rule must be one of {EXIT_RULES}")
    _, close0 = session(d0)
    open1, close1 = session(d1)
    exit_at = -(-(open1 + H) // H) * H if exit_rule == "open+1h" else close1 // H * H
    return {"reference": close0 // H * H, "decision": (open1 - 30 * 60_000) // H * H, "exit": exit_at}


def funding_between(funding: pd.Series, start: int, end: int) -> float:
    """The sum of the funding rates settled after ``start`` up to ``end`` (UTC ms)."""
    if funding.empty:
        return 0.0
    inside = funding[(funding.index > start) & (funding.index <= end)]
    return float(inside.sum())


def event(bars: pd.DataFrame, funding: pd.Series, d0: date, d1: date, *, exit_rule: str) -> dict | None:
    """``bars``: hourly bars indexed by open time (ms) with open and close. None when a bar the event needs is
    missing. The close at time T is the close of the bar opened at T - 1h; a fill at T is the open of the bar opened
    at T."""
    t = times(d0, d1, exit_rule=exit_rule)
    needed = (t["reference"] - H, t["decision"] - H, t["decision"], t["exit"])
    if any(k not in bars.index for k in needed):
        return None
    reference = float(bars.at[t["reference"] - H, "close"])
    decided = float(bars.at[t["decision"] - H, "close"])
    return {**t, "d0": d0.isoformat(), "d1": d1.isoformat(), "reference_price": reference,
            "move": decided / reference - 1.0, "entry": float(bars.at[t["decision"], "open"]),
            "exit_price": float(bars.at[t["exit"], "open"]),
            "funding": funding_between(funding, t["decision"], t["exit"])}


def fade_side(move: float) -> float:
    """Short after a rise, long after a fall."""
    return -1.0 if move > 0 else 1.0


def trades_at(e: dict, *, threshold: float) -> bool:
    """An event is traded when its move is at least the threshold (and not zero)."""
    return e["move"] != 0 and abs(e["move"]) >= threshold


def replay(bars: pd.DataFrame, funding: pd.Series, e: dict, *, rules: SymbolRules, cost_multiplier: int = 1,
           funding_multiplier: float = 1.0, capital: str = "50") -> float | None:
    """The fade of one event through the shared broker (``lab.simulate``; the S12 rule: no research fill model of its
    own): the target is decided at the decision's close and filled at the next bar's open, taken back to 0 at the
    exit's close and filled at the next bar's open. Fees and slippage are the research broker's (5 and 3 basis points
    a side, times ``cost_multiplier``); funding is the rates settled while the position is held (times
    ``funding_multiplier``). Returns the result per unit of the entry's notional; None when nothing filled."""
    lo, hi = e["decision"] - H, e["exit"] + H
    window = bars[(bars.index >= lo) & (bars.index <= hi)]
    series = [bar_from_kline({"start": int(t), "open": str(r.open), "high": str(r.high), "low": str(r.low),
                              "close": str(r.close), "volume": str(r.volume)})
              for t, r in zip(window.index, window.itertuples())]
    targets = pd.DataFrame({"available_at": [e["decision"], e["exit"]],
                            "target": [fade_side(e["move"]) * EXPOSURE, 0.0], "ready": [True, True]})
    settled = {int(t) // H * H: (D(repr(float(rate) * funding_multiplier)),)
               for t, rate in funding.items() if lo <= t <= hi}
    result = simulate("event", series, targets, settled, StrategySpec(execution="market"), start_ms=lo, end_ms=hi + H,
                      rules=rules, cost_multiplier=cost_multiplier, starting_balance=capital)
    if not result.fills:
        return None
    first = _fill(result.fills[0])
    notional = float(D(first["qty"]) * D(first["price"]))
    return (float(result.curve["equity"].iloc[-1]) - float(capital)) / notional


def tracks(perp_price: float, stock_close: float, tolerance: float = 0.10) -> bool:
    """The perpetual prices its stock: within ``tolerance`` of the stock's close (a reused name does not)."""
    return stock_close > 0 and abs(perp_price / stock_close - 1.0) <= tolerance


def median_daily_quote(bars: pd.DataFrame, at: int, days: int = 30) -> float:
    """The median daily quote volume of the ``days`` days before ``at`` (bars opened before it only)."""
    window = bars[(bars.index >= at - days * DAY) & (bars.index < at)]
    if window.empty:
        return 0.0
    return float(window["quote_volume"].groupby(window.index // DAY).sum().median())


__all__ = ["D", "EXIT_RULES", "EXPOSURE", "H", "SymbolRules", "breaks", "event", "fade_side", "funding_between",
           "median_daily_quote", "replay", "session", "times", "trades_at", "tracks"]

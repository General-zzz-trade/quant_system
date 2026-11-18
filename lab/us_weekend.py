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
- the funding settled while the position is held counts (a long pays a positive rate, a short receives it); fees and
  slippage are a fixed cost per side.

An event uses only bars closed by its decision to decide; the prices after it only settle the trade.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

H = 3_600_000
DAY = 24 * H
NEW_YORK = ZoneInfo("America/New_York")
EXIT_RULES = ("open+1h", "close")


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


def net_return(e: dict, *, threshold: float, cost_bp: float, funding_multiplier: float = 1.0) -> float | None:
    """The fade's return per unit of notional after costs and funding; None below the threshold (no trade)."""
    if abs(e["move"]) < threshold or e["move"] == 0:
        return None
    side = -1.0 if e["move"] > 0 else 1.0
    gross = side * (e["exit_price"] / e["entry"] - 1.0)
    return gross - 2 * cost_bp / 1e4 - side * e["funding"] * funding_multiplier


def tracks(perp_price: float, stock_close: float, tolerance: float = 0.10) -> bool:
    """The perpetual prices its stock: within ``tolerance`` of the stock's close (a reused name does not)."""
    return stock_close > 0 and abs(perp_price / stock_close - 1.0) <= tolerance


def median_daily_quote(bars: pd.DataFrame, at: int, days: int = 30) -> float:
    """The median daily quote volume of the ``days`` days before ``at`` (bars opened before it only)."""
    window = bars[(bars.index >= at - days * DAY) & (bars.index < at)]
    if window.empty:
        return 0.0
    return float(window["quote_volume"].groupby(window.index // DAY).sum().median())


__all__ = ["EXIT_RULES", "H", "breaks", "event", "funding_between", "median_daily_quote", "net_return", "session",
           "times", "tracks"]

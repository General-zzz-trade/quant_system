"""Earnings drift on the underlyings of Binance's US-stock perpetuals.

Owner 2026-11-18: "同意，按草案登记" (research/protocols/2026-11-18-us-earnings-drift.md, the first study under the
standards of the 2026-11-18 retrospective).

- An event is a calendar row (day d, symbol) on one of the stock's trading days. Its window runs from the close of the
  trading day before d to the close of the trading day after: the news is inside it whether the report came before
  the open or after the close (the calendar does not say which).
- The decision is at the close of the day after d (``decide``); a position of ``EXPOSURE`` in the signal's direction
  fills at the next open and is held ``hold`` trading days (weights on the decisions ``decide`` .. ``hold`` - 1 days
  later; the next decision is 0, the exit fills at the following open).
- Signals: ``surprise`` (actual minus forecast EPS over max(|forecast|, 0.01), at least 5 %) and ``reaction`` (the
  window's return minus QQQ's over the same days, at least 3 %).
- The trades go through the shared broker (``lab.us_equity.replay_daily``) per stock: longs and shorts in separate
  sleeves so that each can be charged the funding as a cost. ``event_return`` reads an event's result off its sleeve's
  curve; ``portfolio`` averages the sleeves that hold a position each day (cash when none does).
"""
from __future__ import annotations

import pandas as pd

DAY = 86_400_000
EXPOSURE = 0.9                   # under the broker's 1x exposure cap, as the lab's ideas hold 0.95
SURPRISE_MIN, REACTION_MIN = 0.05, 0.03


def parse_eps(text: object) -> float | None:
    """'$4.74' -> 4.74, '($0.08)' -> -0.08, '' -> None."""
    s = str(text or "").strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def surprise(eps: float | None, forecast: float | None) -> float | None:
    if eps is None or forecast is None:
        return None
    return (eps - forecast) / max(abs(forecast), 0.01)


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 10**6)


def events(ticker: str, calendar: pd.DataFrame, stock: pd.DataFrame, qqq: pd.DataFrame) -> tuple[list[dict], int]:
    """(events, rows skipped because their day is not one of the stock's trading days)."""
    index = stock.index
    position = {int(t): i for i, t in enumerate(index)}
    out, skipped = [], 0
    for row in calendar[calendar["symbol"] == ticker].itertuples():
        i = position.get(_ms(str(row.date)))
        if i is None:
            skipped += 1
            continue
        if i < 1 or i + 1 >= len(index):
            continue
        before, after = int(index[i - 1]), int(index[i + 1])
        reaction = float(stock.at[after, "close"]) / float(stock.at[before, "close"]) - 1.0
        adjusted = None
        if before in qqq.index and after in qqq.index:
            adjusted = reaction - (float(qqq.at[after, "close"]) / float(qqq.at[before, "close"]) - 1.0)
        out.append({"ticker": ticker, "day": str(row.date), "decide": after, "reaction": reaction,
                    "reaction_adjusted": adjusted,
                    "surprise": surprise(parse_eps(row.eps), parse_eps(row.epsForecast))})
    return out, skipped


def side(e: dict, kind: str) -> int:
    """+1 long, -1 short, 0 no trade."""
    value, minimum = (e["surprise"], SURPRISE_MIN) if kind == "surprise" else (e["reaction_adjusted"], REACTION_MIN)
    if value is None or abs(value) < minimum - 1e-12:
        return 0
    return 1 if value > 0 else -1


def weights(index: pd.Index, held: list[dict], *, hold: int) -> pd.Series:
    """The target per decision day: each event's side times ``EXPOSURE`` on its ``hold`` decision days."""
    w = pd.Series(0.0, index=index)
    for e in held:
        j = index.get_loc(e["decide"])
        w.iloc[j:j + hold] = e["side"] * EXPOSURE
    return w


def split(w: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(the long weights, the short weights)."""
    return w.clip(lower=0.0), w.clip(upper=0.0)


def event_return(curve: pd.Series, *, decide: int, hold: int) -> float:
    """The sleeve's equity after the exit over its equity before the entry. ``curve``: the equity at each bar's close
    (indexed by the bar's open time plus a day); the entry fills at the bar after ``decide``, the exit at the bar
    ``hold`` + 1 after it."""
    opens = list(curve.index - DAY)
    j = opens.index(decide)
    before = curve.iloc[j]
    after = curve.iloc[min(j + hold + 1, len(curve) - 1)]
    return float(after / before - 1.0)


def portfolio(sleeves: list[pd.Series]) -> pd.Series:
    """Each day the mean return of the sleeves whose equity moved (they hold a position); 0 when none did."""
    panel = pd.concat(sleeves, axis=1).fillna(0.0)
    active = panel.ne(0.0)
    total = panel.where(active).sum(axis=1)
    count = active.sum(axis=1)
    return (total / count.where(count > 0)).fillna(0.0)


def week(ms: int) -> tuple[int, int]:
    """The ISO year and week of a day: earnings events are clustered by it."""
    iso = pd.Timestamp(ms, unit="ms").isocalendar()
    return int(iso[0]), int(iso[1])


__all__ = ["DAY", "EXPOSURE", "REACTION_MIN", "SURPRISE_MIN", "event_return", "events", "parse_eps", "portfolio",
           "side", "split", "surprise", "week", "weights"]

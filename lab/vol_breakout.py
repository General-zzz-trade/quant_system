"""The volatility-expansion breakout study (research/results/vol_breakout_2026-11-17/README.md, registered before any
code; owner 2026-11-17 "跑回测验证"): a fixed channel breakout says the direction, the panel model's trend-coming
prediction says whether to take it (F1) and its 7-day volatility prediction sizes it (S1). Research functions only;
the replays go through lab.um_lab.replay (the shared broker).

- Decisions are made at a day's end (``t``, ms) from complete daily bars (``lab.feature_pool.daily``); a decision's
  weight is held by the hourly bars whose close is at or after it, until the next decision (``weights``).
- A trade: long when the close is above the highest high of the 20 days before, short when below the lowest low;
  out when the close crosses the opposite 10-day channel, after 14 days, or at the data's end; no entry on the day of
  an exit. The 8 % exchange stop is the replay's (``pause_then_reset``: no re-entry until the trade's target is 0).
- The filter (F1) keeps the unfiltered trades (F0) whose entry day ranks the contract in the top fifth of the day's
  universe by the trend prediction; the slots go to the most traded first (``allocate``), so F0 and F1 differ only by
  the filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from lab import feature_pool as fp
from lab import feature_study as st
from lab import panel_study as ps
from trading.domain.strategy.ideas import ON

H, DAY = ps.H, ps.DAY
ENTRY_DAYS, EXIT_DAYS, MAX_HOLD_DAYS = 20, 10, 14
TOP_SHARE, SLOTS = 0.2, 10


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: int          # +1 long, -1 short
    entry: int         # the decision time of the entry (a day's end, ms)
    exit: int          # the decision time of the exit
    reason: str        # "channel", "time" or "data_end"


def breakout_trades(symbol: str, daily: pd.DataFrame, *, allowed: pd.Series | None = None) -> list[Trade]:
    """The channel breakout's trades over ``daily`` (close, high, low; index: the day's end). A missing day is
    unknown: no channel that spans it gives a signal. ``allowed``: the days an entry may happen (the universe)."""
    if daily.empty:
        return []
    start, end = int(daily.index[0]), int(daily.index[-1])
    d = daily.reindex(pd.Index(np.arange(start, end + 1, DAY, dtype=np.int64)))
    up, down = d["high"].rolling(ENTRY_DAYS).max().shift(1), d["low"].rolling(ENTRY_DAYS).min().shift(1)
    out_long, out_short = d["low"].rolling(EXIT_DAYS).min().shift(1), d["high"].rolling(EXIT_DAYS).max().shift(1)
    ok = np.ones(len(d), dtype=bool) if allowed is None else \
        allowed.reindex(d.index).fillna(False).to_numpy(dtype=bool)
    close, times = d["close"].to_numpy(float), d.index.to_numpy(dtype=np.int64)
    trades, side, entry = [], 0, 0
    for i, t in enumerate(times):
        c = close[i]
        if side:
            channel = c < out_long.iloc[i] if side == 1 else c > out_short.iloc[i]
            if channel or t - entry >= MAX_HOLD_DAYS * DAY:
                trades.append(Trade(symbol, side, entry, int(t), "channel" if channel else "time"))
                side = 0
            continue
        if ok[i] and c > up.iloc[i]:
            side, entry = 1, int(t)
        elif ok[i] and c < down.iloc[i]:
            side, entry = -1, int(t)
    if side:
        trades.append(Trade(symbol, side, entry, end, "data_end"))
    return trades


def positions(trades: Iterable[Trade], times: np.ndarray) -> pd.Series:
    """The position (+1, -1, 0) each trade holds at each decision time: from its entry up to its exit; a trade cut by
    the data's end still holds at that last day (nothing decided it out)."""
    t = np.asarray(times, dtype=np.int64)
    pos = np.zeros(len(t))
    for trade in trades:
        held = (t < trade.exit) | ((t == trade.exit) & (trade.reason == "data_end"))
        pos[(t >= trade.entry) & held] = trade.side
    return pd.Series(pos, index=pd.Index(t))


def allocate(trades: Sequence[Trade], priority: Callable[[Trade], float], *, slots: int = SLOTS) -> list[Trade]:
    """The trades that get one of ``slots`` slots: by entry time, the day's new trades by ``priority`` (highest
    first); a slot is held until the trade's exit decision (also after a stop) and is free again at it."""
    taken: list[Trade] = []
    for entry in sorted({t.entry for t in trades}):
        busy = sum(1 for t in taken if t.exit > entry)
        for trade in sorted((t for t in trades if t.entry == entry), key=lambda t: (-priority(t), t.symbol)):
            if busy < slots:
                taken.append(trade)
                busy += 1
    return taken


def top_share(predictions: pd.DataFrame, *, share: float = TOP_SHARE) -> dict[int, set[str]]:
    """Per decision time, the contracts whose prediction ranks in the top ``share`` of that day's rows (time,
    symbol, pred; missing predictions are not ranked): the top round(share x n), at least one."""
    out: dict[int, set[str]] = {}
    rows = predictions[np.isfinite(predictions["pred"].to_numpy(float))]
    for t, g in rows.groupby("time"):
        k = max(1, int(round(share * len(g))))
        out[int(t)] = set(g.sort_values(["pred", "symbol"], ascending=[False, True])["symbol"].iloc[:k])
    return out


def vol_sizes(predictions: pd.DataFrame) -> dict[tuple[str, int], float]:
    """S1's weight per (symbol, time): ``ON`` x min(1, the day's median predicted volatility over the contract's).
    The predictions are logs of the daily volatility (``fwd_vol_7``)."""
    out: dict[tuple[str, int], float] = {}
    rows = predictions[np.isfinite(predictions["pred"].to_numpy(float))]
    for t, g in rows.groupby("time"):
        med = float(np.median(g["pred"]))
        for s, p in zip(g["symbol"], g["pred"]):
            out[(str(s), int(t))] = ON * min(1.0, float(np.exp(med - p)))
    return out


def weights(trades: Iterable[Trade], sizes: dict[Trade, float], index: pd.Index) -> pd.Series:
    """One contract's hourly target weights (index: open times): a bar holds the weight decided at the last
    decision time at or before its close."""
    o = index.to_numpy(dtype=np.int64)
    decided = (o + H) // DAY * DAY
    w = np.zeros(len(o))
    for trade in trades:
        w[(decided >= trade.entry) & (decided < trade.exit)] = trade.side * sizes[trade]
    return pd.Series(w, index=index)


def trade_returns(trades: Sequence[Trade], curve: pd.Series, *, slot: float) -> list[float]:
    """Each trade's money made (fees, funding, slippage and stops included) over one slot: the contract's equity
    from before the entry to two hours after the exit decision (the exit fills at the next bar's open)."""
    out = []
    for trade in trades:
        before = curve[curve.index < trade.entry]
        after = curve[curve.index <= trade.exit + 2 * H]
        start = float(before.iloc[-1]) if len(before) else slot
        out.append((float(after.iloc[-1]) - start) / slot)
    return out


def account_daily(curves: dict[str, pd.Series], *, capital: float, slot: float, days: np.ndarray) -> pd.Series:
    """The account's daily returns: ``capital`` plus every contract's money made (each contract's replay starts with
    one slot), at each day's last equity; a contract adds nothing before it trades and keeps its last equity after."""
    index = pd.Index(np.asarray(days, dtype=np.int64))
    cols = {s: c.groupby(c.index.to_numpy(dtype=np.int64) // DAY * DAY).last().reindex(index).ffill()
            for s, c in curves.items() if len(c)}
    made = (pd.DataFrame(cols, index=index).fillna(slot) - slot).sum(axis=1) if cols else pd.Series(0.0, index=index)
    return (capital + made).pct_change().dropna()


def filter_value(returns: np.ndarray, accepted: np.ndarray, years: np.ndarray) -> dict:
    """The main test of the README: accepted trades make more than rejected ones (Welch t >= 2), the difference is
    positive in at least 3 of 4 years, and each side has at least 50 trades."""
    from scipy.stats import ttest_ind
    r, a, y = np.asarray(returns, float), np.asarray(accepted, bool), np.asarray(years)
    diff = float(r[a].mean() - r[~a].mean()) if a.any() and (~a).any() else float("nan")
    t = float(ttest_ind(r[a], r[~a], equal_var=False).statistic) if a.sum() > 1 and (~a).sum() > 1 else float("nan")
    by_year = {int(v): float(r[a & (y == v)].mean() - r[~a & (y == v)].mean())
               for v in sorted(set(y.tolist())) if (a & (y == v)).any() and (~a & (y == v)).any()}
    checks = {"difference_above_0": diff > 0, "welch_t_at_least_2": t >= 2,
              "years_at_least_3": sum(v > 0 for v in by_year.values()) >= 3,
              "each_side_at_least_50": int(a.sum()) >= 50 and int((~a).sum()) >= 50}
    return {"difference": diff, "welch_t": t, "by_year": by_year, "accepted": int(a.sum()), "rejected": int((~a).sum()),
            "checks": checks, "value": all(checks.values())}


def walk_forward_predictions(x: pd.DataFrame, y: np.ndarray, times: np.ndarray, label_end: np.ndarray,
                             years: Sequence[int], *, kind: str, model: str) -> np.ndarray:
    """``lab.panel_study.evaluate``'s yearly walk-forward predictions for every row of a test year, labelled or not
    (a strategy decides before its label exists); NaN for the first year (training only) and outside the years."""
    known = np.flatnonzero(np.isfinite(y))
    pred = np.full(len(y), np.nan)
    for year in years[1:]:
        start, stop = st.year_start(year), st.year_start(year + 1)
        tr = known[label_end[known] < start]
        te = np.flatnonzero((times >= start) & (times < stop))
        if not len(tr) or not len(te):
            continue
        pred[te] = ps._fit_predict(x.iloc[tr], y[tr], ps._weights(times[tr]), x.iloc[te], kind, model)
    return pred


def median_volumes(volumes: dict[str, pd.Series], first_bar: dict[str, int], times: np.ndarray) -> pd.DataFrame:
    """The volume ``lab.panel_study.liquid_universe`` ranks by: the median daily quote volume of the 30 days before
    each time (25 needed), missing for a contract listed less than 60 days before."""
    medians = {}
    for symbol, v in volumes.items():
        m = fp._last(v.rolling(30, min_periods=25).median(), times, DAY)
        m[times - first_bar[symbol] < 60 * DAY] = np.nan
        medians[symbol] = m
    return pd.DataFrame(medians, index=pd.Index(times))

"""The research lab's judge as a tool (research/lab/README.md). It is the orchestration of the first run's script
(research/lab/runs/2026-11-16/judge.py) with the snapshot, the capital, the slots, the periods and the stop as
parameters; the method is the one written in research/lab/runs/2026-11-16/ideas.md before any result.

- Periods: the first only selects, the middle ones are the walk-forward folds, the last is the holdout.
- The symbol list changes at the start of each period and uses only bars from before that date (``universe``).
- A configuration is an idea's parameter set plus a ``StopVariant`` (the stop distance and what follows a stop).
  Every contract is replayed through the shared broker (``lab.um_lab.replay``), so this module charges no cost
  itself (tests/architecture/test_research_fills.py).
- Each period starts flat with fresh slot capital; its daily returns run from that capital to the period's last
  close, one per UTC day (``period_daily``), so consecutive periods join without an overlap or a gap.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal as D
from pathlib import Path

import pandas as pd

from lab import um_lab as lab

H, DAY = lab.H, lab.DAY
BTC = "BTCUSDT"
STABLE = ("USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT")


def _date(t: int) -> str:
    return pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d")


# -- periods ---------------------------------------------------------------------------------------------------
def periods(start: int = lab.START, folds: Sequence[tuple[int, int]] = lab.FOLDS, end: int = lab.END, *,
            history_days: int = 30) -> tuple[tuple[int, int], ...]:
    """(start + the days the universe rule needs, first fold), every fold, (last fold's end, ``end``): the first
    period only selects, the folds are the walk-forward, the last period is the holdout."""
    out = ((start + history_days * DAY, folds[0][0]), *(tuple(f) for f in folds), (folds[-1][1], end))
    if any(t % DAY for period in out for t in period):
        raise ValueError("a period starts and ends at a UTC midnight")
    if any(a >= b for a, b in out):
        raise ValueError(f"empty period in {[tuple(map(_date, p)) for p in out]}")
    if any(previous[1] != following[0] for previous, following in zip(out, out[1:])):
        raise ValueError("each period must follow the one before it without an overlap or a gap")
    return out


PERIODS = periods()


# -- the universe rule -----------------------------------------------------------------------------------------
def universe(snapshot: Path, dates: Sequence[int], *, capital: str, slots: int, stable: Sequence[str] = STABLE,
             history_days: int = 30, min_days: int = 25) -> list[list[str]]:
    """For each date the ``slots`` contracts with the highest median daily quote volume over the ``history_days``
    days before it, among those with at least ``min_days`` of them, whose smallest order (the exchange minimum or
    one lot at the last close before the date) is at most 2/3 of a slot, stablecoin contracts left out. Only bars
    that opened before the date count."""
    slot = float(D(capital) / slots)
    rows = []
    for symbol in lab.contracts(snapshot):
        if symbol in stable:
            continue
        df = pd.read_csv(snapshot / f"{symbol}_1h.csv.gz", usecols=["open_time", "close", "quote_volume"])
        df = df.set_index("open_time").sort_index()
        rules = lab.rules(symbol, snapshot)
        for k, date in enumerate(dates):
            past = df[(df.index >= date - history_days * DAY) & (df.index < date)]
            daily = past["quote_volume"].groupby(past.index // DAY).sum()
            if len(daily) < min_days:
                continue
            smallest = max(float(rules.min_notional), float(rules.step) * float(past["close"].iloc[-1]))
            if smallest <= slot * 2 / 3:
                rows.append((k, symbol, float(daily.median())))
    table = pd.DataFrame(rows, columns=["period", "symbol", "quote"])
    return [table[table["period"] == k].nlargest(slots, "quote")["symbol"].tolist() for k in range(len(dates))]


def universe_record(lists: Sequence[Sequence[str]], dates: Sequence[int], *, capital: str, slots: int) -> dict:
    """What a run writes next to its results (universe.json)."""
    slot = float(D(capital) / slots)
    return {"rule": f"top {slots} by median daily quote volume of the 30 days before each date; smallest order <= "
                    f"{slot * 2 / 3:.0f} USDT; no stablecoin contracts",
            "dates": [_date(t) for t in dates], "lists": [list(symbols) for symbols in lists]}


# -- the stop: part of a configuration -------------------------------------------------------------------------
@dataclass(frozen=True)
class StopVariant:
    """The exchange stop of every position and what follows a stop: a fixed pause (``pause_hours``, None for no
    pause) or waiting for the idea's target to go back to 0 (``reenter="signal_reset"``, lab.simulate). A stop of
    "0" is no stop: a diagnostic only, the live runtime always has one."""
    stop_pct: str = lab.LIVE.stop_pct
    pause_hours: int | None = lab.LIVE.pause_hours
    reenter: str = "pause"

    def __post_init__(self) -> None:
        stop = D(self.stop_pct)
        if self.reenter not in ("pause", "signal_reset") or stop < 0:
            raise ValueError(f"not a stop variant: {self}")
        if self.reenter == "signal_reset" and (self.pause_hours is not None or stop == 0):
            raise ValueError(f"signal_reset needs a stop and takes no pause: {self}")
        if stop == 0 and self.pause_hours is not None:
            raise ValueError(f"no stop, nothing to pause after: {self}")

    @property
    def name(self) -> str:
        if D(self.stop_pct) == 0:
            return "nostop"
        after = "signal_reset" if self.reenter == "signal_reset" else f"pause{self.pause_hours or 0}h"
        return f"stop{self.stop_pct}_{after}"

    def replay_terms(self) -> dict:
        """The keywords of ``lab.um_lab.replay`` for this variant."""
        return {"stop_pct": self.stop_pct, "pause_ms": self.pause_hours * H if self.pause_hours else None,
                "reenter_after_stop": self.reenter}


LIVE_STOP = StopVariant()                                        # the runtime's: 8 %, entries pause 4 hours
# research/lab/runs/2026-11-17/plan.md: 3 distances x (3 pauses, signal reset); the live variant first (ties go to
# the variant listed first)
STOP_VARIANTS = tuple(StopVariant(stop, hours) if hours else StopVariant(stop, None, "signal_reset")
                      for stop in ("0.08", "0.12", "0.15") for hours in (4, 24, 72, None))


# -- one period's daily returns --------------------------------------------------------------------------------
def period_daily(curves: list[pd.Series], start: int, end: int, *, slots: int, slot_capital: float,
                 legacy: bool = False) -> pd.Series:
    """The account's return on each UTC day of [start, end): ``slots`` slots that start the period flat with
    ``slot_capital`` each. ``curves`` are the slots' equity at each bar's close (``lab.um_lab.replay``); the close
    at midnight belongs to the day that ended, and day one is measured from the fresh capital, so the returns
    compound to the account's value at the period's last close.

    ``legacy`` reproduces the 2026-11-16 script and nothing else: days end at 23:00 and the period's first day
    (the cost of entering included) and its last hour are left out."""
    if legacy:
        daily = lab.account_daily(curves, slots=slots, slot_capital=slot_capital)
        return daily.reindex(pd.RangeIndex(start // DAY + 1, end // DAY)).fillna(0.0)
    opening = pd.Series([slot_capital], index=[start - H])
    by_bar_open = [pd.concat([opening, c.set_axis(c.index - H)]) for c in curves if len(c)]
    daily = lab.account_daily(by_bar_open, slots=slots, slot_capital=slot_capital)
    return daily.reindex(pd.RangeIndex(start // DAY, end // DAY)).fillna(0.0)


__all__ = ["BTC", "LIVE_STOP", "PERIODS", "STABLE", "STOP_VARIANTS", "StopVariant", "period_daily", "periods",
           "universe", "universe_record"]

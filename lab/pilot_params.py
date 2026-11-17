"""The pilot's risk and execution settings on the frozen BTC/ETH overlay.

Owner 2026-11-17: "来开始探索交易的参数数值设置多少最佳", scope A: the settings a person writes in .env, not the
strategy's signal.

A setting: the position multiple (STRATEGY_LEVERAGE), the exchange stop (STOP_PCT) at the largest isolated exchange
leverage the runtime allows for it (at most 10, BINANCE_EXCHANGE_LEVERAGE), what follows a stop (the 4-hour pause, or
the pause and then a decided target of 0: the spec's after_stop) and the daily / weekly drawdown latches
(MAX_DAILY_DRAWDOWN_PCT / MAX_WEEKLY_DRAWDOWN_PCT). ``run`` steps BTC and ETH together hour by hour as the runtime
does; ``walk_forward`` picks each year's setting from the years before it with ``plateau_choice``: the best median
Sharpe of a setting and its neighbours, among the settings within the limits, not the single best.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal as D
from typing import Any

import numpy as np
import pandas as pd

from lab.simulate import Sleeve
from lab.um_lab import exchange_leverage, loss_stop_share
from trading.adapters.simulated.bar_broker import ReplayConfig, SymbolRules

HOUR = 3_600_000
DAY = 24 * HOUR
PAUSE_MS = 4 * HOUR                                          # STOP_PAUSE_SECONDS
MULTIPLES = (1.0, 1.5, 2.0, 2.5, 3.0)
STOPS = ("0.045", "0.08", "0.12")
AFTERS = ("pause", "pause_then_reset")
LATCHES = ((0.03, 0.12), (0.05, 0.12))
MAX_DRAWDOWN, MAX_LOSS_STOP_SHARE = 0.30, 0.10               # a setting beyond either is never chosen


@dataclass(frozen=True)
class Setting:
    multiple: float
    stop: str
    after_stop: str
    latch: tuple[float, float]

    @property
    def margin(self) -> int:
        return min(10, exchange_leverage(self.stop))

    @property
    def key(self) -> str:
        return (f"{self.multiple:g}x stop{self.stop}@{self.margin}x {self.after_stop} "
                f"latch{self.latch[0]:g}/{self.latch[1]:g}")


GRID = [Setting(m, s, a, latch) for m in MULTIPLES for s in STOPS for a in AFTERS for latch in LATCHES]
BASELINE = Setting(2.0, "0.045", "pause", (0.03, 0.12))      # the pilot's settings today (owner 2026-11-16)
_AXES = ((MULTIPLES, "multiple"), (STOPS, "stop"), (AFTERS, "after_stop"), (LATCHES, "latch"))


def neighbours(setting: Setting) -> list[Setting]:
    """The settings one step away in one of the four settings."""
    out = []
    for values, name in _AXES:
        k = values.index(getattr(setting, name))
        for j in (k - 1, k + 1):
            if 0 <= j < len(values):
                out.append(Setting(**{**setting.__dict__, name: values[j]}))
    return out


def stats(daily: pd.Series) -> dict:
    sd = float(daily.std())
    equity = (1 + daily).cumprod()
    return {"sharpe": float(daily.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0,
            "max_drawdown": float((1 - equity / equity.cummax()).max()), "total_return": float(equity.iloc[-1] - 1),
            "loss_stop_share": loss_stop_share(daily, 0.15, 90)}


def plateau_choice(found: Mapping[str, dict], grid: Sequence[Setting] = GRID, *, max_drawdown: float = MAX_DRAWDOWN,
                   max_loss_stop_share: float = MAX_LOSS_STOP_SHARE) -> Setting | None:
    """The setting within the limits whose own Sharpe and its neighbours' have the highest median (the first in grid
    order on a tie); None when no setting is within the limits."""
    within = [s for s in grid if found[s.key]["max_drawdown"] <= max_drawdown
              and (found[s.key]["loss_stop_share"] or 0.0) <= max_loss_stop_share]
    if not within:
        return None

    def score(s: Setting) -> float:
        return statistics.median(found[x.key]["sharpe"] for x in [s, *neighbours(s)])
    return max(within, key=score)


def _year(days: pd.Index) -> np.ndarray:
    return pd.to_datetime(np.asarray(days, dtype=np.int64) * DAY, unit="ms").year.to_numpy()


def walk_forward(dailies: Mapping[str, pd.Series], years: Sequence[int], grid: Sequence[Setting] = GRID) -> dict:
    """Each year after the first runs the setting ``plateau_choice`` picks from the years before it (the pilot's
    settings today when none is within the limits); the stitched daily returns and the choices."""
    parts, choices = [], {}
    for k, year in enumerate(years[1:], start=1):
        found = {}
        for s in grid:
            d = dailies[s.key]
            found[s.key] = stats(d[np.isin(_year(d.index), list(years[:k]))])
        chosen = plateau_choice(found, grid) or BASELINE
        choices[year] = chosen.key
        d = dailies[chosen.key]
        parts.append(d[_year(d.index) == year])
    return {"daily": pd.concat(parts), "choices": choices}


def _config(rules: SymbolRules, setting: Setting, capital: float, cost: int, ttl_bars: int) -> ReplayConfig:
    m = D(cost)
    return ReplayConfig(starting_balance=str(capital), maker_bps=str(2 * m), taker_bps=str(5 * m),
                        slippage_bps=str(3 * m), stop_slippage_bps=str(10 * m), participation="0.01",
                        qty_step=str(rules.step), min_notional=str(rules.min_notional), ttl_bars=ttl_bars,
                        entry_mode="market", isolated_leverage=str(setting.margin), max_exposure=str(setting.multiple),
                        maintenance_margin_rate="0.005", stop_pct=setting.stop)


def run(bars: Mapping[str, list], targets: Mapping[str, pd.DataFrame], funding: Mapping[str, dict], setting: Setting,
        *, rules: Mapping[str, SymbolRules], spec: Any, start: int, end: int, capital: float = 150.0,
        cost: int = 1) -> tuple[pd.DataFrame, dict, dict]:
    """BTC and ETH stepped together, hour by hour, sharing ``capital`` equally, with the runtime's rules: after a stop
    the symbol takes no entry for 4 hours and, with ``pause_then_reset``, until a ready decision since the stop wanted
    exactly 0 (one inside the pause counts); the daily and weekly latches on the whole account close everything and
    refuse entries until the UTC day (week, from Monday) ends, checked on bar closes. Returns the hourly curve
    (at, equity, notional), the counts per symbol and every symbol's fills."""
    symbols = list(bars)
    sleeves, maps, by_time = {}, {}, {}
    for symbol in symbols:
        sleeves[symbol] = Sleeve(symbol, spec, rules[symbol],
                                 _config(rules[symbol], setting, capital / len(symbols), cost, spec.ttl_bars))
        t = targets[symbol]
        maps[symbol] = dict(zip(t["available_at"].astype(np.int64).tolist(),
                                zip((t["target"] * setting.multiple).tolist(), t["ready"].tolist())))
        for bar in bars[symbol]:
            if start < bar.available_at and bar.open_time < end:
                by_time.setdefault(bar.open_time, {})[symbol] = bar
    paused, waiting = {s: 0 for s in symbols}, {s: False for s in symbols}
    day_base = week_base = capital
    day = week = None
    latched_until, latch_trips, rows = 0, 0, []
    for open_time in sorted(by_time):
        step = by_time[open_time]
        if len(step) < len(symbols):
            continue                                         # every symbol on the same hour, as the runtime has them
        for symbol, bar in step.items():
            sleeve = sleeves[symbol]
            before = len(sleeve.broker.fills)
            rate = funding[symbol][bar.open_time][0] if bar.open_time in funding[symbol] else None
            sleeve.on_bar(bar, rate)
            if any(f["order"] == "protective_stop" for f in sleeve.broker.fills[before:]):
                paused[symbol] = bar.available_at + PAUSE_MS
                waiting[symbol] = setting.after_stop == "pause_then_reset"
        close = next(iter(step.values())).available_at
        equity = sum(float(s.broker.equity) for s in sleeves.values())
        this_day, this_week = (close - 1) // DAY, ((close - 1) // DAY + 3) // 7
        if this_day != day:
            day, day_base = this_day, equity if day is not None else capital
        if this_week != week:
            week, week_base = this_week, equity if week is not None else capital
        daily_pct, weekly_pct = setting.latch
        if close >= latched_until and (equity <= day_base * (1 - daily_pct) or equity <= week_base * (1 - weekly_pct)):
            weekly = equity <= week_base * (1 - weekly_pct)
            latched_until = ((this_week + 1) * 7 - 3) * DAY + 1 if weekly else (this_day + 1) * DAY + 1
            latch_trips += 1
        for symbol, bar in step.items():
            target, ready = maps[symbol].get(bar.available_at, (0.0, False))
            if waiting[symbol] and ready and target == 0.0:
                waiting[symbol] = False
            qty = float(sleeves[symbol].broker.position.qty)
            if close < latched_until:
                target, ready = 0.0, True
            elif bar.available_at < paused[symbol] or waiting[symbol]:
                target = 0.0 if target * qty <= 0 or qty == 0 else target
            sleeves[symbol].decide(bar, target, ready)
        rows.append((close, equity, sum(abs(float(s.broker.position.qty) * float(s.broker.mark))
                                        for s in sleeves.values())))
    curve = pd.DataFrame(rows, columns=["at", "equity", "notional"])
    counts: dict = {s: {"fills": len(v.broker.fills),
                        "stops": sum(1 for f in v.broker.fills if f["order"] == "protective_stop"),
                        "liquidations": len(v.broker.liquidations),
                        "entries": sum(1 for o in v.broker.orders if not o["reduce_only"])} for s, v in sleeves.items()}
    counts["latch_trips"] = latch_trips
    return curve, counts, {s: list(v.broker.fills) for s, v in sleeves.items()}

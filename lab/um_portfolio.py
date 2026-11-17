"""The lab's next round (tech lead 2026-11-17): a benchmark of holding BTC with position control, and a portfolio of
several ideas chosen fold by fold instead of one champion.

- ``vol_target_hold``: BTCUSDT long, position = min(0.95, target / the standard deviation of the last ``days`` daily
  returns), set from each completed day (E3's sizing without its trend filter), through the shared broker with the
  run's costs and no stop. The bar an idea must also beat.
- ``walk_forward_portfolio``: for each fold, the ideas whose configuration chosen for that fold (by the judge, from
  the periods before it) made money on those earlier periods and beat every benchmark there; at most ``k`` of them,
  the best Sharpe first. Equal capital at the fold's start, not rebalanced inside the fold; no member, no position.
  Nothing from the fold or after it chooses its members.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lab import um_judge as judge
from lab import um_lab as lab
from trading.domain.strategy.ideas import ON, _hours

DAY, H = lab.DAY, lab.H


def vol_target_weights(snapshot: Path, *, target: float = 0.02, days: int = 30, symbol: str = judge.BTC) -> pd.Series:
    df = lab.frame(symbol, snapshot)
    close = df["close"].groupby(df.index // DAY).last()
    size = (target / close.pct_change().rolling(days).std()).clip(upper=ON)
    return _hours(df, size)


def vol_target_hold(snapshot: Path, periods: Sequence[tuple[int, int]], *, capital: str, target: float = 0.02,
                    days: int = 30) -> dict[int, pd.Series]:
    """{period: daily returns} of the benchmark, each period starting flat with ``capital``."""
    weights = vol_target_weights(snapshot, target=target, days=days)
    out = {}
    for k, (start, end) in enumerate(periods):
        replayed = lab.replay(judge.BTC, weights, start=start, end=end, snapshot=snapshot, starting_balance=capital)
        out[k] = judge.period_daily([replayed["curve"]], start, end, slots=1, slot_capital=float(capital))
    return out


def combine(dailies: Sequence[pd.Series]) -> pd.Series:
    """Equal capital in each series at the start, never rebalanced: the portfolio's daily returns."""
    if not dailies:
        return pd.Series(dtype=float)
    values = pd.concat([(1 + d).cumprod() for d in dailies], axis=1).mean(axis=1)
    out = values.pct_change()
    out.iloc[0] = values.iloc[0] - 1.0
    return out


Runs = list[tuple[judge.Config, Mapping[int, dict]]]


def walk_forward_portfolio(runs: Mapping[str, Runs], periods: Sequence[tuple[int, int]], folds: Sequence[int],
                           benchmarks: Sequence[Mapping[int, pd.Series]], *, eligible: Callable[[str], bool],
                           k: int = 5) -> dict[str, Any]:
    """``runs``: idea -> its configurations and their period results (the judge's cache); ``periods``: the run's
    periods; ``benchmarks``: {period: daily returns} each. Returns the members per fold, every eligible idea's
    chosen configuration per fold, and the portfolio's daily returns over the folds."""
    pieces, members, chosen, by_key = [], [], {}, {}
    for name, configs in runs.items():
        if eligible(name):
            chosen[name] = judge.walk_forward(list(configs), list(folds))["chosen"]
            by_key[name] = {config.key: results for config, results in configs}
    for i, fold in enumerate(folds):
        before = range(fold)
        bars = [lab.summary(pd.concat([b[j] for j in before])).get("sharpe", float("-inf")) if fold else float("inf")
                for b in benchmarks]
        ranked = []
        for name, configs in chosen.items():
            prior = lab.summary(judge.joined(by_key[name][configs[i].key], before))
            if prior.get("days") and prior["total_return"] > 0 and all(prior["sharpe"] > bar for bar in bars):
                ranked.append((-prior["sharpe"], name))
        picked = [name for _, name in sorted(ranked)[:k]]
        members.append(picked)
        days = pd.RangeIndex(periods[fold][0] // DAY, periods[fold][1] // DAY)
        piece = combine([by_key[n][chosen[n][i].key][fold]["daily"] for n in picked])
        pieces.append(piece.reindex(days).fillna(0.0))
    return {"members": members, "chosen": {n: [c.label for c in cs] for n, cs in chosen.items()},
            "daily": pd.concat(pieces) if pieces else pd.Series(dtype=float)}


__all__ = ["combine", "vol_target_hold", "vol_target_weights", "walk_forward_portfolio"]

"""The research lab's harness over every USDT-margined perpetual (tech lead 2026-11-16; research/lab/README.md).

A strategy idea is a function from one contract's hourly frame (and a context: BTC's frame, the funding series,
the parameters) to target weights. Every contract is replayed through the shared conservative broker
(``lab.simulate``: VIP0 fees, slippage, participation cap, real funding), so this module charges no cost itself
(tests/architecture/test_research_fills.py). Contracts join the portfolio when they are liquid enough and leave
when they are delisted (the broker closes the position at the last price).

Windows (research/lab/README.md): selection 2022-11-13..2026-05-12 with yearly walk-forward folds; the last six
months 2026-05-13..2026-11-12 are not used to choose anything and answer "does it still win on data it never
saw". Every configuration tested is appended to research/lab/trials.jsonl.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal as D
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec
from trading.domain.bars import bar_from_kline

H = 3_600_000
DAY = 86_400_000
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data_files" / "binance_usdt_perps_2026-11-12"
TRIALS = ROOT / "research" / "lab" / "trials.jsonl"


def ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


START, SELECT_END, END = ms("2022-11-13"), ms("2026-05-13"), ms("2026-11-13")
FOLDS = ((ms("2023-11-13"), ms("2024-11-13")), (ms("2024-11-13"), ms("2025-11-13")), (ms("2025-11-13"), SELECT_END))
HOLDOUT = (SELECT_END, END)
DEFAULT_RULES = SymbolRules(D("0.001"), D("0.001"), D("5"))   # a delisted contract has no published filters

Idea = Callable[[pd.DataFrame, dict], pd.Series]               # hourly frame, context -> target weight per bar


# -- data ------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def manifest(snapshot: Path = SNAPSHOT) -> dict:
    return json.loads((snapshot / "manifest.json").read_text())


@lru_cache(maxsize=None)
def exchange_info(snapshot: Path = SNAPSHOT) -> dict:
    path = snapshot / "exchange_info.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def contracts(snapshot: Path = SNAPSHOT, *, kind: str | None = None) -> list[str]:
    rows = manifest(snapshot)["contracts"]
    return sorted(r["symbol"] for r in rows if r["files"] and (kind is None or r["type"] == kind))


@lru_cache(maxsize=64)
def frame(symbol: str, snapshot: Path = SNAPSHOT) -> pd.DataFrame:
    """Hourly bars indexed by open time (ms), with ``funding`` (the rate settled at that hour, else 0)."""
    df = pd.read_csv(snapshot / f"{symbol}_1h.csv.gz").set_index("open_time").sort_index()
    funding_path = snapshot / f"{symbol}_funding.csv.gz"
    df["funding"] = 0.0
    if funding_path.is_file():
        fund = pd.read_csv(funding_path)
        hours = (fund["funding_time"].to_numpy(np.int64) // H) * H
        rates = pd.Series(fund["rate"].to_numpy(float), index=hours).groupby(level=0).sum()
        df["funding"] = rates.reindex(df.index).fillna(0.0).to_numpy()
    return df


def rules(symbol: str, snapshot: Path = SNAPSHOT) -> SymbolRules:
    info = exchange_info(snapshot).get(symbol)
    if not isinstance(info, dict) or "step" not in info:
        return DEFAULT_RULES
    step = D(str(info["step"]))
    return SymbolRules(step, step, D(str(info["min_notional"])))


def liquid(symbol: str, *, min_daily_quote: float, snapshot: Path = SNAPSHOT) -> pd.Series:
    """True for each hour after which the median daily quote volume of the previous 30 days is at least
    ``min_daily_quote`` (known at that hour's close)."""
    df = frame(symbol, snapshot)
    daily = df["quote_volume"].groupby(df.index // DAY).sum()
    median = daily.rolling(30, min_periods=30).median().shift(1)          # yesterday and before only
    return pd.Series((median.reindex(df.index // DAY).to_numpy() >= min_daily_quote), index=df.index)


# -- one contract through the shared broker --------------------------------------------------------------------
def replay(symbol: str, weights: pd.Series, *, cost_multiplier: int = 1, start: int = START, end: int = END,
           snapshot: Path = SNAPSHOT) -> dict:
    """Target weights (decided at each bar's close, index = open time) through the shared broker. Returns the
    hourly equity curve and the fills."""
    df = frame(symbol, snapshot)
    df = df[(df.index >= start - 40 * DAY) & (df.index < end)]
    bars = [bar_from_kline({"start": int(t), "open": str(r.open), "high": str(r.high), "low": str(r.low),
                            "close": str(r.close), "volume": str(r.volume)}) for t, r in zip(df.index, df.itertuples())]
    w = weights.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
    targets = pd.DataFrame({"available_at": df.index.to_numpy(np.int64) + H, "target": w.to_numpy(float),
                            "ready": np.ones(len(df), dtype=bool)})
    funding = {int(t): (D(repr(float(r))),) for t, r in df["funding"].items() if r != 0.0}
    result = simulate(symbol, bars, targets, funding, StrategySpec(execution="market"),
                      start_ms=max(start, int(df.index[0]) if len(df) else start), end_ms=end,
                      rules=rules(symbol, snapshot), cost_multiplier=cost_multiplier)
    curve = result.curve.set_index("at")["equity"]
    return {"symbol": symbol, "curve": curve, "fills": [_fill(f) for f in result.fills]}


def _fill(f: Any) -> dict:
    get = (lambda k: getattr(f, k, None)) if not isinstance(f, dict) else f.get
    return {"at": get("at"), "side": get("side"), "qty": str(get("qty")), "price": str(get("price"))}


# -- portfolio and metrics -------------------------------------------------------------------------------------
def portfolio_daily(curves: list[pd.Series]) -> pd.Series:
    """Daily returns of equal-capital sleeves: each day's money change over the money invested the day before,
    counting only the sleeves that existed the day before (a contract that lists later joins with fresh capital)."""
    if not curves:
        return pd.Series(dtype=float)
    days = [c.groupby(c.index // DAY).last() for c in curves if len(c)]
    panel = pd.concat(days, axis=1).sort_index()
    previous = panel.shift(1)
    change = (panel - previous).where(previous.notna()).sum(axis=1)
    base = previous.sum(axis=1)
    return (change / base).where(base > 0).dropna()


def summary(daily: pd.Series) -> dict:
    if daily.empty:
        return {"days": 0}
    equity = (1 + daily).cumprod()
    years = len(daily) / 365.0
    sd = float(daily.std())
    monthly = (1 + daily).groupby(pd.to_datetime(daily.index * DAY, unit="ms").to_period("M")).prod() - 1
    losing = (daily < 0).astype(int)
    streak = int(losing.groupby((losing != losing.shift()).cumsum()).cumsum().max())
    return {"days": int(len(daily)), "total_return": float(equity.iloc[-1] - 1),
            "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else -1.0,
            "sharpe": float(daily.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0,
            "max_drawdown": float((1 - equity / equity.cummax()).max()),
            "worst_month": float(monthly.min()), "worst_month_label": str(monthly.idxmin()),
            "longest_losing_days": streak}


def regimes(btc: pd.DataFrame) -> pd.Series:
    """Per day: bear (BTC down more than 10 % over 30 days), chop (within 5 %) or bull, known at the day's close."""
    close = btc["close"].groupby(btc.index // DAY).last()
    change = close / close.shift(30) - 1
    return pd.Series(np.where(change < -0.10, "bear", np.where(change.abs() <= 0.05, "chop", "other")),
                     index=close.index).where(change.notna())


def by_regime(daily: pd.Series, btc: pd.DataFrame) -> dict:
    labels = regimes(btc).reindex(daily.index)
    return {name: float((1 + daily[labels == name]).prod() - 1) for name in ("bear", "chop")}


def window(daily: pd.Series, start: int, end: int) -> pd.Series:
    return daily[(daily.index >= start // DAY) & (daily.index < end // DAY)]


def buy_and_hold(symbol: str, start: int, end: int, snapshot: Path = SNAPSHOT) -> pd.Series:
    close = frame(symbol, snapshot)["close"]
    daily = close.groupby(close.index // DAY).last()
    return window(daily.pct_change().dropna(), start, end)


# -- walk-forward selection ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Run:
    params: dict
    daily: pd.Series                     # the portfolio's daily returns over the whole span, one parameter set


def walk_forward(runs: list[Run]) -> dict:
    """For each fold, the parameter set with the best Sharpe on all selection data before the fold is used on the
    fold. The holdout uses the set chosen on all selection data. Nothing after a fold's start chooses for it."""
    pieces, chosen = [], []
    for fold_start, fold_end in FOLDS:
        best = max(runs, key=lambda r: (summary(window(r.daily, START, fold_start)).get("sharpe", -9), str(r.params)))
        chosen.append(best.params)
        pieces.append(window(best.daily, fold_start, fold_end))
    final = max(runs, key=lambda r: (summary(window(r.daily, START, SELECT_END)).get("sharpe", -9), str(r.params)))
    wf = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return {"chosen_per_fold": chosen, "final_params": final.params, "walk_forward": summary(wf),
            "walk_forward_daily": wf, "holdout": summary(window(final.daily, *HOLDOUT)),
            "holdout_daily": window(final.daily, *HOLDOUT), "final": final}


# -- running an idea -------------------------------------------------------------------------------------------
def _replay_job(job: tuple) -> dict:
    symbol, idea, params, cost_multiplier, min_daily_quote, snapshot = job
    df = frame(symbol, snapshot)
    context = {"params": params, "btc": frame("BTCUSDT", snapshot), "symbol": symbol}
    weights = idea(df, context).reindex(df.index).fillna(0.0)
    if min_daily_quote:
        weights = weights.where(liquid(symbol, min_daily_quote=min_daily_quote, snapshot=snapshot), 0.0)
    if not (weights != 0).any():
        return {"symbol": symbol, "curve": pd.Series(dtype=float), "fills": []}
    return replay(symbol, weights, cost_multiplier=cost_multiplier, snapshot=snapshot)


def run_idea(idea: Idea, params: dict, symbols: list[str], *, cost_multiplier: int = 1, min_daily_quote: float = 0,
             processes: int = 2, snapshot: Path = SNAPSHOT) -> dict:
    """One parameter set of an idea over ``symbols``: per-contract replays, the portfolio's daily returns and the
    number of fills."""
    jobs = [(s, idea, params, cost_multiplier, min_daily_quote, snapshot) for s in symbols]
    if processes > 1:
        with get_context("fork").Pool(processes) as pool:
            results = pool.map(_replay_job, jobs)
    else:
        results = [_replay_job(j) for j in jobs]
    active = [r for r in results if len(r["curve"])]
    return {"daily": portfolio_daily([r["curve"] for r in active]), "fills": sum(len(r["fills"]) for r in active),
            "contracts_traded": len(active), "per_symbol": {r["symbol"]: len(r["fills"]) for r in active}}


def record(rows: list[dict], path: Path = TRIALS) -> None:
    """Append every tested configuration to the lab's trial registry (never rewritten)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


__all__ = ["DAY", "END", "FOLDS", "H", "HOLDOUT", "Run", "SELECT_END", "START", "buy_and_hold", "by_regime",
           "contracts", "frame", "liquid", "manifest", "portfolio_daily", "record", "regimes", "replay", "rules",
           "run_idea", "summary", "walk_forward", "window"]

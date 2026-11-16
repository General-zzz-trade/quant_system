"""The research lab's harness over every USDT-margined perpetual (tech lead 2026-11-16; research/lab/README.md).

A strategy idea is a function from one contract's hourly frame (and a context: BTC's frame, the funding series,
the parameters) to target weights. Every contract is replayed through the shared conservative broker
(``lab.simulate``: VIP0 fees, slippage, participation cap, real funding), so this module charges no cost itself
(tests/architecture/test_research_fills.py). Contracts join the portfolio when they are liquid enough and leave
when they are delisted (the broker closes the position at the last price).

Windows (research/lab/README.md): selection 2022-11-13..2026-05-12 with yearly walk-forward folds; the last six
months 2026-05-13..2026-11-12 are not used to choose anything and answer "does it still win on data it never
saw". Every configuration tested is appended to research/lab/trials.jsonl.

Live terms (``LiveTerms``, tech lead 2026-11-16: the backtest follows the live runtime): one strategy capital split
equally over the strategy's fixed symbol list (the runtime's 1/N allocation; an order below the exchange minimum is
not sent), the runtime's exchange stop, an entry pause after a stop that ends by itself, and the account's daily
returns over the whole capital (idle slots hold cash). The report gives what the order limits must allow (the tech
lead sets the live limits from it) and how often the pilot's hard stops would have fired.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
from trading.domain.issues import STOP_PAUSE_SECONDS

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


@dataclass(frozen=True)
class LiveTerms:
    """What the live runtime does that the backtest follows (tech lead 2026-11-16)."""
    capital: str = "300"          # STRATEGY_CAPITAL_USDT of the small live pilot
    stop_pct: str = "0.08"        # the runtime's exchange stop (the entry point's --stop-pct default)
    pause_hours: int = STOP_PAUSE_SECONDS // 3600   # the live pause after a stop ends by itself (4 hours)
    daily_stop: float = 0.03      # the pilot stops for the day at -3 % (reported, from daily closes)
    losing_trades: int = 3        # the pilot's rule until 2026-11-16, replaced by loss_stop; still reported
    loss_stop: float = 0.15       # the pilot flattens and stops once the strategy has lost 15 % of its capital
    loss_stop_days: int = 90      # reported: the share of pilots of this length that the loss stop would end


LIVE = LiveTerms()


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
           snapshot: Path = SNAPSHOT, starting_balance: str = "5000", stop_pct: str = "0",
           pause_ms: int | None = None) -> dict:
    """Target weights (decided at each bar's close, index = open time) through the shared broker. Returns the
    hourly equity curve, the fills and the orders sent (time, notional at the reference price, reduce-only)."""
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
                      rules=rules(symbol, snapshot), cost_multiplier=cost_multiplier,
                      starting_balance=starting_balance, stop_pct=stop_pct, pause_after_stop_ms=pause_ms)
    curve = result.curve.set_index("at")["equity"]
    orders = [(int(o["submitted_at"]), float(D(o["qty"]) * D(o["reference"])), bool(o["reduce_only"]))
              for o in result.order_rows if o.get("rejection") != "below trading minimum"]
    return {"symbol": symbol, "curve": curve, "fills": [_fill(f) for f in result.fills], "orders": orders}


def _fill(f: Any) -> dict:
    get = (lambda k: getattr(f, k, None)) if not isinstance(f, dict) else f.get
    return {"at": get("at"), "side": get("side"), "qty": str(get("qty")), "price": str(get("price")),
            "fee": str(get("fee") or 0), "order": get("order")}


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


def account_daily(curves: list[pd.Series], *, slots: int, slot_capital: float) -> pd.Series:
    """Daily returns of the live account: ``slots`` equal slots of ``slot_capital``. A slot holds cash before its
    contract trades (or when it never does) and keeps its last equity after the contract ends."""
    days = [c.groupby(c.index // DAY).last() for c in curves if len(c)]
    if not days:
        return pd.Series(dtype=float)
    panel = pd.concat(days, axis=1).sort_index().ffill().fillna(slot_capital)
    total = panel.sum(axis=1) + (slots - panel.shape[1]) * slot_capital
    return total.pct_change().dropna()


def limits_needed(entries: list[tuple[int, float]]) -> dict:
    """What the live order limits (they count entries) must allow for these entry orders (time, notional)."""
    if not entries:
        return {"entry_orders": 0}
    df = pd.DataFrame(entries, columns=["at", "notional"])
    per_day = df.groupby(df["at"] // DAY)
    return {"entry_orders": len(df), "max_orders_per_day": int(per_day.size().max()),
            "p95_orders_per_active_day": float(per_day.size().quantile(0.95)),
            "max_daily_notional": float(per_day["notional"].sum().max()),
            "max_order_notional": float(df["notional"].max()),
            "max_orders_in_one_hour": int(df.groupby(df["at"] // H).size().max())}


def round_trips(fills: list[dict]) -> list[tuple[int, float]]:
    """(time closed, money made after fees, funding excluded) of each trade from flat back to flat. A fill through
    zero closes one trade and opens the next; its whole fee goes to the trade it closes."""
    position, flow, trips = D(0), 0.0, []
    for f in fills:
        sign, qty, price, fee = (1 if f["side"] == "buy" else -1), D(f["qty"]), float(f["price"]), float(f["fee"])
        if position * sign < 0 and qty > abs(position):              # through zero: close, then open the rest
            trips.append((int(f["at"]), flow - sign * float(abs(position)) * price - fee))
            position = sign * (qty - abs(position))
            flow = -sign * float(abs(position)) * price
            continue
        position += sign * qty
        flow += -sign * float(qty) * price - fee
        if position == 0:
            trips.append((int(f["at"]), flow))
            flow = 0.0
    return trips


def losing_streaks(trips: list[tuple[int, float]], n: int) -> list[int]:
    """Times at which ``n`` losing trades in a row completed (the count starts again after each)."""
    run, out = 0, []
    for at, money in sorted(trips):
        run = run + 1 if money < 0 else 0
        if run == n:
            out.append(at)
            run = 0
    return out


def loss_stop_share(daily: pd.Series, loss: float, days: int) -> float | None:
    """The share of pilots, one starting on each day with ``days`` days ahead of it, whose value falls ``loss``
    below its start within those days (the pilot's cumulative loss stop, from daily closes)."""
    value = (1 + daily).cumprod().to_numpy()
    if len(value) <= days:
        return None
    base = np.concatenate(([1.0], value[:-1]))
    hits = sum(bool((value[s:s + days] / base[s] <= 1 - loss).any()) for s in range(len(value) - days))
    return hits / (len(value) - days)


def live_report(results: list[dict], daily: pd.Series, terms: LiveTerms, slots: int) -> dict:
    trips = [t for r in results for t in round_trips(r["fills"])]
    streaks = losing_streaks(trips, terms.losing_trades)
    days = max(1, len(daily))
    return {"terms": asdict(terms), "slots": slots, "slot_capital": float(D(terms.capital) / slots),
            "limits_needed": limits_needed([(at, n) for r in results for at, n, reduce in r["orders"] if not reduce]),
            "round_trips": len(trips), "losing_trades": sum(1 for _, m in trips if m < 0),
            "exchange_stops": sum(1 for r in results for f in r["fills"] if f["order"] == "protective_stop"),
            "losing_streak_stops": len(streaks), "losing_streak_stops_per_30_days": len(streaks) * 30 / days,
            "losing_streak_stops_in_holdout": sum(1 for at in streaks if HOLDOUT[0] <= at < HOLDOUT[1]),
            "days_below_daily_stop": int((daily < -terms.daily_stop).sum()),
            "loss_stop_share": loss_stop_share(daily, terms.loss_stop, terms.loss_stop_days)}


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
    symbol, idea, params, cost_multiplier, min_daily_quote, snapshot, terms = job
    df = frame(symbol, snapshot)
    context = {"params": params, "btc": frame("BTCUSDT", snapshot), "symbol": symbol}
    weights = idea(df, context).reindex(df.index).fillna(0.0)
    if min_daily_quote:
        weights = weights.where(liquid(symbol, min_daily_quote=min_daily_quote, snapshot=snapshot), 0.0)
    if not (weights != 0).any():
        return {"symbol": symbol, "curve": pd.Series(dtype=float), "fills": [], "orders": []}
    return replay(symbol, weights, cost_multiplier=cost_multiplier, snapshot=snapshot, **terms)


def run_idea(idea: Idea, params: dict, symbols: list[str], *, cost_multiplier: int = 1, min_daily_quote: float = 0,
             processes: int = 2, snapshot: Path = SNAPSHOT, live: LiveTerms | None = None) -> dict:
    """One parameter set of an idea over ``symbols``: per-contract replays, the portfolio's daily returns and the
    number of fills. With ``live`` the symbols are the strategy's fixed list sharing ``live.capital`` equally, each
    replay has the live stop and pause, the daily returns are the whole account's, and ``live`` reports what the
    order limits must allow and how often the pilot's hard stops would have fired."""
    terms: dict = {}
    if live is not None:
        terms = {"starting_balance": str(D(live.capital) / len(symbols)), "stop_pct": live.stop_pct,
                 "pause_ms": live.pause_hours * H}
    jobs = [(s, idea, params, cost_multiplier, min_daily_quote, snapshot, terms) for s in symbols]
    if processes > 1:
        with get_context("fork").Pool(processes) as pool:
            results = pool.map(_replay_job, jobs)
    else:
        results = [_replay_job(j) for j in jobs]
    active = [r for r in results if len(r["curve"])]
    out = {"daily": portfolio_daily([r["curve"] for r in active]), "fills": sum(len(r["fills"]) for r in active),
           "contracts_traded": len(active), "per_symbol": {r["symbol"]: len(r["fills"]) for r in active}}
    if live is not None:
        out["daily"] = account_daily([r["curve"] for r in active], slots=len(symbols),
                                     slot_capital=float(D(live.capital) / len(symbols)))
        out["live"] = live_report(active, out["daily"], live, len(symbols))
    return out


def record(rows: list[dict], path: Path = TRIALS) -> None:
    """Append every tested configuration to the lab's trial registry (never rewritten)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


__all__ = ["DAY", "END", "FOLDS", "H", "HOLDOUT", "LIVE", "LiveTerms", "Run", "SELECT_END", "START",
           "account_daily", "buy_and_hold", "by_regime", "contracts", "frame", "limits_needed", "liquid",
           "live_report", "losing_streaks", "loss_stop_share", "manifest", "portfolio_daily", "record", "regimes",
           "replay", "round_trips", "rules", "run_idea", "summary", "walk_forward", "window"]

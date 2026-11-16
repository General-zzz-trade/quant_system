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

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal as D
from multiprocessing import get_context
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


def joined(periods: Mapping[int, dict], ks: Iterable[int]) -> pd.Series:
    """The daily returns of periods ``ks`` of one configuration, end to end."""
    parts = [periods[k]["daily"] for k in ks]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


# -- configurations and their replays --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """One configuration: an idea's parameter set, the stop variant and the cost multiplier of the broker."""
    idea: str
    params: dict
    stop: StopVariant = LIVE_STOP
    cost: int = 1

    @property
    def label(self) -> dict:
        return {"params": self.params, "stop": asdict(self.stop)}

    @property
    def key(self) -> str:
        return json.dumps(self.label, sort_keys=True)


# -- the walk-forward choice, the score and the rules (research/lab/runs/2026-11-16/ideas.md) ------------------
Run = tuple[Config, Mapping[int, dict]]                          # a configuration and its periods' results


def _stitch(picks: Sequence[dict]) -> dict:
    return {"daily": pd.concat([p["daily"] for p in picks]), "trips": [t for p in picks for t in p["trips"]],
            "entries": [e for p in picks for e in p["entries"]], "stops": sum(p["stops"] for p in picks)}


def walk_forward(runs: Sequence[Run], folds: Sequence[int], first: int = 0) -> dict:
    """Fold k runs the configuration with the best Sharpe over periods ``first`` .. k - 1: nothing from period k
    or later chooses for it. ``final`` is the best over every period up to the last fold (what the holdout gets).
    Ties: the larger text of the parameters (the 2026-11-16 rule), then the configuration listed first. Returns
    the choices and the chosen configurations' folds end to end."""
    def best(before: range) -> Run:
        def rank(i: int) -> tuple:
            config, results = runs[i]
            return lab.summary(joined(results, before)).get("sharpe", -9), str(config.params), -i
        return runs[max(range(len(runs)), key=rank)]
    chosen = [best(range(first, k)) for k in folds]
    final = best(range(first, folds[-1] + 1))
    return {"chosen": [config for config, _ in chosen], "final": final[0],
            **_stitch([results[k] for (_, results), k in zip(chosen, folds)])}


def longest_losing(trips: Sequence[tuple[int, float]]) -> int:
    """The longest run of losing round trips, in time order."""
    run = best = 0
    for _, money in sorted(trips):
        run = run + 1 if money < 0 else 0
        best = max(best, run)
    return best


def score(wf: dict, trips: Sequence[tuple[int, float]], regime: dict) -> dict:
    """The five 1-5 scores of ideas.md on the walk-forward and their total. Where the table names an edge it is
    kept as written (a drawdown of exactly 15 % is not "< 15 %"; "5 % or more" includes 5 %); an edge the table
    leaves open goes where the 2026-11-16 script put it (10 % a year scores 2, a 25 % drawdown scores 4)."""
    def above(x: float, edges: tuple) -> int:
        return sum(x > edge for edge in edges)
    bear, chop, drawdown = regime["bear"], regime["chop"], wf["max_drawdown"]
    worst = min(bear, chop)
    parts = {"return": 1 + above(wf["cagr"], (0, 0.10, 0.25, 0.50)),
             "drawdown": 5 if drawdown < 0.15 else 4 - above(drawdown, (0.25, 0.35, 0.50)),
             "trades": 1 + above(len(trips), (29, 49, 99, 199)),
             "losing_streak": 5 - above(longest_losing(trips), (3, 5, 7, 10)),
             "bear_and_chop": (5 if worst >= 0.05 else 4 if worst >= 0 else 3 if bear + chop >= 0
                               else 2 if max(bear, chop) >= 0 else 1)}
    return {**parts, "total": sum(parts.values())}


def survives(result: dict, hold: dict) -> bool:
    """Not eliminated: a return above 0 and a Sharpe above holding BTC over the same period, both after costs."""
    return bool(result.get("total_return", 0.0) > 0 and result.get("sharpe", 0.0) > hold["sharpe"])


def final_checks(*, holdout: dict, holdout_hold: dict, round_trips: int, doubled: dict, walk_forward_hold: dict,
                 selection_sharpe: Mapping[str, float], lookahead: Sequence[str] | None) -> dict:
    """The final check of a survivor (ideas.md): it still wins on the holdout, at least 50 round trips in the
    walk-forward, it still wins there with fees and slippage doubled, every configuration of the idea has a
    Sharpe above 0 on the selection periods, and the look-ahead check found nothing (not run counts as failed)."""
    return {"unseen_data": survives(holdout, holdout_hold), "at_least_50_trades": round_trips >= 50,
            "double_fees": survives(doubled, walk_forward_hold),
            "neighbouring_settings": all(v > 0 for v in selection_sharpe.values()),
            "no_lookahead": lookahead is not None and not lookahead}


Job = tuple[Config, tuple[int, ...]]                             # a configuration and the periods to replay
_WORKER: "Judge | None" = None                                   # the judge of a running pool (forked workers)


def _work(job: Job) -> dict:
    assert _WORKER is not None
    return _WORKER.run_config(job)


@dataclass(frozen=True)
class Judge:
    """One run's settings. ``ideas``: name -> (function, parameter grid) as lab.um_lab.Idea; ``lists``: one symbol
    list per period (``universe``); ``terms`` gives the capital and the pilot's limits, the stop is each
    configuration's ``StopVariant``. ``cross_sectional`` ideas read the period's list (``ctx["universe"]``).
    ``note`` goes into every registry row. ``legacy_days``: see ``period_daily``. ``lookahead``: the run's check of
    one idea and parameter set with lab/causality.py, returning the violations (``final`` fails without it)."""
    ideas: Mapping[str, tuple[Callable, list[dict]]]
    lists: Sequence[Sequence[str]]
    cache: Path
    run_date: str
    snapshot: Path = lab.SNAPSHOT
    terms: lab.LiveTerms = lab.LIVE
    slots: int = 10
    periods: tuple[tuple[int, int], ...] = PERIODS
    cross_sectional: frozenset = frozenset()
    trials: Path = lab.TRIALS
    note: str = ""
    legacy_days: bool = False
    lookahead: Callable[[str, dict], list[str]] | None = None

    def __post_init__(self) -> None:
        if len(self.lists) != len(self.periods):
            raise ValueError(f"one symbol list per period: {len(self.lists)} lists, {len(self.periods)} periods")
        if any(len(symbols) > self.slots for symbols in self.lists):
            raise ValueError(f"a list has more symbols than slots ({self.slots})")

    @property
    def slot_capital(self) -> float:
        return float(D(self.terms.capital) / self.slots)

    @property
    def holdout(self) -> int:
        return len(self.periods) - 1

    @property
    def selection(self) -> range:
        return range(self.holdout)

    @property
    def folds(self) -> range:
        return range(1, self.holdout)

    def candidates(self, name: str, variants: Sequence[StopVariant] = (LIVE_STOP,)) -> list[Config]:
        """Every parameter set of an idea with every stop variant, in the order ties are broken."""
        return [Config(name, params, variant) for params in self.ideas[name][1] for variant in variants]

    # -- one configuration ---------------------------------------------------------------------------------
    def run_config(self, job: Job) -> dict:
        """One configuration over the periods asked for: per period the account's daily returns, the round
        trips, the entry orders (time, notional), the exchange stops and the number of fills. Every period starts
        flat: each listed contract is replayed from ``slot_capital``, whatever the period before left."""
        config, wanted = job
        fn = self.ideas[config.idea][0]
        btc, slot, out, weights = lab.frame(BTC, self.snapshot), self.slot_capital, {}, {}
        for k in wanted:
            start, end = self.periods[k]
            symbols, curves, trips, entries, stops, fills = list(self.lists[k]), [], [], [], 0, 0
            for symbol in symbols:
                df = lab.frame(symbol, self.snapshot)
                key = (symbol, tuple(symbols) if config.idea in self.cross_sectional else None)
                if key not in weights:
                    ctx = {"params": config.params, "btc": btc, "symbol": symbol, "universe": symbols}
                    weights[key] = fn(df, ctx).reindex(df.index).fillna(0.0)
                w = weights[key]
                if not (w[(w.index >= start - H) & (w.index < end)] != 0).any():
                    continue
                r = lab.replay(symbol, w, cost_multiplier=config.cost, start=start, end=end, snapshot=self.snapshot,
                               starting_balance=str(slot), **config.stop.replay_terms())
                if not len(r["curve"]):
                    continue
                curves.append(r["curve"])
                trips += lab.round_trips(r["fills"])
                entries += [(at, notional) for at, notional, reduce in r["orders"] if not reduce]
                stops += sum(1 for f in r["fills"] if f["order"] == "protective_stop")
                fills += len(r["fills"])
            daily = period_daily(curves, start, end, slots=self.slots, slot_capital=slot, legacy=self.legacy_days)
            out[k] = {"daily": daily, "trips": sorted(trips), "entries": sorted(entries), "stops": stops,
                      "fills": fills}
        return {"config": config, "periods": out}

    # -- cache and registry --------------------------------------------------------------------------------
    def _settings(self) -> dict:
        """What a cached result depends on besides its configuration."""
        return {"snapshot": self.snapshot.name, "capital": self.terms.capital, "slots": self.slots,
                "periods": [list(p) for p in self.periods], "lists": [list(symbols) for symbols in self.lists],
                "legacy_days": self.legacy_days}

    def _path(self, config: Config) -> Path:
        tag = "_".join(f"{k}{v}" for k, v in sorted(config.params.items()))
        return self.cache / f"{config.idea}__{tag}__{config.stop.name}__x{config.cost}.json"

    def _cached(self, config: Config) -> dict:
        """{period: result} from the cache file (JSON: floats come back exactly), {} without one."""
        path = self._path(config)
        if not path.exists():
            return {}
        stored = json.loads(path.read_text())
        if stored["settings"] != self._settings():
            raise ValueError(f"{path} was made with other settings; use another cache directory")
        return {int(k): {"daily": pd.Series(p["daily"], index=pd.RangeIndex(p["first_day"],
                                                                             p["first_day"] + len(p["daily"])),
                                            dtype=float),
                         "trips": [tuple(t) for t in p["trips"]], "entries": [tuple(e) for e in p["entries"]],
                         "stops": p["stops"], "fills": p["fills"]} for k, p in stored["periods"].items()}

    def _store(self, config: Config, periods_: Mapping[int, dict]) -> None:
        def plain(p: dict) -> dict:
            days = p["daily"].index
            if len(days) and list(days) != list(range(int(days[0]), int(days[0]) + len(days))):
                raise ValueError("a period's days must be consecutive")
            return {"first_day": int(days[0]) if len(days) else 0, "daily": [float(x) for x in p["daily"]],
                    "trips": [[int(at), float(money)] for at, money in p["trips"]],
                    "entries": [[int(at), float(notional)] for at, notional in p["entries"]],
                    "stops": int(p["stops"]), "fills": int(p["fills"])}
        stored = {"settings": self._settings(), "periods": {str(k): plain(p) for k, p in sorted(periods_.items())}}
        self._path(config).write_text(json.dumps(stored))

    def load(self, config: Config) -> dict:
        """The cached periods of a configuration: {period: result}."""
        found = self._cached(config)
        if not found:
            raise KeyError(f"not in the cache: {config.idea} {config.key} x{config.cost}")
        return found

    def _row(self, config: Config, k: int, result: dict) -> dict:
        s = lab.summary(result["daily"])
        row = {"date": self.run_date, "idea": config.idea, "params": config.params, "stop": config.stop.name,
               "cost_multiplier": config.cost, "holdout": k == self.holdout,
               "terms": f"{self.terms.capital} USDT, {self.slots} slots, yearly list, {config.stop.name}",
               "window": [_date(t) for t in self.periods[k]], "total_return": s.get("total_return"),
               "sharpe": s.get("sharpe"), "max_drawdown": s.get("max_drawdown"),
               "round_trips": len(result["trips"]), "fills": result["fills"]}
        if self.note:
            row["note"] = self.note
        if self.legacy_days:
            row["days"] = "as the 2026-11-16 script: each period's first day is not counted"
        return row

    def run_all(self, jobs: Sequence[Job], *, processes: int = 2, holdout: bool = False) -> int:
        """Replay what the cache lacks (a cached configuration with fewer periods is extended, not rerun) and
        append one registry row per configuration and period before the cache keeps it. Returns the number of
        configurations replayed. A job on the holdout period is refused unless ``holdout`` (``final`` only)."""
        global _WORKER
        if not holdout and any(self.holdout in wanted for _, wanted in jobs):
            raise ValueError("the holdout period is read only in the final check of a survivor")
        todo = []
        for config, wanted in jobs:
            missing = tuple(k for k in wanted if k not in self._cached(config))
            if missing:
                todo.append((config, missing))
        if not todo:
            return 0
        self.cache.mkdir(parents=True, exist_ok=True)
        for symbol in sorted({s for _, wanted in todo for k in wanted for s in self.lists[k]} | {BTC}):
            lab.frame(symbol, self.snapshot)                      # loaded once, shared by the forked workers
        _WORKER = self
        pool = get_context("fork").Pool(processes) if processes > 1 else None
        try:
            results = pool.imap_unordered(_work, todo) if pool else map(self.run_config, todo)
            for result in results:
                config = result["config"]
                old = self._cached(config)
                lab.record([self._row(config, k, p) for k, p in result["periods"].items() if k not in old],
                           self.trials)
                self._store(config, {**old, **result["periods"]})
        finally:
            _WORKER = None
            if pool:
                pool.terminate()
        return len(todo)

    # -- the screen and the final check --------------------------------------------------------------------
    def list_hold(self, ks: Iterable[int]) -> pd.Series:
        """Each period's list bought in equal parts at the close of the period's first day and held (no costs)."""
        parts = []
        for k in ks:
            start, end = self.periods[k]
            columns = {}
            for symbol in self.lists[k]:
                close = lab.frame(symbol, self.snapshot)["close"]
                close = close[(close.index >= start) & (close.index < end)]
                if len(close):
                    by_day = close.groupby(close.index // DAY).last()
                    columns[symbol] = by_day / by_day.iloc[0]
            parts.append(pd.DataFrame(columns).ffill().mean(axis=1).pct_change().dropna())
        return pd.concat(parts)

    def _pilot(self, daily: pd.Series, trips: list) -> dict:
        streaks = lab.losing_streaks(trips, self.terms.losing_trades)
        return {"losing_streak_stops": len(streaks), "per_30_days": len(streaks) * 30 / max(1, len(daily)),
                "days_below_daily_stop": int((daily < -self.terms.daily_stop).sum()),
                "loss_stop_share": lab.loss_stop_share(daily, self.terms.loss_stop, self.terms.loss_stop_days)}

    def _screen_row(self, name: str, variants: Sequence[StopVariant], btc: pd.DataFrame, hold: dict) -> dict:
        runs = [(config, self.load(config)) for config in self.candidates(name, variants)]
        wf = walk_forward(runs, list(self.folds))
        result, regime = lab.summary(wf["daily"]), lab.by_regime(wf["daily"], btc)
        return {"chosen_per_fold": [config.label for config in wf["chosen"]], "final_choice": wf["final"].label,
                "walk_forward": result, "regime": regime, "round_trips": len(wf["trips"]),
                "longest_losing_trades": longest_losing(wf["trips"]), "exchange_stops": wf["stops"],
                "score": score(result, wf["trips"], regime), "survives": survives(result, hold),
                "selection_sharpe": {config.key: lab.summary(joined(results, self.selection))["sharpe"]
                                     for config, results in runs},
                "pilot": self._pilot(wf["daily"], wf["trips"]), "limits_needed": lab.limits_needed(wf["entries"])}

    def screen(self, names: Sequence[str] | None = None, variants: Sequence[StopVariant] = (LIVE_STOP,), *,
               processes: int = 2) -> dict:
        """Every parameter set of the ideas with every stop variant on the selection periods (replayed, cached,
        registered), then per idea the walk-forward over the folds, its score and whether it survives: a return
        above 0 and a Sharpe above holding BTC over the folds. The holdout period is not read."""
        names = list(self.ideas) if names is None else list(names)
        self.run_all([(config, tuple(self.selection)) for name in names for config in self.candidates(name, variants)],
                     processes=processes)
        btc = lab.frame(BTC, self.snapshot)
        first, last = self.periods[self.folds[0]][0], self.periods[self.folds[-1]][1]
        hold = lab.summary(lab.buy_and_hold(BTC, first, last, self.snapshot))
        return {"stop_variants": [variant.name for variant in variants], "buy_and_hold_btc": hold,
                "buy_and_hold_list": lab.summary(self.list_hold(self.folds)),
                "ideas": {name: self._screen_row(name, variants, btc, hold) for name in names}}

    def final(self, screened: dict, *, processes: int = 2) -> dict:
        """The final check of the ideas that survived ``screen`` (its result, as returned or read back from JSON).
        Only here is the holdout period replayed, and only for a survivor's final choice; the folds are replayed
        again at their chosen configurations with fees and slippage doubled. The champion is the passing idea
        with the highest score, then the highest walk-forward Sharpe; nobody passes, no champion."""
        names = [name for name, row in screened["ideas"].items() if row["survives"]]
        if not names:
            return {"buy_and_hold": None, "final_check": {}, "ranking": [], "champion": None}

        def config(name: str, label: dict, multiplier: int = 1) -> Config:
            return Config(name, label["params"], StopVariant(**label["stop"]), multiplier)
        jobs: dict[tuple, tuple[Config, list[int]]] = {}
        for name in names:
            row = screened["ideas"][name]
            wanted = [(config(name, row["final_choice"]), self.holdout)]
            wanted += [(config(name, label, 2), k) for k, label in zip(self.folds, row["chosen_per_fold"])]
            for c, k in wanted:
                jobs.setdefault((c.idea, c.key, c.cost), (c, []))[1].append(k)
        self.run_all([(c, tuple(ks)) for c, ks in jobs.values()], processes=processes, holdout=True)
        btc = lab.frame(BTC, self.snapshot)
        hold = {"holdout_btc": lab.summary(lab.buy_and_hold(BTC, *self.periods[self.holdout], self.snapshot)),
                "holdout_list": lab.summary(self.list_hold((self.holdout,)))}
        out = {}
        for name in names:
            row = screened["ideas"][name]
            choice = config(name, row["final_choice"])
            unseen = self.load(choice)[self.holdout]
            doubled = lab.summary(_stitch([self.load(config(name, label, 2))[k]
                                           for k, label in zip(self.folds, row["chosen_per_fold"])])["daily"])
            found = self.lookahead(name, choice.params) if self.lookahead else None
            checks = final_checks(
                holdout=lab.summary(unseen["daily"]), holdout_hold=hold["holdout_btc"], round_trips=row["round_trips"],
                doubled=doubled, walk_forward_hold=screened["buy_and_hold_btc"],
                selection_sharpe=row["selection_sharpe"], lookahead=found)
            out[name] = {"final_choice": row["final_choice"], "holdout": lab.summary(unseen["daily"]),
                         "holdout_regime": lab.by_regime(unseen["daily"], btc),
                         "holdout_round_trips": len(unseen["trips"]), "holdout_exchange_stops": unseen["stops"],
                         "holdout_pilot": self._pilot(unseen["daily"], unseen["trips"]),
                         "holdout_limits_needed": lab.limits_needed(unseen["entries"]),
                         "double_fees_walk_forward": doubled,
                         "lookahead_violations": ["not checked"] if found is None else list(found),
                         "checks": checks, "passes": all(checks.values()), "score": row["score"]["total"]}
        order = sorted(out, key=lambda n: (-int(out[n]["passes"]), -out[n]["score"],
                                           -screened["ideas"][n]["walk_forward"]["sharpe"]))
        return {"buy_and_hold": hold, "final_check": out, "ranking": order,
                "champion": order[0] if out[order[0]]["passes"] else None}


__all__ = ["BTC", "Config", "Judge", "LIVE_STOP", "PERIODS", "STABLE", "STOP_VARIANTS", "StopVariant", "final_checks",
           "joined", "longest_losing", "period_daily", "periods", "score", "survives", "universe", "universe_record",
           "walk_forward"]

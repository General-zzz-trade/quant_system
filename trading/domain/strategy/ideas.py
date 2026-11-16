"""The research lab's ideas as strategies behind the shared interface (plan R7b,
docs/arch/runtime-lab-strategies-plan.md).

The 29 ideas of the lab's first run (research/lab/runs/2026-11-16/ideas.py, reasons in ideas.md) with the same bodies:
each maps one contract's hourly frame (index = bar open time in ms) and a context to a target weight per bar, decided
at that bar's close. Daily ideas decide once a day from completed UTC days. What changed is where the inputs come from:
the frame is built from a ``Window`` (``lab_frame``), BTCUSDT's frame is the window of BTCUSDT (``references``), and a
cross-sectional idea ranks the windows it is given instead of loading the data snapshot (``ctx["frames"]``).

``IdeaStrategy`` is the interface implementation: ``history`` gives every close (the lab's judge replays it) and
``latest`` the last one (the runtime acts on it). A decision reads at most ``window_hours`` closed bars before it, so a
trailing window of that length gives the full-history value; three ideas carry a state from bar to bar (``_state``),
for which ``window_hours`` includes a margin and tests/unit/trading/test_lab_ideas.py measures the agreement.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

H = 3_600_000
DAY = 24 * H
ON = 0.95
BTC = "BTCUSDT"
STATE_MARGIN_DAYS = 120          # extra history for the ideas whose position carries over (``_state``)
Ctx = dict[str, Any]             # params, btc (BTCUSDT's frame), symbol, universe, frames


def lab_frame(window: Any) -> pd.DataFrame:
    """The lab's hourly frame from a window: indexed by open time; the bar and kline columns; ``funding``, the rate
    settled at that hour (summed; 0 when none), as lab.um_lab.frame builds it from a snapshot."""
    hourly = window.hourly
    frame = hourly.drop(columns=[c for c in ("open_time", "available_at") if c in hourly]).set_index(
        pd.Index(hourly["open_time"].to_numpy(np.int64), name="open_time"))
    funding = window.funding
    hours = (np.asarray(funding.index, dtype=np.int64) // H) * H
    rates = pd.Series(np.asarray(funding.to_numpy(), dtype=float), index=hours).groupby(level=0).sum()
    frame["funding"] = rates.reindex(frame.index).fillna(0.0).to_numpy()
    return frame


# -- helpers (as in the lab) ---------------------------------------------------------------------------------------
def _daily(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index // DAY
    out = pd.DataFrame({"close": df["close"].groupby(day).last(), "quote": df["quote_volume"].groupby(day).sum(),
                        "taker": df["taker_buy_quote_volume"].groupby(day).sum(),
                        "funding": df["funding"].groupby(day).sum()})
    out["ret"] = out["close"].pct_change()
    return out


def _hours(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """A per-day signal (made from that completed day) onto the bars decided after the day is complete."""
    done = (df.index + H) // DAY - 1
    return pd.Series(signal.reindex(done).to_numpy(dtype=float), index=df.index).fillna(0.0)


def _btc(ctx: Ctx, d: pd.DataFrame) -> pd.DataFrame:
    return _daily(ctx["btc"]).reindex(d.index)


def _state(enter_long: Any, exit_long: Any, enter_short: Any = None, exit_short: Any = None) -> np.ndarray:
    """+1 / -1 / 0 with hysteresis over boolean arrays (NaN comparisons are False)."""
    n, pos, out = len(enter_long), 0, np.zeros(len(enter_long))
    for i in range(n):
        if pos == 1 and exit_long[i]:
            pos = 0
        elif pos == -1 and exit_short is not None and exit_short[i]:
            pos = 0
        if pos == 0:
            if enter_long[i]:
                pos = 1
            elif enter_short is not None and enter_short[i]:
                pos = -1
        out[i] = pos
    return out


def _panel(ctx: Ctx, column: str) -> pd.DataFrame:
    """The daily ``column`` of every symbol the strategy was given (the lab read the period's list from the
    snapshot)."""
    return pd.DataFrame({s: _daily(ctx["frames"][s])[column] for s in ctx["universe"]})


def _rank_pick(score: pd.DataFrame, symbol: str, k: int, *, top: bool) -> pd.Series:
    """True on the days ``symbol`` is among the ``k`` highest (or lowest) scores of the universe."""
    rank = score.rank(axis=1, ascending=not top, method="first")
    return (rank[symbol] <= k) & score[symbol].notna()


# -- A: trend and momentum -----------------------------------------------------------------------------------------
def _tsmom_long(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    return _hours(df, (d["close"] / d["close"].shift(ctx["params"]["n"]) - 1 > 0) * ON)


def _tsmom_ls(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    return _hours(df, np.sign(d["close"] / d["close"].shift(ctx["params"]["n"]) - 1) * ON)


def _ma_cross_long(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, p = _daily(df), ctx["params"]
    fast, slow = d["close"].rolling(p["fast"]).mean(), d["close"].rolling(p["slow"]).mean()
    return _hours(df, (fast > slow) * ON)


def _donchian_long(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, n = _daily(df), ctx["params"]["n"]
    high, low = d["close"].rolling(n).max().shift(1), d["close"].rolling(n // 2).min().shift(1)
    state = _state((d["close"] > high).to_numpy(), (d["close"] < low).to_numpy())
    return _hours(df, pd.Series(state * ON, index=d.index))


def _btc_gate_tsmom(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, p = _daily(df), ctx["params"]
    b = _btc(ctx, d)["close"]
    gate = b > _daily(ctx["btc"])["close"].rolling(p["m"]).mean().reindex(d.index)
    return _hours(df, ((d["close"] / d["close"].shift(p["n"]) - 1 > 0) & gate) * ON)


def _voladj_mom(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, p = _daily(df), ctx["params"]
    n = p["n"]
    score = (d["close"] / d["close"].shift(n) - 1) / (d["ret"].rolling(n).std() * np.sqrt(n))
    return _hours(df, (score > p["z"]) * ON)


def _rel_strength_btc(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, n = _daily(df), ctx["params"]["n"]
    b = _btc(ctx, d)["close"]
    own, btc = d["close"] / d["close"].shift(n) - 1, b / b.shift(n) - 1
    return _hours(df, ((own > btc) & (btc > 0)) * ON)


def _xs_momentum_top(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    n = ctx["params"]["n"]
    close = _panel(ctx, "close")
    mom = close / close.shift(n) - 1
    return _hours(df, (_rank_pick(mom, ctx["symbol"], 3, top=True) & (mom[ctx["symbol"]] > 0)) * ON)


# -- B: mean reversion ---------------------------------------------------------------------------------------------
def _xs_reversal_long(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    n = ctx["params"]["days"]
    close = _panel(ctx, "close")
    return _hours(df, _rank_pick(close / close.shift(n) - 1, ctx["symbol"], 3, top=False) * ON)


def _xs_reversal_ls(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    n = ctx["params"]["days"]
    close = _panel(ctx, "close")
    move = close / close.shift(n) - 1
    long, short = _rank_pick(move, ctx["symbol"], 3, top=False), _rank_pick(move, ctx["symbol"], 3, top=True)
    return _hours(df, (long.astype(float) - short.astype(float)) * ON)


def _z20(d: pd.DataFrame) -> pd.Series:
    return (d["close"] - d["close"].rolling(20).mean()) / d["close"].rolling(20).std()


def _dip_in_uptrend(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, z0 = _daily(df), ctx["params"]["z"]
    z, up = _z20(d), d["close"] > d["close"].rolling(100).mean()
    state = _state(((z < -z0) & up).to_numpy(), (z > 0).to_numpy())
    return _hours(df, pd.Series(state * ON, index=d.index))


def _band_reversion_ls(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, z0 = _daily(df), ctx["params"]["z"]
    z = _z20(d)
    state = _state((z < -z0).to_numpy(), (z >= 0).to_numpy(), (z > z0).to_numpy(), (z <= 0).to_numpy())
    return _hours(df, pd.Series(state * ON, index=d.index))


def _flush_reversal(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    p = ctx["params"]
    drop = df["close"] / df["open"] - 1 < -p["drop"]
    busy = df["quote_volume"] > 3 * df["quote_volume"].rolling(24).median().shift(1)
    return (drop & busy).astype(float).rolling(p["hold"], min_periods=1).max() * ON


# -- C: funding ----------------------------------------------------------------------------------------------------
def _funding_carry_short(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    return _hours(df, (d["funding"].rolling(7).sum() > ctx["params"]["thr"]) * -ON)


def _funding_negative_long(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    return _hours(df, (d["funding"].rolling(7).sum() < -ctx["params"]["thr"]) * ON)


def _xs_funding_top_long(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    score = _panel(ctx, "funding").rolling(ctx["params"]["window"]).sum()
    return _hours(df, _rank_pick(score, ctx["symbol"], 3, top=True) * ON)


def _xs_funding_ls(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    score = _panel(ctx, "funding").rolling(ctx["params"]["window"]).sum()
    long, short = _rank_pick(score, ctx["symbol"], 3, top=True), _rank_pick(score, ctx["symbol"], 3, top=False)
    return _hours(df, (long.astype(float) - short.astype(float)) * ON)


def _trend_not_crowded(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    calm = d["funding"].rolling(7).sum() < ctx["params"]["thr"]
    return _hours(df, ((d["close"] / d["close"].shift(40) - 1 > 0) & calm) * ON)


# -- D: volume -----------------------------------------------------------------------------------------------------
def _surge(d: pd.DataFrame, m: float) -> pd.Series:
    return d["quote"] > m * d["quote"].rolling(30).median().shift(1)


def _volume_breakout(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, p = _daily(df), ctx["params"]
    event = (_surge(d, p["m"]) & (d["ret"] > 0)).astype(float)
    return _hours(df, event.rolling(p["hold"], min_periods=1).max() * ON)


def _taker_imbalance(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    share = d["taker"].rolling(3).sum() / d["quote"].rolling(3).sum()
    return _hours(df, (share > 0.5 + ctx["params"]["x"]) * ON)


def _surge_fade_short(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, p = _daily(df), ctx["params"]
    event = (_surge(d, p["m"]) & (d["ret"] > 0)).astype(float)
    return _hours(df, event.rolling(p["hold"], min_periods=1).max() * -ON)


# -- E: volatility -------------------------------------------------------------------------------------------------
def _squeeze_breakout(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    tight = d["ret"].rolling(10).std() / d["ret"].rolling(120).std() < ctx["params"]["r"]
    return _hours(df, (tight & (d["close"] > d["close"].rolling(20).mean())) * ON)


def _calm_trend(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    vol = d["ret"].rolling(ctx["params"]["w"]).std()
    return _hours(df, ((d["close"] / d["close"].shift(40) - 1 > 0) & (vol < vol.rolling(180).median())) * ON)


def _vol_target_trend(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    size = (ctx["params"]["target"] / d["ret"].rolling(30).std()).clip(upper=ON)
    return _hours(df, (d["close"] / d["close"].shift(40) - 1 > 0) * size)


def _high_vol_off(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    b = _daily(ctx["btc"])
    vol = b["ret"].rolling(7).std()
    storm = (vol > vol.rolling(90).quantile(ctx["params"]["q"])).reindex(d.index)
    known = vol.rolling(90).quantile(ctx["params"]["q"]).reindex(d.index).notna()
    return _hours(df, (known & ~storm.fillna(True).astype(bool)) * ON)


# -- F: the market leader ------------------------------------------------------------------------------------------
def _btc_trend_beta(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d = _daily(df)
    b = _btc(ctx, d)["close"]
    return _hours(df, (b / b.shift(ctx["params"]["n"]) - 1 > 0) * ON)


def _btc_lead_hourly(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    p = ctx["params"]
    close = ctx["btc"]["close"].reindex(df.index)
    return (close / close.shift(4) - 1 > p["x"]).astype(float).rolling(p["hold"], min_periods=1).max() * ON


def _bear_short(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, p = _daily(df), ctx["params"]
    below = _btc(ctx, d)["close"] < _daily(ctx["btc"])["close"].rolling(p["m"]).mean().reindex(d.index)
    return _hours(df, (below & (d["close"] / d["close"].shift(p["n"]) - 1 < 0)) * -ON)


def _regime_switch_ls(df: pd.DataFrame, ctx: Ctx) -> pd.Series:
    d, n = _daily(df), ctx["params"]["n"]
    ma = _daily(ctx["btc"])["close"].rolling(100).mean().reindex(d.index)
    b, own = _btc(ctx, d)["close"], d["close"] / d["close"].shift(n) - 1
    return _hours(df, (((b > ma) & (own > 0)).astype(float) - ((b < ma) & (own < 0)).astype(float)) * ON)


# -- the ideas: function, parameter grid, days of history a decision reads, inputs ---------------------------------
@dataclass(frozen=True)
class Idea:
    fn: Callable[[pd.DataFrame, Ctx], pd.Series]
    grid: tuple[dict[str, Any], ...]
    days: Callable[[Mapping[str, Any]], int]      # completed days before the decision it reads, at most
    btc: bool = False                              # reads BTCUSDT's frame
    cross: bool = False                            # reads every symbol's frame
    state: bool = False                            # carries its position over from bar to bar


def _grid(**axes: list[Any]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = [{}]
    for k in axes:
        out = [dict(o, **{k: v}) for o in out for v in axes[k]]
    return tuple(out)


IDEAS: dict[str, Idea] = {
    "A1_tsmom_long": Idea(_tsmom_long, _grid(n=[20, 40, 80]), lambda p: p["n"] + 1),
    "A2_tsmom_ls": Idea(_tsmom_ls, _grid(n=[20, 40, 80]), lambda p: p["n"] + 1),
    "A3_ma_cross_long": Idea(_ma_cross_long, ({"fast": 10, "slow": 50}, {"fast": 20, "slow": 100},
                                              {"fast": 50, "slow": 200}), lambda p: max(p["fast"], p["slow"])),
    "A4_donchian_long": Idea(_donchian_long, _grid(n=[20, 40, 60]), lambda p: p["n"] + 1, state=True),
    "A5_btc_gate_tsmom": Idea(_btc_gate_tsmom, ({"n": 20, "m": 100}, {"n": 40, "m": 100}, {"n": 40, "m": 200}),
                              lambda p: max(p["n"] + 1, p["m"]), btc=True),
    "A6_voladj_mom": Idea(_voladj_mom, _grid(n=[30], z=[0.5, 1.0, 1.5]), lambda p: p["n"] + 1),
    "A7_rel_strength_btc": Idea(_rel_strength_btc, _grid(n=[14, 30, 60]), lambda p: p["n"] + 1, btc=True),
    "A8_xs_momentum_top3": Idea(_xs_momentum_top, _grid(n=[14, 30, 60]), lambda p: p["n"] + 1, cross=True),
    "B1_xs_reversal_long": Idea(_xs_reversal_long, _grid(days=[1, 2, 3]), lambda p: p["days"] + 1, cross=True),
    "B2_xs_reversal_ls": Idea(_xs_reversal_ls, _grid(days=[1, 2, 3]), lambda p: p["days"] + 1, cross=True),
    "B3_dip_in_uptrend": Idea(_dip_in_uptrend, _grid(z=[1.0, 1.5, 2.0]), lambda p: 100, state=True),
    "B4_band_reversion_ls": Idea(_band_reversion_ls, _grid(z=[1.5, 2.0, 2.5]), lambda p: 20, state=True),
    "B5_flush_reversal": Idea(_flush_reversal, _grid(drop=[0.04, 0.06], hold=[24, 72]),
                              lambda p: (25 + p["hold"]) // 24 + 1),
    "C1_funding_carry_short": Idea(_funding_carry_short, _grid(thr=[0.003, 0.006, 0.010]), lambda p: 7),
    "C2_funding_negative_long": Idea(_funding_negative_long, _grid(thr=[0.001, 0.003, 0.006]), lambda p: 7),
    "C3_xs_funding_top_long": Idea(_xs_funding_top_long, _grid(window=[3, 7, 14]), lambda p: p["window"],
                                   cross=True),
    "C4_xs_funding_ls": Idea(_xs_funding_ls, _grid(window=[3, 7, 14]), lambda p: p["window"], cross=True),
    "C5_trend_not_crowded": Idea(_trend_not_crowded, _grid(thr=[0.003, 0.006, 0.010]), lambda p: 41),
    "D1_volume_breakout": Idea(_volume_breakout, _grid(m=[2, 3], hold=[3, 7]), lambda p: 31 + p["hold"]),
    "D2_taker_imbalance": Idea(_taker_imbalance, _grid(x=[0.01, 0.02, 0.03]), lambda p: 3),
    "D3_surge_fade_short": Idea(_surge_fade_short, _grid(m=[3, 5], hold=[1, 3]), lambda p: 31 + p["hold"]),
    "E1_squeeze_breakout": Idea(_squeeze_breakout, _grid(r=[0.5, 0.7, 0.9]), lambda p: 121),
    "E2_calm_trend": Idea(_calm_trend, _grid(w=[20, 30, 60]), lambda p: p["w"] + 180),
    "E3_vol_target_trend": Idea(_vol_target_trend, _grid(target=[0.02, 0.03, 0.04]), lambda p: 41),
    "E4_high_vol_off": Idea(_high_vol_off, _grid(q=[0.7, 0.8, 0.9]), lambda p: 98, btc=True),
    "F1_btc_trend_beta": Idea(_btc_trend_beta, _grid(n=[10, 20, 40]), lambda p: p["n"] + 1, btc=True),
    "F2_btc_lead_hourly": Idea(_btc_lead_hourly, _grid(x=[0.02, 0.03], hold=[8, 24]),
                               lambda p: (4 + p["hold"]) // 24 + 1, btc=True),
    "F3_bear_short": Idea(_bear_short, ({"m": 100, "n": 20}, {"m": 100, "n": 40}, {"m": 200, "n": 40}),
                          lambda p: max(p["m"], p["n"] + 1), btc=True),
    "F4_regime_switch_ls": Idea(_regime_switch_ls, _grid(n=[20, 40, 80]), lambda p: max(100, p["n"] + 1), btc=True),
}


def idea_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """An idea's parameters from a spec: exactly the names its grid uses, each a finite positive number."""
    idea = IDEAS[name]
    wanted = set(idea.grid[0])
    if not isinstance(params, Mapping) or set(params) != wanted:
        raise ValueError(f"{name} takes exactly the parameters {sorted(wanted)}")
    out = {}
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: {key} must be a positive number")
        out[key] = value
    return out


class IdeaStrategy:
    """One idea with one parameter set, behind the strategy interface."""

    def __init__(self, idea: str, params: Mapping[str, Any]) -> None:
        self.name, self.idea, self.params = idea, IDEAS[idea], idea_params(idea, params)
        self.cross_sectional = self.idea.cross

    def window_hours(self) -> int:
        days = int(self.idea.days(self.params)) + 3 + (STATE_MARGIN_DAYS if self.idea.state else 0)
        return days * 24

    def funding_window(self) -> int:
        return self.window_hours()                 # at most one settlement an hour

    def references(self) -> tuple[str, ...]:
        return (BTC,) if self.idea.btc else ()

    def history(self, windows: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
        frames = {symbol: lab_frame(window) for symbol, window in windows.items()}
        if self.idea.btc and BTC not in frames:
            raise ValueError(f"{self.name} reads {BTC}, which it was not given")
        need = self.window_hours()
        out = {}
        for symbol, frame in frames.items():
            ctx = {"params": self.params, "btc": frames.get(BTC), "symbol": symbol, "universe": tuple(frames),
                   "frames": frames}
            weight = self.idea.fn(frame, ctx).reindex(frame.index).fillna(0.0)
            ready = np.arange(len(frame)) >= need - 1
            out[symbol] = pd.DataFrame({"available_at": frame.index.to_numpy(np.int64) + H,
                                        "target": weight.to_numpy(dtype=float), "ready": ready})
        return out

    def latest(self, windows: Mapping[str, Any]) -> dict[str, tuple[float, bool]]:
        return {symbol: (float(frame["target"].iloc[-1]), bool(frame["ready"].iloc[-1]))
                for symbol, frame in self.history(windows).items()}


__all__ = ["IDEAS", "Idea", "IdeaStrategy", "idea_params", "lab_frame"]

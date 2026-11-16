"""The 29 ideas of the 2026-11-16 run as target-weight functions (see ideas.md for the reasons).

Every function maps one contract's hourly frame (index = bar open time in ms) and a context to a target weight per
bar; the weight on a bar's row is decided at that bar's close and filled at the next open (lab.um_lab.replay).
Daily ideas decide once a day from completed UTC days. ``ctx`` holds ``params``, ``btc`` (BTCUSDT's frame) and, for
the cross-sectional ideas, ``universe`` (the period's symbol list). ON is the weight of a full position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import um_lab as lab

H, DAY = lab.H, lab.DAY
ON = 0.95


# -- helpers ---------------------------------------------------------------------------------------------------
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


def _btc(ctx: dict, d: pd.DataFrame) -> pd.DataFrame:
    return _daily(ctx["btc"]).reindex(d.index)


def _state(enter_long, exit_long, enter_short=None, exit_short=None) -> np.ndarray:
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


_PANELS: dict = {}


def _panel(universe: tuple, column: str) -> pd.DataFrame:
    key = (universe, column)
    if key not in _PANELS:
        _PANELS[key] = pd.DataFrame({s: _daily(lab.frame(s))[column] for s in universe})
    return _PANELS[key]


def _rank_pick(score: pd.DataFrame, symbol: str, k: int, *, top: bool) -> pd.Series:
    """True on the days ``symbol`` is among the ``k`` highest (or lowest) scores of the universe."""
    rank = score.rank(axis=1, ascending=not top, method="first")
    return (rank[symbol] <= k) & score[symbol].notna()


# -- A: trend and momentum -------------------------------------------------------------------------------------
def tsmom_long(df, ctx):
    d = _daily(df)
    return _hours(df, (d["close"] / d["close"].shift(ctx["params"]["n"]) - 1 > 0) * ON)


def tsmom_ls(df, ctx):
    d = _daily(df)
    return _hours(df, np.sign(d["close"] / d["close"].shift(ctx["params"]["n"]) - 1) * ON)


def ma_cross_long(df, ctx):
    d, p = _daily(df), ctx["params"]
    fast, slow = d["close"].rolling(p["fast"]).mean(), d["close"].rolling(p["slow"]).mean()
    return _hours(df, (fast > slow) * ON)


def donchian_long(df, ctx):
    d, n = _daily(df), ctx["params"]["n"]
    high, low = d["close"].rolling(n).max().shift(1), d["close"].rolling(n // 2).min().shift(1)
    state = _state((d["close"] > high).to_numpy(), (d["close"] < low).to_numpy())
    return _hours(df, pd.Series(state * ON, index=d.index))


def btc_gate_tsmom(df, ctx):
    d, p = _daily(df), ctx["params"]
    b = _btc(ctx, d)["close"]
    gate = b > _daily(ctx["btc"])["close"].rolling(p["m"]).mean().reindex(d.index)
    return _hours(df, ((d["close"] / d["close"].shift(p["n"]) - 1 > 0) & gate) * ON)


def voladj_mom(df, ctx):
    d, p = _daily(df), ctx["params"]
    n = p["n"]
    score = (d["close"] / d["close"].shift(n) - 1) / (d["ret"].rolling(n).std() * np.sqrt(n))
    return _hours(df, (score > p["z"]) * ON)


def rel_strength_btc(df, ctx):
    d, n = _daily(df), ctx["params"]["n"]
    b = _btc(ctx, d)["close"]
    own, btc = d["close"] / d["close"].shift(n) - 1, b / b.shift(n) - 1
    return _hours(df, ((own > btc) & (btc > 0)) * ON)


def xs_momentum_top(df, ctx):
    n, uni = ctx["params"]["n"], tuple(ctx["universe"])
    close = _panel(uni, "close")
    mom = close / close.shift(n) - 1
    return _hours(df, (_rank_pick(mom, ctx["symbol"], 3, top=True) & (mom[ctx["symbol"]] > 0)) * ON)


# -- B: mean reversion -----------------------------------------------------------------------------------------
def xs_reversal_long(df, ctx):
    n, uni = ctx["params"]["days"], tuple(ctx["universe"])
    close = _panel(uni, "close")
    return _hours(df, _rank_pick(close / close.shift(n) - 1, ctx["symbol"], 3, top=False) * ON)


def xs_reversal_ls(df, ctx):
    n, uni = ctx["params"]["days"], tuple(ctx["universe"])
    close = _panel(uni, "close")
    move = close / close.shift(n) - 1
    long, short = _rank_pick(move, ctx["symbol"], 3, top=False), _rank_pick(move, ctx["symbol"], 3, top=True)
    return _hours(df, (long.astype(float) - short.astype(float)) * ON)


def _z20(d):
    return (d["close"] - d["close"].rolling(20).mean()) / d["close"].rolling(20).std()


def dip_in_uptrend(df, ctx):
    d, z0 = _daily(df), ctx["params"]["z"]
    z, up = _z20(d), d["close"] > d["close"].rolling(100).mean()
    state = _state(((z < -z0) & up).to_numpy(), (z > 0).to_numpy())
    return _hours(df, pd.Series(state * ON, index=d.index))


def band_reversion_ls(df, ctx):
    d, z0 = _daily(df), ctx["params"]["z"]
    z = _z20(d)
    state = _state((z < -z0).to_numpy(), (z >= 0).to_numpy(), (z > z0).to_numpy(), (z <= 0).to_numpy())
    return _hours(df, pd.Series(state * ON, index=d.index))


def flush_reversal(df, ctx):
    p = ctx["params"]
    drop = df["close"] / df["open"] - 1 < -p["drop"]
    busy = df["quote_volume"] > 3 * df["quote_volume"].rolling(24).median().shift(1)
    return (drop & busy).astype(float).rolling(p["hold"], min_periods=1).max() * ON


# -- C: funding ------------------------------------------------------------------------------------------------
def funding_carry_short(df, ctx):
    d = _daily(df)
    return _hours(df, (d["funding"].rolling(7).sum() > ctx["params"]["thr"]) * -ON)


def funding_negative_long(df, ctx):
    d = _daily(df)
    return _hours(df, (d["funding"].rolling(7).sum() < -ctx["params"]["thr"]) * ON)


def xs_funding_top_long(df, ctx):
    w, uni = ctx["params"]["window"], tuple(ctx["universe"])
    score = _panel(uni, "funding").rolling(w).sum()
    return _hours(df, _rank_pick(score, ctx["symbol"], 3, top=True) * ON)


def xs_funding_ls(df, ctx):
    w, uni = ctx["params"]["window"], tuple(ctx["universe"])
    score = _panel(uni, "funding").rolling(w).sum()
    long, short = _rank_pick(score, ctx["symbol"], 3, top=True), _rank_pick(score, ctx["symbol"], 3, top=False)
    return _hours(df, (long.astype(float) - short.astype(float)) * ON)


def trend_not_crowded(df, ctx):
    d = _daily(df)
    calm = d["funding"].rolling(7).sum() < ctx["params"]["thr"]
    return _hours(df, ((d["close"] / d["close"].shift(40) - 1 > 0) & calm) * ON)


# -- D: volume -------------------------------------------------------------------------------------------------
def _surge(d, m):
    return d["quote"] > m * d["quote"].rolling(30).median().shift(1)


def volume_breakout(df, ctx):
    d, p = _daily(df), ctx["params"]
    event = (_surge(d, p["m"]) & (d["ret"] > 0)).astype(float)
    return _hours(df, event.rolling(p["hold"], min_periods=1).max() * ON)


def taker_imbalance(df, ctx):
    d = _daily(df)
    share = d["taker"].rolling(3).sum() / d["quote"].rolling(3).sum()
    return _hours(df, (share > 0.5 + ctx["params"]["x"]) * ON)


def surge_fade_short(df, ctx):
    d, p = _daily(df), ctx["params"]
    event = (_surge(d, p["m"]) & (d["ret"] > 0)).astype(float)
    return _hours(df, event.rolling(p["hold"], min_periods=1).max() * -ON)


# -- E: volatility ---------------------------------------------------------------------------------------------
def squeeze_breakout(df, ctx):
    d = _daily(df)
    tight = d["ret"].rolling(10).std() / d["ret"].rolling(120).std() < ctx["params"]["r"]
    return _hours(df, (tight & (d["close"] > d["close"].rolling(20).mean())) * ON)


def calm_trend(df, ctx):
    d = _daily(df)
    vol = d["ret"].rolling(ctx["params"]["w"]).std()
    return _hours(df, ((d["close"] / d["close"].shift(40) - 1 > 0) & (vol < vol.rolling(180).median())) * ON)


def vol_target_trend(df, ctx):
    d = _daily(df)
    size = (ctx["params"]["target"] / d["ret"].rolling(30).std()).clip(upper=ON)
    return _hours(df, (d["close"] / d["close"].shift(40) - 1 > 0) * size)


def high_vol_off(df, ctx):
    d = _daily(df)
    b = _daily(ctx["btc"])
    vol = b["ret"].rolling(7).std()
    storm = (vol > vol.rolling(90).quantile(ctx["params"]["q"])).reindex(d.index)
    known = vol.rolling(90).quantile(ctx["params"]["q"]).reindex(d.index).notna()
    return _hours(df, (known & ~storm.fillna(True).astype(bool)) * ON)


# -- F: the market leader --------------------------------------------------------------------------------------
def btc_trend_beta(df, ctx):
    d = _daily(df)
    b = _btc(ctx, d)["close"]
    return _hours(df, (b / b.shift(ctx["params"]["n"]) - 1 > 0) * ON)


def btc_lead_hourly(df, ctx):
    p = ctx["params"]
    close = ctx["btc"]["close"].reindex(df.index)
    return (close / close.shift(4) - 1 > p["x"]).astype(float).rolling(p["hold"], min_periods=1).max() * ON


def bear_short(df, ctx):
    d, p = _daily(df), ctx["params"]
    below = _btc(ctx, d)["close"] < _daily(ctx["btc"])["close"].rolling(p["m"]).mean().reindex(d.index)
    return _hours(df, (below & (d["close"] / d["close"].shift(p["n"]) - 1 < 0)) * -ON)


def regime_switch_ls(df, ctx):
    d, n = _daily(df), ctx["params"]["n"]
    ma = _daily(ctx["btc"])["close"].rolling(100).mean().reindex(d.index)
    b, own = _btc(ctx, d)["close"], d["close"] / d["close"].shift(n) - 1
    return _hours(df, (((b > ma) & (own > 0)).astype(float) - ((b < ma) & (own < 0)).astype(float)) * ON)


def _grid(**axes):
    keys = list(axes)
    out = [{}]
    for k in keys:
        out = [dict(o, **{k: v}) for o in out for v in axes[k]]
    return out


IDEAS = {
    "A1_tsmom_long": (tsmom_long, _grid(n=[20, 40, 80])),
    "A2_tsmom_ls": (tsmom_ls, _grid(n=[20, 40, 80])),
    "A3_ma_cross_long": (ma_cross_long, [{"fast": 10, "slow": 50}, {"fast": 20, "slow": 100}, {"fast": 50, "slow": 200}]),  # noqa: E501
    "A4_donchian_long": (donchian_long, _grid(n=[20, 40, 60])),
    "A5_btc_gate_tsmom": (btc_gate_tsmom, [{"n": 20, "m": 100}, {"n": 40, "m": 100}, {"n": 40, "m": 200}]),
    "A6_voladj_mom": (voladj_mom, _grid(n=[30], z=[0.5, 1.0, 1.5])),
    "A7_rel_strength_btc": (rel_strength_btc, _grid(n=[14, 30, 60])),
    "A8_xs_momentum_top3": (xs_momentum_top, _grid(n=[14, 30, 60])),
    "B1_xs_reversal_long": (xs_reversal_long, _grid(days=[1, 2, 3])),
    "B2_xs_reversal_ls": (xs_reversal_ls, _grid(days=[1, 2, 3])),
    "B3_dip_in_uptrend": (dip_in_uptrend, _grid(z=[1.0, 1.5, 2.0])),
    "B4_band_reversion_ls": (band_reversion_ls, _grid(z=[1.5, 2.0, 2.5])),
    "B5_flush_reversal": (flush_reversal, _grid(drop=[0.04, 0.06], hold=[24, 72])),
    "C1_funding_carry_short": (funding_carry_short, _grid(thr=[0.003, 0.006, 0.010])),
    "C2_funding_negative_long": (funding_negative_long, _grid(thr=[0.001, 0.003, 0.006])),
    "C3_xs_funding_top_long": (xs_funding_top_long, _grid(window=[3, 7, 14])),
    "C4_xs_funding_ls": (xs_funding_ls, _grid(window=[3, 7, 14])),
    "C5_trend_not_crowded": (trend_not_crowded, _grid(thr=[0.003, 0.006, 0.010])),
    "D1_volume_breakout": (volume_breakout, _grid(m=[2, 3], hold=[3, 7])),
    "D2_taker_imbalance": (taker_imbalance, _grid(x=[0.01, 0.02, 0.03])),
    "D3_surge_fade_short": (surge_fade_short, _grid(m=[3, 5], hold=[1, 3])),
    "E1_squeeze_breakout": (squeeze_breakout, _grid(r=[0.5, 0.7, 0.9])),
    "E2_calm_trend": (calm_trend, _grid(w=[20, 30, 60])),
    "E3_vol_target_trend": (vol_target_trend, _grid(target=[0.02, 0.03, 0.04])),
    "E4_high_vol_off": (high_vol_off, _grid(q=[0.7, 0.8, 0.9])),
    "F1_btc_trend_beta": (btc_trend_beta, _grid(n=[10, 20, 40])),
    "F2_btc_lead_hourly": (btc_lead_hourly, _grid(x=[0.02, 0.03], hold=[8, 24])),
    "F3_bear_short": (bear_short, [{"m": 100, "n": 20}, {"m": 100, "n": 40}, {"m": 200, "n": 40}]),
    "F4_regime_switch_ls": (regime_switch_ls, _grid(n=[20, 40, 80])),
}
XS = {"A8_xs_momentum_top3", "B1_xs_reversal_long", "B2_xs_reversal_ls", "C3_xs_funding_top_long", "C4_xs_funding_ls"}

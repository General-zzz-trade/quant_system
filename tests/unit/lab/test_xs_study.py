"""The cross-sectional long/short ideas of the 2026-11-17e study (lab/xs_study.py; owner 2026-11-17: "开广度和成本",
"做多空行情趋势震荡"): each day, long 3 and short 3 of the period's list, decided from completed days only; the regime
idea trades momentum while BTC trends and reversal while it ranges. Synthetic frames only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import um_lab as lab
from lab import xs_study as xs
from trading.domain.strategy.ideas import ON, _xs_reversal_ls

H, DAY = lab.H, lab.DAY
NAMES = ("AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "FUSDT", "GUSDT")


def _frame(close, funding=0.0):
    hours = len(close)
    index = pd.Index(np.arange(hours, dtype=np.int64) * H)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0,
                         "quote_volume": 1e6, "taker_buy_quote_volume": 5e5,
                         "funding": np.where(np.arange(hours) % 8 == 0, funding, 0.0)}, index=index)


def _drift(rate, days, wiggle=0.0, seed=0):
    t = np.arange(days * 24) / 24
    noise = wiggle * np.sin(np.arange(days * 24) * (1.3 + seed))
    return 100 * np.exp(rate * t + noise)


def _frames(days=60, btc=None, **kw):
    out = {s: _frame(_drift(rate, days, **kw)) for s, rate in zip(NAMES, (0.03, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03))}
    out["BTCUSDT"] = _frame(btc if btc is not None else _drift(0.005, days))
    return out


def _weights(fn, frames, params):
    return {s: fn(f, {"params": params, "symbol": s, "universe": list(frames), "frames": frames,
                      "btc": frames["BTCUSDT"]}) for s, f in frames.items()}


def _last(weights):
    return {s: float(w.iloc[-1]) for s, w in weights.items()}


def test_momentum_is_long_the_three_strongest_and_short_the_three_weakest():
    last = _last(_weights(xs.momentum_ls, _frames(), {"n": 7}))
    assert last == {"AUSDT": ON, "BUSDT": ON, "CUSDT": ON, "DUSDT": 0.0, "EUSDT": -ON, "FUSDT": -ON, "GUSDT": -ON,
                    "BTCUSDT": 0.0}


def test_a_decision_needs_n_completed_days_and_holds_for_the_next_day():
    w = _weights(xs.momentum_ls, _frames(), {"n": 7})["AUSDT"]
    first = int(w[w != 0].index[0])
    assert first == 8 * DAY - H                  # day 7's close is the first with 7 days behind it: its last bar
    assert (w[w.index >= first] == ON).all()


def test_carry_is_short_the_highest_funding_and_long_the_lowest():
    frames = _frames()
    for s, rate in zip(NAMES, (0.003, 0.002, 0.001, 0.0, -0.001, -0.002, -0.003)):
        frames[s]["funding"] = np.where(np.arange(len(frames[s])) % 8 == 0, rate, 0.0)
    last = _last(_weights(xs.carry_ls, frames, {"window": 7}))
    assert [last[s] for s in NAMES] == [-ON, -ON, -ON, 0.0, ON, ON, ON]


def test_low_volatility_is_long_the_calmest_and_short_the_wildest():
    def swing(w):                                  # each day's close w above or below 100, in turn
        return 100 * (1 + w * np.where(np.arange(60 * 24) // 24 % 2, 1.0, -1.0))
    frames = {s: _frame(swing(w)) for s, w in zip(NAMES, (0.001, 0.002, 0.003, 0.01, 0.03, 0.05, 0.08))}
    frames["BTCUSDT"] = _frame(swing(0.008))
    last = _last(_weights(xs.lowvol_ls, frames, {"n": 14}))
    assert [last[s] for s in NAMES] == [ON, ON, ON, 0.0, -ON, -ON, -ON]


def test_the_efficiency_ratio_is_one_on_a_straight_line_and_near_zero_in_a_range():
    line = pd.Series(np.arange(40, dtype=float) + 100)
    swing = pd.Series(100 + np.where(np.arange(40) % 2, 1.0, -1.0))
    assert xs.efficiency(line, 30).iloc[-1] == pytest.approx(1.0)
    assert xs.efficiency(swing, 30).iloc[-1] < 0.1
    assert xs.efficiency(line, 30).iloc[:30].isna().all()


@pytest.mark.parametrize("btc_moves, leg", [("trend", "momentum"), ("range", "reversal")])
def test_the_regime_idea_trades_momentum_in_a_trend_and_reversal_in_a_range(btc_moves, leg):
    days = 60
    btc = _drift(0.01, days) if btc_moves == "trend" else 100 * (1 + 0.03 * np.where(np.arange(days * 24) // 24 % 2,
                                                                                     1, -1))
    frames = _frames(days, btc=btc, wiggle=0.02)
    got = _weights(xs.regime_ls, frames, {"thr": 0.3})
    if leg == "momentum":
        want = _weights(xs.momentum_ls, frames, {"n": xs.REGIME_MOMENTUM_DAYS})
    else:
        want = _weights(_xs_reversal_ls, frames, {"days": xs.REGIME_REVERSAL_DAYS})
    late = frames["AUSDT"].index >= 40 * DAY     # once BTC's 30-day efficiency is known
    for s in frames:
        assert np.array_equal(got[s][late].to_numpy(), want[s][late].to_numpy()), s
    assert any((want[s][late] != 0).any() for s in frames)
    early = frames["AUSDT"].index < 31 * DAY - H  # no efficiency yet: no position
    assert all((got[s][early] == 0).all() for s in frames)


def test_the_judge_replays_them_through_the_strategy_interface():
    frames = _frames()
    windows = {s: _Window(f) for s, f in frames.items()}
    history = lab.legacy(xs.momentum_ls, cross=True, reads=("BTCUSDT",))({"n": 7}).history(windows)
    assert float(history["AUSDT"]["target"].iloc[-1]) == ON and float(history["GUSDT"]["target"].iloc[-1]) == -ON


class _Window:
    def __init__(self, frame):
        self.hourly = frame.drop(columns=["funding"]).reset_index(names="open_time")
        self.funding = pd.Series(frame["funding"][frame["funding"] != 0].to_numpy(),
                                 index=frame.index[frame["funding"] != 0].to_numpy(np.int64))

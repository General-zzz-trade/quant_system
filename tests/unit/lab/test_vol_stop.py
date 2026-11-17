"""The lab's volatility-tightened stop (tech lead 2026-11-17; research/lab/runs/2026-11-17/vol_stop.md).

- ``lab.um_lab.vol_stop_pct``: at each hourly close, k x the standard deviation of the last 14 daily (24-hour)
  returns ending at that close, clipped to [1.5 %, cap]. Known at the close (causal). Without 14 whole daily returns
  (history too short, a missing hour among the closes it reads) it is the cap.
- ``lab.simulate(stop_pct_at=...)``: the distance decided at a close goes with that decision's orders; the position
  they open keeps it (the broker's ``set_next_stop_pct``).
- ``StopVariant(cap, pause, reenter, vol_k=k)``: a configuration's stop; its name and replay keywords.
"""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from data.quality.causal_bars import HOUR, Bar
from lab import um_judge as judge
from lab import um_lab as lab
from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec
from tests.unit.lab.test_um_judge_runs import market  # noqa: F401  (a fixture)

H = lab.H


def _frame(n=40 * 24, seed=7, drop=()):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    index = np.arange(n, dtype=np.int64) * H + 1_700_000_000_000 // H * H
    df = pd.DataFrame({"close": close}, index=index)
    return df.drop(index=[index[i] for i in drop])


def _expected(df, t, k, cap, floor=0.015):
    c = df["close"]
    rets = [c[t - 24 * H * j] / c[t - 24 * H * (j + 1)] - 1 for j in range(14)]
    return min(max(k * float(np.std(rets, ddof=1)), floor), cap)


def test_the_distance_is_k_times_the_spread_of_14_daily_returns_clipped():
    df = _frame()
    out = lab.vol_stop_pct(df, "2", "0.08")
    for i in (15 * 24, 20 * 24 + 5, 39 * 24 + 23):
        t = int(df.index[i])
        assert out[t] == pytest.approx(_expected(df, t, 2.0, 0.08), rel=1e-12)
    assert out.index.equals(df.index)
    assert 0.015 <= out.min() and out.max() <= 0.08


def test_floor_and_cap_hold():
    df = _frame()
    assert (lab.vol_stop_pct(df, "0.01", "0.08").iloc[15 * 24:] == 0.015).all()   # tiny k: the floor
    assert (lab.vol_stop_pct(df, "100", "0.045").iloc[15 * 24:] == 0.045).all()   # huge k: the cap


def test_short_history_is_the_cap():
    df = _frame()
    out = lab.vol_stop_pct(df, "2", "0.045")
    assert (out.iloc[:14 * 24] == 0.045).all()                                     # fewer than 15 daily closes
    assert out.iloc[14 * 24] < 0.045                                                # 15 closes: 14 returns


def test_a_missing_hour_makes_the_closes_that_read_it_the_cap():
    m = 20 * 24
    df = _frame(drop=(m,))
    out = lab.vol_stop_pct(df, "2", "0.08")
    reads = int(df.index[0]) + (m + 3 * 24) * H                                     # its sample includes hour m
    misses = reads + H                                                               # never reads hour m or m + 24
    assert out[reads] == 0.08 and out[misses] < 0.08


def test_the_distance_is_known_at_the_close():
    df = _frame()
    t = int(df.index[25 * 24])
    later = df.copy()
    later.loc[later.index > t, "close"] *= 3.0
    a, b = lab.vol_stop_pct(df, "2", "0.08"), lab.vol_stop_pct(later, "2", "0.08")
    assert a.loc[:t].equals(b.loc[:t]) and not a.loc[t + H:].equals(b.loc[t + H:])


@pytest.mark.parametrize("k, cap", [("0", "0.08"), ("-1", "0.08"), ("2", "0"), ("2", "0.01"), ("NaN", "0.08")])
def test_a_bad_rule_is_refused(k, cap):
    with pytest.raises(ValueError):
        lab.vol_stop_pct(_frame(), k, cap)


# -- through lab.simulate -------------------------------------------------------------------------------------
RULES = SymbolRules(D("0.001"), D("0.001"), D("5"))
N = 30


def _sim(stop_pct_at, dip_at=12, dip="96"):
    """Long from the decision at hour 5 (fills at bar 5's open); bar ``dip_at``'s low is 4 % down."""
    bars = [Bar(i * HOUR, D(100), D("100.5"), D(dip) if i == dip_at else D("99.5"), D(100), D(10_000_000))
            for i in range(N)]
    hours = list(range(1, N + 1))
    targets = pd.DataFrame({"available_at": np.array(hours, dtype=np.int64) * HOUR,
                            "target": [0.5 if k >= 5 else 0.0 for k in hours], "ready": [True] * len(hours)})
    return simulate("ALTUSDT", bars, targets, {}, StrategySpec(execution="market"), start_ms=0, rules=RULES,
                    starting_balance="300", stop_pct="0.08", stop_pct_at=stop_pct_at)


def _stopped(result):
    return [int(f["at"]) // HOUR for f in result.fills if f["order"] == "protective_stop"]


def test_the_entry_decision_s_distance_is_the_position_s_stop():
    at = {k * HOUR: "0.08" for k in range(1, N + 1)}
    assert _stopped(_sim(at)) == []                                     # 8 %: a 4 % dip passes
    assert _stopped(_sim({**at, 5 * HOUR: "0.03"})) == [13]              # decided with the entry at hour 5
    assert _stopped(_sim({**at, 4 * HOUR: "0.03"})) == []                # an hour early: not the entry's decision
    assert _stopped(_sim({**at, 6 * HOUR: "0.03"})) == []                # an hour late: the position is open already


def test_no_distances_is_the_fixed_stop():
    assert _stopped(_sim(None)) == [] and _stopped(_sim(None, dip="91")) == [13]


# -- the variant --------------------------------------------------------------------------------------------
def test_a_vol_variant_s_name_and_replay_keywords():
    v = judge.StopVariant("0.045", 4, vol_k="1.5")
    assert v.name == "vol1.5x_cap0.045_pause4h"
    assert v.replay_terms() == {"stop_pct": "0.045", "pause_ms": 4 * H, "reenter_after_stop": "pause",
                                "stop_k": "1.5"}
    w = judge.StopVariant("0.08", None, "signal_reset", vol_k="2")
    assert w.name == "vol2x_cap0.08_signal_reset"
    assert "stop_k" not in judge.LIVE_STOP.replay_terms() and judge.LIVE_STOP.name == "stop0.08_pause4h"


@pytest.mark.parametrize("kwargs", [dict(stop_pct="0", pause_hours=None, vol_k="2"),
                                    dict(stop_pct="0.08", pause_hours=4, vol_k="0"),
                                    dict(stop_pct="0.08", pause_hours=4, vol_k="-1"),
                                    dict(stop_pct="0.08", pause_hours=4, vol_k="NaN")])
def test_a_bad_vol_variant_is_refused(kwargs):
    with pytest.raises(ValueError):
        judge.StopVariant(**kwargs)


def test_a_fixed_stop_s_label_is_unchanged_and_a_vol_stop_s_carries_k():
    assert judge.Config("idea", {"n": 1}).label == {"params": {"n": 1}, "stop": {
        "stop_pct": "0.08", "pause_hours": 4, "reenter": "pause"}}
    assert judge.Config("idea", {}, judge.StopVariant("0.045", 4, vol_k="2")).label["stop"]["vol_k"] == "2"


def test_replay_gives_each_decision_the_distance_known_at_its_close(market, monkeypatch):  # noqa: F811
    """The distance of the bar that opens at t is known at its close, t + 1 hour: the time of that decision."""
    seen = {}
    real = lab.simulate

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(lab, "simulate", spy)
    weights = pd.Series(0.5, index=lab.frame("UPUSDT", market).index)
    lab.replay("UPUSDT", weights, snapshot=market, stop_pct="0.08", stop_k="2", start=lab.START,
               end=lab.START + 60 * lab.DAY)
    df = lab.frame("UPUSDT", market)
    df = df[(df.index >= lab.START - 40 * lab.DAY) & (df.index < lab.START + 60 * lab.DAY)]
    expected = lab.vol_stop_pct(df, "2", "0.08")
    assert seen["stop_pct_at"] == {int(t) + H: f"{v:.6f}" for t, v in expected.items()}
    assert len(set(seen["stop_pct_at"].values())) > 1                       # not a constant

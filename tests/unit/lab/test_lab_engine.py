"""Lab engine: causality, exact trailing-window evaluation, rebalancing rules."""
from __future__ import annotations

import math
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from data.quality.causal_bars import Bar
from lab.data import DAY, HOUR, aggregate, frame_from_bars
from lab.rebalance import plan
from lab.simulate import SYMBOL_RULES, simulate
from lab.strategy import StrategySpec, compute_targets, latest_target

T0 = 1_577_836_800_000  # 2020-01-01 00:00 UTC


def _bars(n, seed=7):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, n)))
    out = []
    for i in range(n):
        o = price[i - 1] if i else price[0]
        c = price[i]
        out.append(Bar(T0 + i * HOUR, D(str(round(o, 4))), D(str(round(max(o, c) * 1.002, 4))),
                       D(str(round(min(o, c) * 0.998, 4))), D(str(round(c, 4))), D("1000")))
    return out


def _funding(n_hours, seed=3):
    rng = np.random.default_rng(seed)
    times = np.arange(T0, T0 + n_hours * HOUR, 8 * HOUR)
    return pd.Series(rng.normal(0.0001, 0.0002, len(times)), index=times)


SPECS = [StrategySpec(), StrategySpec(signal="scaled", long_only=True, lookbacks=(10, 30, 90)),
         StrategySpec(funding_filter=True, vol_spike_cut=True, confirm_4h=True)]


def test_aggregates_are_complete_and_visible_after_close():
    frame = frame_from_bars(_bars(24 * 3 + 5))
    daily = aggregate(frame, 24)
    assert list(daily["available_at"]) == [T0 + DAY, T0 + 2 * DAY, T0 + 3 * DAY]  # partial 4th day dropped
    four = aggregate(frame, 4)
    assert four["available_at"].iloc[0] == T0 + 4 * HOUR and len(four) == (24 * 3 + 5) // 4


@pytest.mark.parametrize("spec", SPECS)
def test_targets_are_causal(spec):
    bars = _bars(24 * 420)
    funding = _funding(len(bars))
    base = compute_targets(frame_from_bars(bars), funding, spec)
    cut = 24 * 400
    changed = bars[:cut] + [Bar(b.open_time, b.open * 3, b.high * 3, b.low * 3, b.close * 3, b.volume)
                            for b in bars[cut:]]
    changed_funding = funding.copy()
    changed_funding[changed_funding.index > bars[cut].open_time] = 0.05  # strictly after bar cut-1 closes
    again = compute_targets(frame_from_bars(changed), changed_funding, spec)
    pd.testing.assert_frame_equal(base.iloc[:cut], again.iloc[:cut])


@pytest.mark.parametrize("spec", SPECS)
def test_trailing_window_equals_full_history(spec):
    bars = _bars(24 * 420)
    funding = _funding(len(bars))
    full = compute_targets(frame_from_bars(bars), funding, spec).set_index("available_at")
    checked = 0
    for end in (24 * 400 + 7, 24 * 410 + 23, len(bars)):
        window = bars[max(0, end - spec.window_hours()):end]
        f = funding[funding.index < bars[end - 1].available_at + 1].iloc[-spec.funding_window():]
        target, ready = latest_target(frame_from_bars(window), f, spec)
        row = full.loc[bars[end - 1].available_at]
        assert ready == bool(row["ready"])
        assert math.isclose(target, row["target"], rel_tol=1e-12, abs_tol=1e-12)
        checked += ready
    assert checked >= 2


def test_rebalancer_rules():
    common = dict(equity=D("1000"), price=D("100"), band=0.1, step=D("0.001"), min_qty=D("0.001"),
                  min_notional=D("5"))
    flip = plan(0.5, current_qty=D("-2"), entry_pending=False, **common)
    assert [(i.side, i.qty, i.reduce_only) for i in flip] == [("buy", D("2"), True), ("buy", D("5.000"), False)]
    assert plan(0.55, current_qty=D("5"), entry_pending=False, **common) == []        # inside the band
    grow = plan(0.8, current_qty=D("5"), entry_pending=False, **common)
    assert [(i.side, i.qty, i.reduce_only) for i in grow] == [("buy", D("3.000"), False)]
    assert plan(0.8, current_qty=D("5"), entry_pending=True, **common) == []          # never stack entries
    shrink = plan(0.2, current_qty=D("5"), entry_pending=False, **common)
    assert [(i.side, i.qty, i.reduce_only) for i in shrink] == [("sell", D("3.000"), True)]
    flat = plan(0.0, current_qty=D("0.5"), entry_pending=False, **common)
    assert [(i.side, i.reduce_only) for i in flat] == [("sell", True)]
    assert plan(0.003, current_qty=D("0"), entry_pending=False, **common) == []       # below min notional


def test_simulation_trades_only_after_start_and_reconciles():
    bars = _bars(24 * 300)
    funding = {int(t): (D(str(round(r, 8))), D("100")) for t, r in _funding(len(bars)).items()}
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, execution="market")
    targets = compute_targets(frame_from_bars(bars), pd.Series({t: float(r) for t, (r, _) in funding.items()}), spec)
    start = bars[24 * 60].open_time
    result = simulate("ETHUSDT", bars, targets, funding, spec, start_ms=start, rules=SYMBOL_RULES["ETHUSDT"])
    assert result.fills and min(f["at"] for f in result.fills) >= start
    last = result.curve.iloc[-1]
    assert last["equity"] > 0 and len(result.curve) == len(bars) - 24 * 60

"""Causality of the strategy function (migration step 7): the target at time T uses only data visible at T.

Replaces the fixed-midnight, x3-scaling check with random cut points over every hour phase, adversarial
replacement of the invisible data, bit-exact comparison, a power check, and a mutation self-check
(an aggregate made visible one hour early must be caught).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.domain import market
from trading.domain.strategy import overlay
from trading.domain.strategy.overlay import StrategySpec, compute_targets

HOUR, DAY = market.HOUR, market.DAY
START = 1_640_995_200_000            # 2022-01-01 00:00 UTC
N_HOURS = 24 * 170
SPECS = {
    "trend_long_only": StrategySpec(lookbacks=(10, 20, 30), vol_lookback=10, long_only=True),
    "all_layers": StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, funding_filter=True, vol_spike_cut=True,
                               confirm_4h=True),
    "scaled_layers": StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, signal="scaled", vol_spike_cut=True,
                                  confirm_4h=True),
}


def _bars(rng: np.random.Generator, n: int, start: int, price: float) -> pd.DataFrame:
    close = price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[price], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n))
    times = start + np.arange(n, dtype=np.int64) * HOUR
    return pd.DataFrame({"open_time": times, "open": open_, "high": np.maximum(open_, close) * (1 + spread),
                         "low": np.minimum(open_, close) * (1 - spread), "close": close,
                         "volume": rng.uniform(100, 1000, n), "available_at": times + HOUR})


def _funding(rng: np.random.Generator, start: int, end: int) -> pd.Series:
    times = np.arange(start, end, 8 * HOUR, dtype=np.int64)
    rates = np.where(rng.random(len(times)) < 0.4, 0.0001, rng.normal(0.0001, 0.0003, len(times)))  # exact ties
    return pd.Series(rates, index=pd.Index(times, dtype=np.int64), dtype=float)


def _world(seed: int):
    rng = np.random.default_rng(seed)
    return _bars(rng, N_HOURS, START, 30_000.0), _funding(rng, START, START + N_HOURS * HOUR)


def _cut_points(seed: int) -> list[int]:
    """One cut per hour of the day, at random days late enough for every layer to be warm."""
    rng = np.random.default_rng(1000 + seed)
    days = rng.integers(140, 168, 24)
    return [START + int(day) * DAY + phase * HOUR for phase, day in enumerate(days)]


def _visible(targets: pd.DataFrame, cut: int) -> pd.DataFrame:
    return targets[targets["available_at"] <= cut].reset_index(drop=True)


def _violations(spec: StrategySpec, bars: pd.DataFrame, funding: pd.Series, seed: int) -> list[str]:
    """Cut points where data invisible at the cut changed a target at or before it."""
    full = compute_targets(bars, funding, spec)
    rng = np.random.default_rng(2000 + seed)
    found = []
    for cut in _cut_points(seed):
        later = bars["open_time"] >= cut                     # bars closing after the cut
        replaced = bars.copy()
        fake = _bars(rng, int(later.sum()), cut, float(rng.uniform(5_000, 90_000)))
        replaced.loc[later, fake.columns] = fake.to_numpy()
        noisy = funding.copy()
        noisy[noisy.index > cut] = rng.normal(0.0, 0.002, int((noisy.index > cut).sum()))
        seen = _visible(compute_targets(replaced, noisy, spec), cut)
        prefix = _visible(compute_targets(bars[bars["available_at"] <= cut].reset_index(drop=True),
                                          funding[funding.index <= cut], spec), cut)
        for name, other in (("future replaced", seen), ("prefix only", prefix)):
            try:
                pd.testing.assert_frame_equal(other, _visible(full, cut), check_exact=True)
            except AssertionError:
                found.append(f"{name} @ {cut} (hour {cut // HOUR % 24})")
    return found


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("seed", [1, 2])
def test_targets_depend_only_on_data_visible_at_the_decision(name, seed):
    bars, funding = _world(seed)
    full = compute_targets(bars, funding, SPECS[name])
    assert full["ready"].iloc[-24 * 30:].all(), "the cut points must fall where every layer is warm"
    assert _violations(SPECS[name], bars, funding, seed) == []


def test_the_bar_closing_at_a_daily_boundary_is_visible_at_that_boundary():
    """Power check: the property test is not vacuous; the newest visible bar does move the target."""
    spec = SPECS["scaled_layers"]
    bars, funding = _world(3)
    moved = 0
    for day in range(140, 168):
        cut = START + day * DAY
        last = bars["open_time"] == cut - HOUR
        changed = bars.copy()
        changed.loc[last, ["close", "high"]] = changed.loc[last, ["close", "high"]] * 1.2
        before = compute_targets(bars, funding, spec)
        after = compute_targets(changed, funding, spec)
        at = before["available_at"] == cut
        moved += int(before.loc[at, "target"].iloc[0] != after.loc[at, "target"].iloc[0])
    assert moved > 0


def test_a_lookahead_bug_is_caught(monkeypatch):
    """Mutation self-check: aggregates visible one hour before they close must be detected."""
    def early(frame, hours):
        out = market.aggregate(frame, hours)
        return out.assign(available_at=out["available_at"] - HOUR)

    monkeypatch.setattr(overlay, "aggregate", early)
    bars, funding = _world(1)
    assert _violations(SPECS["trend_long_only"], bars, funding, 1) != []

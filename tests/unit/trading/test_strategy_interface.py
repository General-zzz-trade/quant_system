"""R1 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16): one strategy interface for
research and the runtime, with no change in behaviour.

A strategy turns each symbol's closed hourly bars and funding settlements (``Window``) into a target weight and a
ready flag at every close. ``history`` gives every close (research replays it); ``latest`` gives the last close (the
runtime acts on it) and equals history's last row. The existing BTC/ETH overlay is the first implementation and
gives exactly what ``compute_targets`` / ``latest_target`` gave. A per-symbol strategy reads only its own window.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from trading.domain.strategy.base import STRATEGIES, PerSymbol, Window, strategy_from
from trading.domain.strategy.overlay import StrategySpec, compute_targets, latest_target

H = 3_600_000
SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market",
                    vol_spike_cut=True, confirm_4h=True)
FUNDED = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market",
                      funding_filter=True)          # needs a year of settlements: 150-day windows below


def _window(seed, hours=24 * 60):
    rng = np.random.default_rng(seed)
    start = 1_700_000_000_000 // H * H
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, hours)))
    open_ = np.r_[close[0], close[:-1]]
    times = start + np.arange(hours) * H
    hourly = pd.DataFrame({"open_time": times, "available_at": times + H, "open": open_,
                           "high": np.maximum(open_, close) * 1.002,
                           "low": np.minimum(open_, close) * 0.998, "close": close, "volume": rng.uniform(1, 9, hours)})
    funding = pd.Series(rng.normal(1e-4, 3e-4, hours // 8), index=start + np.arange(hours // 8) * 8 * H)
    return Window(hourly, funding)


@pytest.mark.parametrize("spec,days", [(SPEC, 60), (FUNDED, 400)])
def test_the_overlay_strategy_gives_exactly_what_the_overlay_functions_gave(spec, days):
    window = _window(1, 24 * days)
    overlay = strategy_from("overlay", spec)
    history = overlay.history({"BTCUSDT": window})["BTCUSDT"]
    expected = compute_targets(window.hourly, window.funding, spec)
    assert expected["ready"].sum() > 24 and (expected["target"] != 0).sum() > 24        # not a comparison of zeros
    pd.testing.assert_frame_equal(history, expected)
    assert overlay.latest({"BTCUSDT": window})["BTCUSDT"] == latest_target(window.hourly, window.funding, spec)
    assert overlay.window_hours() == spec.window_hours() and overlay.funding_window() == spec.funding_window()
    assert overlay.spec is spec and overlay.name == "overlay"


def test_latest_is_the_last_row_of_history_for_every_symbol():
    windows = {"BTCUSDT": _window(1), "ETHUSDT": _window(2)}
    overlay = strategy_from("overlay", SPEC)
    history, latest = overlay.history(windows), overlay.latest(windows)
    assert set(latest) == set(windows)
    for symbol, frame in history.items():
        assert latest[symbol] == (float(frame["target"].iloc[-1]), bool(frame["ready"].iloc[-1]))


def test_a_per_symbol_strategy_reads_only_its_own_window():
    overlay = strategy_from("overlay", SPEC)
    alone = overlay.latest({"BTCUSDT": _window(1)})["BTCUSDT"]
    beside = overlay.latest({"BTCUSDT": _window(1), "ETHUSDT": _window(7)})["BTCUSDT"]
    assert alone == beside


def test_the_registry_names_every_strategy_and_refuses_an_unknown_one():
    assert set(STRATEGIES) == {"overlay"}
    with pytest.raises(ValueError, match="unknown strategy"):
        strategy_from("momentum_xs", SPEC)


def test_a_per_symbol_strategy_is_written_as_one_function_of_a_window():
    class Constant(PerSymbol):
        name = "constant"

        def window_hours(self):
            return 3

        def funding_window(self):
            return 1

        def symbol_history(self, window):
            times = window.hourly["available_at"].to_numpy(dtype=np.int64)
            return pd.DataFrame({"available_at": times, "target": 0.25, "ready": True})
    out = Constant().latest({"A": _window(1, 48), "B": _window(2, 48)})
    assert out == {"A": (0.25, True), "B": (0.25, True)}


def test_the_session_asks_its_strategy_for_the_target(tmp_path):
    from trading.application.session import TargetSession

    class Fixed(PerSymbol):
        name = "fixed"

        def __init__(self):
            self.seen = []

        def window_hours(self):
            return 30

        def funding_window(self):
            return 4

        def symbol_history(self, window):
            self.seen.append((len(window.hourly), len(window.funding)))
            times = window.hourly["available_at"].to_numpy(dtype=np.int64)
            return pd.DataFrame({"available_at": times, "target": -0.4, "ready": True})

    class Parts:
        executor = consumer = stops = gate = flattener = None
        equity = None
    fixed = Fixed()
    session = TargetSession("BTCUSDT", SPEC, journal=None, rules=None, make_order=None, execution=Parts(),
                            allocation=D("0.5"), strategy=fixed)
    assert session.strategy is fixed and session.bars.maxlen == 30 + 48
    window = _window(1, 40)
    session.funding = {int(t): 1e-4 for t in window.hourly["open_time"][::4]}          # 10 settlements
    for row in window.hourly.itertuples():
        session.bars.append(type("Bar", (), {"open_time": row.open_time, "available_at": row.available_at,
                                             "open": row.open, "high": row.high, "low": row.low, "close": row.close,
                                             "volume": row.volume})())
    assert session._target() == (-0.4, True) and fixed.seen == [(40, 4)]   # the strategy's funding window


def test_a_session_without_a_strategy_uses_the_overlay_of_its_spec():
    from trading.application.session import TargetSession

    class Parts:
        executor = consumer = stops = gate = flattener = equity = None
    session = TargetSession("BTCUSDT", SPEC, journal=None, rules=None, make_order=None, execution=Parts())
    assert session.strategy.name == "overlay" and session.strategy.spec is SPEC
    assert isinstance(session.bars, deque) and session.bars.maxlen == SPEC.window_hours() + 48


def test_each_symbol_gets_its_own_targets_when_several_are_asked_together():
    overlay = strategy_from("overlay", SPEC)
    together = overlay.history({"BTCUSDT": _window(1), "ETHUSDT": _window(2)})
    for symbol, seed in (("BTCUSDT", 1), ("ETHUSDT", 2)):
        pd.testing.assert_frame_equal(together[symbol], overlay.history({symbol: _window(seed)})[symbol])
    assert not together["BTCUSDT"]["target"].equals(together["ETHUSDT"]["target"])


def test_a_window_too_short_to_warm_up_is_not_ready():
    assert strategy_from("overlay", SPEC).latest({"BTCUSDT": _window(1, 48)})["BTCUSDT"] == (0.0, False)

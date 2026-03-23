from __future__ import annotations

import numpy as np

from shared.signal_postprocess import rolling_zscore, should_exit_position


def test_rolling_zscore_returns_zero_before_warmup():
    pred = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    z = rolling_zscore(pred, window=10, warmup=5)
    assert np.allclose(z, np.zeros_like(pred))


def test_rolling_zscore_matches_expected_after_warmup():
    pred = np.array([1.0, 2.0, 3.0], dtype=float)
    z = rolling_zscore(pred, window=10, warmup=3)
    arr = np.array([1.0, 2.0, 3.0], dtype=float)
    expected = (3.0 - arr.mean()) / arr.std()
    assert z[0] == 0.0
    assert z[1] == 0.0
    assert z[2] == expected


def test_should_exit_position_honors_max_hold():
    assert should_exit_position(
        position=1.0,
        z_value=2.0,
        held_bars=10,
        min_hold=3,
        max_hold=10,
    )


def test_should_exit_position_blocks_early_exit_before_min_hold():
    assert not should_exit_position(
        position=1.0,
        z_value=-1.0,
        held_bars=2,
        min_hold=3,
        max_hold=10,
    )


def test_should_exit_position_exits_on_reversal_after_min_hold():
    assert should_exit_position(
        position=1.0,
        z_value=-0.5,
        held_bars=4,
        min_hold=3,
        max_hold=10,
    )


def test_should_exit_position_exits_on_deadzone_fade_after_min_hold():
    assert should_exit_position(
        position=1.0,
        z_value=0.1,
        held_bars=4,
        min_hold=3,
        max_hold=10,
    )


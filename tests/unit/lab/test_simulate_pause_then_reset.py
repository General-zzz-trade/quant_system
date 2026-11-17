"""lab.simulate: after a stop, pause, then wait for the target to return to 0 (tech lead 2026-11-17).

``reenter_after_stop="pause_then_reset"`` is the rule a spec's ``"after_stop": "pause_then_reset"`` gives the live
runtime (tests/unit/trading/test_stop_reset_reentry.py), so the backtest models it the same way: after a stop fills,
no entry until BOTH the pause has passed AND a ready decision since the stop wanted exactly 0. A 0 inside the pause
counts. Unlike ``signal_reset`` there is no exception for the other side: a short after a stopped long waits too.

Bars as in test_simulate_signal_reset: a flat price of 100, bar 10's low 15 % down trips the 8 % stop at 11 h.
"""
from __future__ import annotations

import pytest

from lab import um_judge as judge
from tests.unit.lab.test_simulate_signal_reset import HOUR, N, _after_stop, _run, _stops

RULE = dict(reenter_after_stop="pause_then_reset", pause_after_stop_ms=4 * HOUR)


def test_a_target_that_stays_on_never_enters_again():
    result = _run(**RULE)
    assert _stops(result) == [11] and _after_stop(result) == []


def test_after_the_pause_the_first_signal_after_a_zero_enters():
    assert _after_stop(_run({20: 0.0}, **RULE)) == [(21, "buy")]


def test_a_zero_inside_the_pause_counts_but_the_pause_still_holds():
    assert _after_stop(_run({12: 0.0}, **RULE)) == [(15, "buy")]          # pause until 15 h; the reset came at 12 h
    assert _after_stop(_run({12: 0.0}, reenter_after_stop="signal_reset")) == [(13, "buy")]   # no pause: earlier


def test_a_zero_before_the_stop_or_not_ready_does_not_count():
    assert _after_stop(_run({5: 0.0}, **RULE)) == []
    assert _after_stop(_run({20: 0.0}, not_ready=(20,), **RULE)) == []


def test_the_other_side_waits_for_a_zero_too():
    shorts = {k: -0.5 for k in range(20, N + 1)}
    assert _after_stop(_run(shorts, **RULE)) == []                      # stricter than signal_reset
    assert _after_stop(_run({**shorts, 30: 0.0}, **RULE))[0] == (31, "sell")


def test_it_needs_a_pause_and_a_stop():
    with pytest.raises(ValueError, match="pause_after_stop_ms"):
        _run(reenter_after_stop="pause_then_reset")
    with pytest.raises(ValueError, match="stop_pct"):
        _run(reenter_after_stop="pause_then_reset", pause_after_stop_ms=4 * HOUR, stop_pct="0")


def test_the_stop_variant_names_it_and_passes_it_on():
    v = judge.StopVariant("0.08", 4, "pause_then_reset")
    assert v.name == "stop0.08_pause4h_then_reset"
    assert v.replay_terms() == {"stop_pct": "0.08", "pause_ms": 4 * HOUR, "reenter_after_stop": "pause_then_reset"}
    with pytest.raises(ValueError):
        judge.StopVariant("0.08", None, "pause_then_reset")
    with pytest.raises(ValueError):
        judge.StopVariant("0", None, "pause_then_reset")

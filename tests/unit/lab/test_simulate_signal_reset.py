"""lab.simulate: after a protective stop, wait for the signal to reset (research/lab/runs/2026-11-17/plan.md).

``reenter_after_stop="signal_reset"``: after a stop fills, the sleeve does not enter until the idea's target has
been 0 on at least one decision since the stop; the next non-zero target may then enter. The default, ``"pause"``,
is the fixed-time pause of test_simulate_live_terms and is unchanged.

Bars as in test_simulate_live_terms: a flat price of 100, a bar whose low is 15 % down trips the 8 % stop. Bar k
opens at k hours; the decision at its close (k + 1 hours) fills at the next bar's open, the same time.
"""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from data.quality.causal_bars import HOUR, Bar
from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec

RULES = SymbolRules(D("0.001"), D("0.001"), D("5"))
SPEC = StrategySpec(execution="market")
N = 60


def _bars(crashes=(10,)):
    return [Bar(i * HOUR, D(100), D("100.5"), D(85) if i in crashes else D("99.5"), D(100), D(10_000_000))
            for i in range(N)]


def _run(target_at=None, *, crashes=(10,), not_ready=(), missing=(), **kwargs):
    """``target_at``: {decision hour: target} over a default of 0.5 (long the whole time)."""
    bars = _bars(crashes)
    hours = [k for k in range(1, N + 1) if k not in missing]
    target = [(target_at or {}).get(k, 0.5) for k in hours]
    targets = pd.DataFrame({"available_at": np.array(hours, dtype=np.int64) * HOUR, "target": target,
                            "ready": [k not in not_ready for k in hours]})
    kwargs.setdefault("stop_pct", "0.08")
    return simulate("ALTUSDT", bars, targets, {}, SPEC, start_ms=0, rules=RULES, starting_balance="300", **kwargs)


def _stops(result):
    return [int(f["at"]) // HOUR for f in result.fills if f["order"] == "protective_stop"]


def _after_stop(result, stop_hour=11):
    """(hour, side) of every fill after the stop that is not a stop."""
    return [(int(f["at"]) // HOUR, f["side"]) for f in result.fills
            if f["order"] != "protective_stop" and int(f["at"]) >= stop_hour * HOUR]


def test_a_target_that_stays_on_after_the_stop_never_enters_again():
    result = _run(reenter_after_stop="signal_reset")
    assert _stops(result) == [11]                                        # bar 10's low is 15 % down
    assert _after_stop(result) == []
    assert result.curve["qty"].iloc[-1] == 0


def test_the_sleeve_enters_at_the_first_signal_after_the_target_was_zero():
    result = _run({20: 0.0}, reenter_after_stop="signal_reset")          # one decision at 0, then long again
    assert _stops(result) == [11]
    assert _after_stop(result) == [(21, "buy")]                          # decided at 21 h, filled at that open
    assert result.curve["qty"].iloc[-1] > 0


def test_a_zero_target_at_the_close_of_the_stop_bar_counts():
    result = _run({11: 0.0}, reenter_after_stop="signal_reset")          # the stop filled inside bar 10
    assert _after_stop(result) == [(12, "buy")]


def test_a_zero_target_before_the_stop_does_not_count():
    result = _run({5: 0.0}, reenter_after_stop="signal_reset")           # flat at 5 h, long again from 6 h
    assert _stops(result) == [11]
    assert _after_stop(result) == []


def test_a_decision_that_is_not_ready_or_missing_is_not_a_reset():
    assert _after_stop(_run({20: 0.0}, not_ready=(20,), reenter_after_stop="signal_reset")) == []
    assert _after_stop(_run(missing=(20,), reenter_after_stop="signal_reset")) == []


def test_a_target_that_changes_sign_without_passing_zero_is_not_a_reset():
    short_later = {k: -0.5 for k in range(20, N + 1)}
    assert _after_stop(_run(short_later, reenter_after_stop="signal_reset")) == []
    assert _after_stop(_run(short_later))[0] == (11, "buy")              # the default enters again at once
    assert (20, "sell") in _after_stop(_run(short_later))                # and follows the target short


def test_every_stop_waits_for_its_own_reset():
    result = _run({20: 0.0, 40: 0.0}, crashes=(10, 30), reenter_after_stop="signal_reset")
    assert _stops(result) == [11, 31]
    assert _after_stop(result) == [(21, "buy"), (41, "buy")]             # nothing between 31 h and the second zero


def test_the_default_is_the_fixed_pause_bit_for_bit():
    for kwargs in ({}, {"pause_after_stop_ms": 24 * HOUR}, {"stop_pct": "0"}):
        plain, explicit = _run(**kwargs), _run(reenter_after_stop="pause", **kwargs)
        assert plain.fills == explicit.fills and plain.order_rows == explicit.order_rows
        assert plain.curve.equals(explicit.curve)
    assert _after_stop(_run(pause_after_stop_ms=24 * HOUR))[0] == (35, "buy")     # 11 h + 24 h, as before


def test_combinations_that_make_no_sense_are_refused():
    with pytest.raises(ValueError, match="pause_after_stop_ms"):         # two rules for the same stop
        _run(reenter_after_stop="signal_reset", pause_after_stop_ms=4 * HOUR)
    with pytest.raises(ValueError, match="stop_pct"):                    # nothing to wait for
        _run(reenter_after_stop="signal_reset", stop_pct="0")
    with pytest.raises(ValueError, match="reenter_after_stop"):
        _run(reenter_after_stop="reset")

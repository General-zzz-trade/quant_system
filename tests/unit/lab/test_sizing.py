"""Sizing rule: 1-3x when a position is taken, lower only by the per-trade loss cap, refused near liquidation."""
from __future__ import annotations

import math

import pytest

from lab.sizing import SizingRule, liquidation_distance

RULE = SizingRule(target_vol=0.6, max_trade_loss=0.06)


def test_liquidation_distance_of_10x_isolated():
    assert math.isclose(liquidation_distance(10, 0.005), 0.095)
    assert math.isclose(RULE.max_stop_distance, 0.0475)


def test_no_signal_or_unknown_inputs_size_to_zero():
    assert RULE.exposure(0, 0.5, 0.02)[0] == 0
    assert RULE.exposure(1, float("nan"), 0.02)[0] == 0
    assert RULE.exposure(-1, 0.5, 0.0)[0] == 0


def test_volatility_target_is_floored_at_one_and_capped_at_three():
    assert math.isclose(RULE.exposure(1, 0.4, 0.01)[0], 1.5)    # 0.6 / 0.4
    assert RULE.exposure(-1, 0.1, 0.01)[0] == 3.0                # 6.0 -> capped at 3
    assert RULE.exposure(1, 1.2, 0.01)[0] == 1.0                 # 0.5 -> floored at 1


def test_loss_cap_may_go_below_one():
    exposure, reason = RULE.exposure(1, 1.2, 0.04)               # base 1.0, cap 0.06 / 0.04 = 1.5
    assert exposure == 1.0
    exposure, reason = SizingRule(target_vol=0.6, max_trade_loss=0.03).exposure(1, 1.2, 0.04)
    assert math.isclose(exposure, 0.75) and "max trade loss" in reason


def test_stop_too_close_to_liquidation_is_refused():
    exposure, reason = RULE.exposure(1, 0.4, 0.05)
    assert exposure == 0 and "liquidation distance" in reason


@pytest.mark.parametrize("kwargs", [{"max_exposure": 3.5}, {"isolated_leverage": 2.0}, {"min_open": 4.0},
                                    {"max_trade_loss": 1.0}, {"target_vol": -0.1}, {"maintenance_margin_rate": 0.2}])
def test_invalid_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SizingRule(**{"target_vol": 0.6, "max_trade_loss": 0.06, **kwargs})

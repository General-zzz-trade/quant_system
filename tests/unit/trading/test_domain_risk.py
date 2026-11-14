"""Pure risk rules in trading.domain.risk (no storage, no clock) and the strict execution gate factory."""
from __future__ import annotations

import math
from decimal import Decimal as D

import pytest

from trading.application.risk.entry_gate import CompositeRiskGate
from trading.domain.risk.drawdown import advance, periods, validate_limits
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE, exposure_breach, validated_leverage


@pytest.mark.parametrize("gross,proposed,allowed", [
    (D("0"), D("30000"), True),          # exactly 3 x equity is allowed
    (D("0"), D("30000.01"), False),
    (D("29999"), D("1"), True),
    (D("29999"), D("1.01"), False),
])
def test_exposure_cap_boundary(gross, proposed, allowed):
    assert (exposure_breach(D("10000"), gross, proposed, MAX_TOTAL_LEVERAGE) == "") is allowed


@pytest.mark.parametrize("equity,proposed", [(D(0), D(1)), (D(-1), D(1)), (D("NaN"), D(1)), (D(10), D(0))])
def test_exposure_refuses_unknown_equity_or_empty_orders(equity, proposed):
    assert exposure_breach(equity, D(0), proposed, MAX_TOTAL_LEVERAGE) != ""


@pytest.mark.parametrize("value", ["0", "-1", "3.0001", "NaN", "Infinity"])
def test_configured_leverage_can_only_lower_the_cap(value):
    with pytest.raises(ValueError):
        validated_leverage(value)
    assert validated_leverage("1.5") == D("1.5")


def test_drawdown_periods_follow_utc_day_and_iso_week():
    assert periods(1735689599.0) == {"day": "2024-12-31", "week": "2025-W01"}   # ISO week 1 of 2025
    assert periods(1735689600.0) == {"day": "2025-01-01", "week": "2025-W01"}


def test_drawdown_advance_tracks_the_peak_and_keeps_the_tighter_limit():
    assert advance(1000.0, 3.0, 20.0, 1100.0) == (1100.0, 3.0, False)
    assert advance(1100.0, 3.0, 20.0, 1067.0) == (1100.0, 3.0, True)    # exactly 3 % below the peak
    assert advance(1000.0, 5.0, 2.0, 985.0) == (1000.0, 2.0, False)     # a tighter config applies at once


@pytest.mark.parametrize("baseline,stored", [(0.0, 3.0), (math.nan, 3.0), (1000.0, 0.0), (1000.0, 100.0)])
def test_drawdown_advance_rejects_corrupt_stored_state(baseline, stored):
    with pytest.raises(ValueError):
        advance(baseline, stored, 3.0, 1000.0)


def test_drawdown_limits_are_bounded_percentages():
    with pytest.raises(ValueError):
        validate_limits(0.0, 10.0)
    with pytest.raises(ValueError):
        validate_limits(3.0, 100.0)


def test_execution_gate_requires_every_component():
    parts = dict(kill_switch=object(), drawdown_guard=object(), portfolio_reader=object(), venue="binance")
    assert isinstance(CompositeRiskGate.for_execution(object(), **parts), CompositeRiskGate)
    for name in ("kill_switch", "drawdown_guard", "portfolio_reader"):
        with pytest.raises(ValueError, match=name):
            CompositeRiskGate.for_execution(object(), **{**parts, name: None})

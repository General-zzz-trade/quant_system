"""The shared look-ahead check (lab/causality.py; design 2026-11 M5, step S12) passes what is known in time and
catches what is not.

Data: the synthetic snapshot of test_pair_study (280 days of both symbols with funding, premium and metrics),
loaded by lab.pair.load_inputs exactly as a study loads the real one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import candidates as c
from lab import pair
from lab.causality import violations
from tests.unit.lab.test_pair_study import write_snapshot

HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_609_459_200_000          # 2021-01-01
DAYS = 280
END = T0 + DAYS * DAY
CUTS = (T0 + 230 * DAY, T0 + 245 * DAY + 13 * HOUR, T0 + 262 * DAY + 4 * HOUR)    # midnight, an odd hour, a 4h close
BTC, ETH = "BTCUSDT", "ETHUSDT"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    market_dir, premium_dir = write_snapshot(tmp_path_factory.mktemp("lookahead"), DAYS)
    return pair.load_inputs(market_dir, premium_dir, END)


# -- the check itself: it passes what is known in time and catches what is not ----------------------------------------
def _next_close(m):
    return m["hourly"][BTC].assign(next_close=lambda f: f["close"].shift(-1))           # the next bar's close


def _whole_sample_zscore(m):
    close = c.daily_close(m["hourly"][BTC])
    return (close - close.mean()) / close.std()                                          # mean of the whole sample


def _open_interest_without_the_lag(m):
    rows = m["metrics"][BTC]
    oi = pd.Series(rows["sum_open_interest"].to_numpy(float), index=rows["timestamp"].to_numpy(dtype=np.int64))
    times = m["hourly"][BTC]["available_at"].to_numpy(dtype=np.int64)
    return pd.Series(pair._last_value(oi, times, HOUR), index=times)                    # a row stamped t, used at t


def _ready_unless_last_day(m):
    hourly = m["hourly"][BTC]
    return hourly.assign(ready=np.arange(len(hourly)) < len(hourly) - 24)                # knows where the data ends


@pytest.mark.parametrize("case,where,per_cut", [
    (_next_close, "output['next_close']", 2),
    (_whole_sample_zscore, "output", 2),
    (_open_interest_without_the_lag, "output", 2),
    (_ready_unless_last_day, "output['ready']", 1),                 # only removing data shows it
])
def test_the_check_catches_an_injected_lookahead(inputs, case, where, per_cut):
    found = violations(case, inputs, CUTS)
    assert len(found) == per_cut * len(CUTS) and all(line.endswith(f"changes {where}") for line in found)


def test_values_known_exactly_at_the_cut_are_not_changed(inputs):
    assert violations(lambda m: m["hourly"][BTC], inputs, CUTS) == []
    assert violations(lambda m: m["funding"][ETH], inputs, CUTS) == []
    assert violations(lambda m: m["metrics"][ETH], inputs, CUTS) == []


def test_the_check_refuses_to_be_vacuous(inputs):
    with pytest.raises(ValueError, match="never changes the output"):
        violations(lambda m: pd.Series([1.0], index=[T0]), inputs, CUTS)
    with pytest.raises(ValueError, match="no input is known after"):
        violations(lambda m: m["hourly"][BTC], inputs, [END + DAY])
    with pytest.raises(ValueError, match="no output is known at"):
        violations(lambda m: m["hourly"][BTC], inputs, [T0 - DAY])
    with pytest.raises(TypeError, match="cannot tell when"):
        violations(lambda m: m["hourly"][BTC]["close"].to_numpy(), inputs, CUTS)

"""Maker entries with the live stop in research (the cost lever, owner 2026-11-17: "开广度和成本").

A maker entry is a post-only limit at the decision close x (1 -/+ LIMIT_OFFSET_BPS); it fills at its limit, paying
the maker fee, only in a bar that trades through the limit. OHLC cannot say whether the bar's extreme came before
or after that fill, so the stop also checks the bar of a maker fill, as if the extreme came after it: the adverse
order. (A continuous path from the open to a low beyond the stop crosses the limit first, so this is also the
likely order.) Paper's simulated exchange refuses a broker with ``stop_pct`` and places its own stops; this is
research only. Synthetic bars only.
"""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from data.quality.causal_bars import HOUR, Bar
from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec
from trading.adapters.simulated.bar_broker import ReplayConfig
from trading.domain.orders import LIMIT_OFFSET_BPS

RULES = SymbolRules(D("0.001"), D("0.001"), D("5"))
MAKER = StrategySpec(execution="maker", ttl_bars=1)
LIMIT = D(100) * (1 - LIMIT_OFFSET_BPS / 10000)                 # a buy decided at a close of 100


def _bars(crash_at, n=40):
    bars = []
    for i in range(n):
        low = D(85) if i == crash_at else D("99.5")
        bars.append(Bar(i * HOUR, D(100), D("100.5"), low, D(100), D(10_000_000)))
    return bars


def _run(crash_at, **kwargs):
    bars = _bars(crash_at)
    targets = pd.DataFrame({"available_at": np.array([b.available_at for b in bars], dtype=np.int64),
                            "target": 0.5, "ready": True})
    return simulate("ALTUSDT", bars, targets, {}, MAKER, start_ms=0, rules=RULES, starting_balance="300",
                    stop_pct="0.08", **kwargs)


def test_a_maker_entry_with_a_stop_fills_at_its_limit_and_pays_the_maker_fee():
    result = _run(crash_at=30)
    entry = result.fills[0]
    assert entry["order"] != "protective_stop" and entry["side"] == "buy"
    assert int(entry["at"]) == 2 * HOUR                         # decided at bar 0's close, filled in bar 1
    assert D(entry["price"]) == LIMIT
    assert D(entry["fee"]) == D(entry["qty"]) * LIMIT * D(2) / 10000


def test_a_stop_reached_in_the_bar_of_a_maker_fill_fires_in_that_bar():
    result = _run(crash_at=1)                                   # the entry's own bar falls 15 %
    entry, stop = result.fills[0], result.fills[1]
    assert D(entry["price"]) == LIMIT and stop["order"] == "protective_stop"
    assert int(entry["at"]) == int(stop["at"]) == 2 * HOUR      # after the fill, in the same bar
    assert D(stop["price"]) == LIMIT * (1 - D("0.08")) * (1 - D(10) / 10000)   # at the stop, less stop slippage
    assert D(stop["qty"]) == D(entry["qty"])


def test_a_maker_position_is_stopped_later_like_a_market_one():
    result = _run(crash_at=20)
    stops = [f for f in result.fills if f["order"] == "protective_stop"]
    assert [int(f["at"]) for f in stops] == [21 * HOUR]
    assert D(stops[0]["price"]) == LIMIT * (1 - D("0.08")) * (1 - D(10) / 10000)


def test_maker_entries_with_isolated_margin_and_a_stop_are_modelled():
    result = _run(crash_at=1, leverage="2", exchange_leverage=10)
    entry, stop = result.fills[0], result.fills[1]
    assert D(entry["price"]) == LIMIT and stop["order"] == "protective_stop"
    assert int(stop["at"]) == 2 * HOUR


def test_the_broker_takes_a_stop_with_maker_entries():
    assert ReplayConfig(entry_mode="maker", stop_pct="0.08").stop_pct == "0.08"
    with pytest.raises(ValueError):
        ReplayConfig(entry_mode="limit")

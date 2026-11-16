"""lab.simulate with the live runtime's stop and entry pause (tech lead 2026-11-16: the backtest follows live).

(A target of 0.5: at exactly 1.0 the broker's unlevered cap, which counts the bar's high, refuses an entry whose
size rounding leaves no room; real lot steps leave room.)

Live, a triggered exchange stop pauses the symbol's entries; the tech lead chose that the pause ends by itself
after a fixed time, in live and in the backtest alike. Here the sleeve is stopped out and enters again only when
the pause is over, although the strategy's target stayed long the whole time. Without these options the simulator
is unchanged.
"""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd

from data.quality.causal_bars import HOUR, Bar
from lab.simulate import SymbolRules, simulate
from lab.strategy import StrategySpec

RULES = SymbolRules(D("0.001"), D("0.001"), D("5"))
SPEC = StrategySpec(execution="market")


def _bars(prices, crash_at):
    bars = []
    for i, p in enumerate(prices):
        low = p * 0.85 if i == crash_at else p * 0.995
        bars.append(Bar(i * HOUR, D(str(p)), D(str(p * 1.005)), D(str(low)), D(str(p)), D(10_000_000)))
    return bars


def _run(**kwargs):
    prices = [100.0] * 60
    bars = _bars(prices, crash_at=10)
    targets = pd.DataFrame({"available_at": np.array([b.available_at for b in bars], dtype=np.int64),
                            "target": 0.5, "ready": True})
    return simulate("ALTUSDT", bars, targets, {}, SPEC, start_ms=0, rules=RULES, starting_balance="300", **kwargs)


def _entries(result):
    return sorted(int(f["at"]) for f in result.fills if f["order"] != "protective_stop" and f["side"] == "buy")


def test_a_stop_pauses_entries_for_the_chosen_time_then_the_sleeve_enters_again():
    result = _run(stop_pct="0.08", pause_after_stop_ms=24 * HOUR)
    stops = [f for f in result.fills if f["order"] == "protective_stop"]
    assert [int(f["at"]) for f in stops] == [11 * HOUR]                  # bar 10's low is 15 % down
    entries = _entries(result)
    assert entries[0] == HOUR                                            # first decision, filled at bar 1's open
    again = [t for t in entries if t >= 11 * HOUR]
    assert again and again[0] == 11 * HOUR + 24 * HOUR                   # decided as the pause ends, filled then
    assert result.curve["qty"].iloc[-1] > 0


def test_without_a_pause_the_sleeve_enters_again_at_the_next_open():
    result = _run(stop_pct="0.08")
    assert [t for t in _entries(result) if t >= 11 * HOUR][0] == 11 * HOUR   # decided at the stop bar's close


def test_without_a_stop_nothing_changes():
    plain, explicit = _run(), _run(stop_pct="0", pause_after_stop_ms=None)
    assert plain.fills == explicit.fills
    assert not [f for f in plain.fills if f["order"] == "protective_stop"]
    assert _entries(plain) == [HOUR]

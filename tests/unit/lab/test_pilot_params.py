"""The pilot's risk and execution settings study (lab/pilot_params.py; owner 2026-11-17: "来开始探索交易的参数数值设置
多少最佳", scope A): the grid, the plateau choice, the year-by-year walk-forward and the two-symbol run with the
runtime's rules (pause after a stop, pause then wait for a target of 0, drawdown latches). Synthetic data only."""
from __future__ import annotations

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from data.quality.causal_bars import HOUR, Bar
from lab import pilot_params as pp
from lab import um_lab
from lab.strategy import StrategySpec
from trading.adapters.simulated.bar_broker import SymbolRules

DAY = 24 * HOUR


def test_the_grid_has_sixty_settings_each_at_the_exchange_leverage_its_stop_allows():
    assert len(pp.GRID) == 60 and len({s.key for s in pp.GRID}) == 60
    for s in pp.GRID:
        assert s.margin == min(10, um_lab.exchange_leverage(s.stop)) and s.multiple <= s.margin
    assert pp.BASELINE in pp.GRID


def test_neighbours_differ_by_one_step_in_one_setting():
    middle = pp.Setting(2.0, "0.08", "pause", (0.03, 0.12))
    found = {n.key for n in pp.neighbours(middle)}
    latch = (0.03, 0.12)
    assert found == {pp.Setting(1.5, "0.08", "pause", latch).key, pp.Setting(2.5, "0.08", "pause", latch).key,
                     pp.Setting(2.0, "0.045", "pause", latch).key, pp.Setting(2.0, "0.12", "pause", latch).key,
                     pp.Setting(2.0, "0.08", "pause_then_reset", latch).key,
                     pp.Setting(2.0, "0.08", "pause", (0.05, 0.12)).key}
    corner = pp.Setting(1.0, "0.045", "pause", (0.03, 0.12))
    assert len(pp.neighbours(corner)) == 4


def _stats(sharpe, drawdown=0.1, stop_share=0.0, total=0.1):
    return {"sharpe": sharpe, "max_drawdown": drawdown, "loss_stop_share": stop_share, "total_return": total}


def test_the_plateau_beats_a_lone_peak():
    peak = pp.Setting(1.0, "0.045", "pause", (0.03, 0.12))
    plateau = pp.Setting(2.0, "0.08", "pause", (0.03, 0.12))
    stats = {s.key: _stats(0.1) for s in pp.GRID}
    stats[peak.key] = _stats(2.0)
    for s in [plateau, *pp.neighbours(plateau)]:
        stats[s.key] = _stats(0.8)
    assert pp.plateau_choice(stats) == plateau


def test_a_setting_outside_the_limits_is_never_chosen():
    best = pp.Setting(2.0, "0.08", "pause", (0.03, 0.12))
    stats = {s.key: _stats(0.1) for s in pp.GRID}
    for s in [best, *pp.neighbours(best)]:
        stats[s.key] = _stats(0.8)
    stats[best.key] = _stats(0.8, drawdown=0.31)
    assert pp.plateau_choice(stats) != best
    stats = {s.key: _stats(0.5, stop_share=0.2) for s in pp.GRID}
    assert pp.plateau_choice(stats) is None


def _daily(values_by_year):
    parts = []
    for year, value in values_by_year.items():
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        index = (days.asi8 // 10**6 // DAY)
        rng = np.random.default_rng(year)
        parts.append(pd.Series(value + rng.normal(0, 0.01, len(days)), index=index))
    return pd.concat(parts)


def test_each_year_is_chosen_from_the_years_before_it_only():
    good, other = pp.GRID[10], pp.GRID[40]                    # GRID[40] is not one of GRID[10]'s neighbours
    assert other not in pp.neighbours(good)
    dailies = {s.key: _daily({2021: 0.0, 2022: 0.0, 2023: 0.0}) for s in pp.GRID}
    for s in [good, *pp.neighbours(good)]:                    # a plateau, so the plateau rule can pick its centre
        dailies[s.key] = _daily({2021: 0.003, 2022: 0.003, 2023: -0.01})
    first = pp.walk_forward(dailies, (2021, 2022, 2023))
    dailies[other.key] = _daily({2021: 0.0, 2022: 0.0, 2023: 0.02})        # only the last year changes
    second = pp.walk_forward(dailies, (2021, 2022, 2023))
    assert first["choices"] == second["choices"] and first["choices"][2022] == good.key
    assert first["daily"].equals(second["daily"])                             # the choice made before 2023 holds


def _bars(n, crash_at):
    out = []
    for i in range(n):
        low = D(85) if i == crash_at else D("99.5")
        out.append(Bar(i * HOUR, D(100), D("100.5"), low, D(100), D(10_000_000)))
    return out


def _targets(bars, values):
    return pd.DataFrame({"available_at": np.array([b.available_at for b in bars], dtype=np.int64),
                         "target": values, "ready": True})


@pytest.mark.parametrize("after, first_entry_after_stop", [("pause", 11 + 4), ("pause_then_reset", 27)])
def test_after_a_stop_the_runtime_s_rule_decides_the_next_entry(after, first_entry_after_stop):
    n, crash = 60, 10
    bars = {s: _bars(n, crash if s == "BTCUSDT" else -1) for s in ("BTCUSDT", "ETHUSDT")}
    values = np.full(n, 0.5)
    values[25] = 0.0                                          # the target is 0 once, at the close of bar 25
    targets = {s: _targets(bars[s], values) for s in bars}
    rules = {s: SymbolRules(D("0.001"), D("0.001"), D("5")) for s in bars}
    setting = pp.Setting(1.0, "0.045", after, (0.5, 0.9))   # latches far away
    curve, counts, fills = pp.run(bars, targets, {s: {} for s in bars}, setting, rules=rules,
                                  spec=StrategySpec(execution="market"), start=0, end=n * HOUR, capital=1000.0)
    btc = fills["BTCUSDT"]
    stop_at = [int(f["at"]) for f in btc if f["order"] == "protective_stop"]
    assert stop_at == [(crash + 1) * HOUR]
    again = [int(f["at"]) for f in btc if f["order"] != "protective_stop" and f["side"] == "buy"
             and int(f["at"]) > stop_at[0]]
    # pause: decided as the 4 hours end (bar 14's close) and filled then; pause_then_reset: the 0 at bar 25's close
    # ends the wait, the 0.5 decided at bar 26's close enters
    assert again[0] == first_entry_after_stop * HOUR
    assert counts["BTCUSDT"]["stops"] == 1 and counts["ETHUSDT"]["stops"] == 0

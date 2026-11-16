"""R7c of the runtime extension (docs/arch/runtime-lab-strategies-plan.md): the lab's judge backtests through the
strategy interface, so what it screens is the code the runtime runs.

- An idea registered in trading/domain/strategy/ideas.py (``lab.um_lab.lab_ideas``) gives the judge exactly the
  results the first run's research function gives (``lab.um_lab.legacy``): per period the daily returns, the round
  trips, the entries, the stops and the fills - for a per-symbol idea, one that reads BTCUSDT (not in the lists) and
  a cross-sectional one.
- ``snapshot_window`` is the interface's window of a contract; ``lab_frame`` of it is exactly ``lab.um_lab.frame``.
- A research function behind the interface has no runtime window: it cannot be started live.
The market is the judge tests' synthetic snapshot (test_um_judge_runs).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from lab import um_judge as judge
from lab import um_lab as lab
from tests.unit.lab.test_um_judge_runs import LISTS, make, market  # noqa: F401  (market: a fixture)
from trading.domain.strategy.ideas import lab_frame

ROOT = Path(__file__).resolve().parents[3]
NAMES = ("A1_tsmom_long", "F1_btc_trend_beta", "D2_taker_imbalance", "B1_xs_reversal_long", "E3_vol_target_trend")


def _research():
    path = ROOT / "research/lab/runs/2026-11-16/ideas.py"
    spec = importlib.util.spec_from_file_location("lab_run_2026_11_16_ideas_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_snapshot_window_is_the_lab_s_frame(market):  # noqa: F811  (the imported fixture)
    for symbol in ("BTCUSDT", "UPUSDT", "CRASHUSDT"):
        pd.testing.assert_frame_equal(lab_frame(lab.snapshot_window(symbol, market)), lab.frame(symbol, market))


def test_a_registered_idea_gives_the_judge_the_research_function_s_results(market, tmp_path, monkeypatch):  # noqa: F811
    research = _research()
    frame = lab.frame
    monkeypatch.setattr(lab, "frame", lambda symbol, snapshot=None: frame(symbol, market))   # the research panels
    research._PANELS.clear()
    registered = lab.lab_ideas()
    legacy = {n: (lab.legacy(research.IDEAS[n][0], cross=n in research.XS), research.IDEAS[n][1]) for n in NAMES}
    new = make(market, tmp_path / "new", ideas={n: registered[n] for n in NAMES})
    old = make(market, tmp_path / "old", ideas=legacy)
    traded = 0
    for name in NAMES:
        params = research.IDEAS[name][1][0]
        a = new.run_config((judge.Config(name, params), (0, 1, 2)))["periods"]
        b = old.run_config((judge.Config(name, params), (0, 1, 2)))["periods"]
        assert set(a) == set(b) == {0, 1, 2}
        for k in a:
            pd.testing.assert_series_equal(a[k]["daily"], b[k]["daily"], obj=f"{name} period {k}")
            assert (a[k]["trips"], a[k]["entries"], a[k]["stops"], a[k]["fills"]) == \
                (b[k]["trips"], b[k]["entries"], b[k]["stops"], b[k]["fills"]), (name, k)
            traded += a[k]["fills"]
    assert traded > 0                                                      # not a comparison of idle runs
    assert "BTCUSDT" not in {s for symbols in LISTS for s in symbols}     # F1 read BTCUSDT without trading it


def test_a_research_function_has_no_runtime_window():
    strategy = lab.legacy(lambda df, ctx: pd.Series(0.0, index=df.index))({"w": 1})
    with pytest.raises(NotImplementedError, match="register"):
        strategy.window_hours()
    with pytest.raises(NotImplementedError, match="register"):
        strategy.funding_window()
    assert strategy.references() == ("BTCUSDT",) and lab.legacy(len, cross=True)({}).references() == ()

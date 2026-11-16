"""R7b of the runtime extension (docs/arch/runtime-lab-strategies-plan.md): the research lab's 29 ideas behind the
strategy interface, so the lab's backtest and the runtime run the same code.

- Same results: for every idea and every parameter set of its grid, ``IdeaStrategy.history`` over whole windows gives
  exactly the weights of the lab's own function (research/lab/runs/2026-11-16/ideas.py) on the same data.
- The runtime's trailing window gives the whole history's value: ``latest`` on windows of ``window_hours`` closed bars
  equals ``history`` at the same close (to 1e-9 relative: pandas' rolling statistics differ in the last digits), for
  every idea at closes spread over the last 120 days (the three ideas that carry their position over get a margin of
  ``STATE_MARGIN_DAYS``).
- The registry knows every idea by its lab name, checks its parameters, marks the cross-sectional ones and those that
  read BTCUSDT, and a spec of such an idea must list BTCUSDT.
Data: synthetic symbols (three, seven for the cross-sectional ideas), 420 days of hourly bars with kline fields
and 8-hour funding (seeded).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.domain.strategy import ideas
from trading.domain.strategy.base import STRATEGIES, Window
from trading.domain.strategy.spec import PARAMS, parse_spec

ROOT = Path(__file__).resolve().parents[3]
H, DAY = ideas.H, ideas.DAY
T0 = 1_640_995_200_000                       # 2022-01-01
DAYS = 420
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MORE = ("XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT")      # a cross-sectional idea ranks 7 (its top and bottom 3)
CROSS = {"A8_xs_momentum_top3", "B1_xs_reversal_long", "B2_xs_reversal_ls", "C3_xs_funding_top_long",
         "C4_xs_funding_ls"}
READS_BTC = {"A5_btc_gate_tsmom", "A7_rel_strength_btc", "E4_high_vol_off", "F1_btc_trend_beta",
             "F2_btc_lead_hourly", "F3_bear_short", "F4_regime_switch_ls"}


def _lab_module():
    path = ROOT / "research/lab/runs/2026-11-16/ideas.py"
    spec = importlib.util.spec_from_file_location("lab_run_2026_11_16_ideas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _symbol(seed: int, base: float) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = DAYS * 24
    regime = np.repeat(rng.normal(0, 0.002, n // 240 + 1), 240)[:n]          # trends that come and go
    flush = rng.random(n) < 0.003                                             # an hour that drops 7 % on volume
    close = base * np.exp(np.cumsum(regime + rng.normal(0, 0.008, n) - 0.07 * flush))
    open_ = np.r_[close[0], close[:-1]]
    times = T0 + np.arange(n, dtype=np.int64) * H
    surge_day = np.repeat(rng.random(n // 24 + 1) < 0.04, 24)[:n]             # a whole day of 8x volume
    volume = rng.lognormal(3, 0.6, n) * np.where(flush, 10, 1) * np.where(surge_day, 8, 1)
    quote = volume * close
    hourly = pd.DataFrame({"open_time": times, "open": open_, "high": np.maximum(open_, close) * 1.003,
                           "low": np.minimum(open_, close) * 0.997, "close": close, "volume": volume,
                           "quote_volume": quote, "trades": np.round(volume * 10),
                           "taker_buy_volume": volume * rng.uniform(0.4, 0.6, n),
                           "taker_buy_quote_volume": quote * rng.uniform(0.4, 0.62, n)})
    hourly["available_at"] = hourly["open_time"] + H
    settle = T0 + np.arange(n // 8, dtype=np.int64) * 8 * H
    crowd = np.repeat(rng.normal(0, 7e-4, len(settle) // 30 + 1), 30)[:len(settle)]   # 10-day funding regimes
    funding = pd.Series(rng.normal(2e-4, 3e-4, len(settle)) + crowd, index=settle)
    return hourly, funding


@pytest.fixture(scope="module")
def data():
    out = {}
    for k, symbol in enumerate(SYMBOLS + MORE):
        out[symbol] = _symbol(10 + k, (30_000.0, 2_000.0, 100.0, 0.5, 0.1, 0.4, 15.0)[k])
    return out


def _symbols(name):
    return SYMBOLS + MORE if ideas.IDEAS[name].cross else SYMBOLS


def _windows(data, upto=None, hours=None, symbols=SYMBOLS):
    """Every symbol's window at the close ``upto`` (all data when None), at most ``hours`` bars long."""
    out = {}
    for symbol in symbols:
        hourly, funding = data[symbol]
        rows = hourly if upto is None else hourly[hourly["available_at"] <= upto]
        if hours is not None:
            rows = rows.iloc[-hours:]
        known = funding if upto is None else funding[funding.index <= upto]
        out[symbol] = Window(rows.reset_index(drop=True), known)
    return out


def _lab_frame(hourly, funding):
    """lab.um_lab.frame on a snapshot made of the same data."""
    df = hourly.drop(columns=["available_at"]).set_index("open_time")
    df["funding"] = 0.0
    hours = (funding.index.to_numpy(np.int64) // H) * H
    rates = pd.Series(funding.to_numpy(float), index=hours).groupby(level=0).sum()
    df["funding"] = rates.reindex(df.index).fillna(0.0).to_numpy()
    return df


CASES = [(name, params) for name, idea in ideas.IDEAS.items() for params in idea.grid]


def test_the_ideas_are_the_lab_s_29_with_its_grids():
    lab = _lab_module()
    assert list(ideas.IDEAS) == list(lab.IDEAS) and len(ideas.IDEAS) == 29
    for name, idea in ideas.IDEAS.items():
        assert list(idea.grid) == list(lab.IDEAS[name][1]), name
    assert {n for n, i in ideas.IDEAS.items() if i.cross} == CROSS == set(lab.XS)
    assert {n for n, i in ideas.IDEAS.items() if i.btc} == READS_BTC


@pytest.mark.parametrize("name,params", CASES, ids=[f"{n}-{i}" for i, (n, _) in enumerate(CASES)])
def test_every_idea_gives_exactly_the_lab_s_weights(data, monkeypatch, name, params):
    lab = _lab_module()
    symbols = _symbols(name)
    frames = {s: _lab_frame(*data[s]) for s in symbols}
    monkeypatch.setattr(lab.lab, "frame", lambda s, snapshot=None: frames[s])
    lab._PANELS.clear()
    strategy = ideas.IdeaStrategy(name, params)
    history = strategy.history(_windows(data, symbols=symbols))
    for symbol in symbols:
        # the registered ideas rank in symbol-name order (R7 audit F1): the research function is given that order
        ctx = {"params": params, "btc": frames["BTCUSDT"], "symbol": symbol, "universe": sorted(symbols)}
        expected = lab.IDEAS[name][0](frames[symbol], ctx).reindex(frames[symbol].index).fillna(0.0)
        got = history[symbol]
        assert got["available_at"].tolist() == (frames[symbol].index + H).tolist()
        np.testing.assert_array_equal(got["target"].to_numpy(), expected.to_numpy(dtype=float), err_msg=name)
    assert any((history[s]["target"] != 0).any() for s in symbols), f"{name}: nothing to compare"


CLOSES = [T0 + (DAYS - 120) * DAY + k * 113 * H for k in range(26)]     # every hour of the day, over 120 days


@pytest.mark.parametrize("name", list(ideas.IDEAS))
def test_the_runtime_s_trailing_window_gives_the_whole_history_s_value(data, name):
    symbols = _symbols(name)
    for params in ideas.IDEAS[name].grid:
        strategy = ideas.IdeaStrategy(name, params)
        whole = strategy.history(_windows(data, symbols=symbols))
        hours = strategy.window_hours() + 48                          # what TargetSession keeps
        for close in CLOSES:
            latest = strategy.latest(_windows(data, upto=close, hours=hours, symbols=symbols))
            for symbol in symbols:
                row = whole[symbol][whole[symbol]["available_at"] == close].iloc[0]
                # pandas' rolling std accumulates from where the data starts: the last digit may differ, a decision
                # may not (a flip is a difference of at least 0.95 or of a whole position size)
                target, ready = latest[symbol]
                assert ready and target == pytest.approx(float(row["target"]), rel=1e-9, abs=1e-12), \
                    (name, params, symbol, close)


def test_the_registry_knows_every_idea_and_checks_its_parameters():
    for name, idea in ideas.IDEAS.items():
        strategy = STRATEGIES[name](dict(idea.grid[0]))
        assert strategy.name == name and strategy.cross_sectional is idea.cross
        assert strategy.references() == (("BTCUSDT",) if idea.btc else ())
        assert PARAMS[name](dict(idea.grid[0])) == idea.grid[0]
    with pytest.raises(ValueError, match="exactly the parameters"):
        PARAMS["A1_tsmom_long"]({"n": 20, "extra": 1})
    with pytest.raises(ValueError, match="exactly the parameters"):
        PARAMS["A1_tsmom_long"]({})
    for bad in (0, -5, True, "20", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive number"):
            PARAMS["A1_tsmom_long"]({"n": bad})


def test_the_window_covers_each_idea_s_longest_lookback():
    assert ideas.IdeaStrategy("A3_ma_cross_long", {"fast": 50, "slow": 200}).window_hours() == (200 + 3) * 24
    assert ideas.IdeaStrategy("E2_calm_trend", {"w": 60}).window_hours() == (60 + 180 + 3) * 24
    assert ideas.IdeaStrategy("B4_band_reversion_ls", {"z": 2.0}).window_hours() == \
        (20 + 3 + ideas.STATE_MARGIN_DAYS) * 24
    strategy = ideas.IdeaStrategy("D2_taker_imbalance", {"x": 0.02})
    assert strategy.funding_window() == strategy.window_hours()


def test_ready_only_once_the_window_is_full(data):
    strategy = ideas.IdeaStrategy("A1_tsmom_long", {"n": 20})
    frame = strategy.history(_windows(data))["ETHUSDT"]
    need = strategy.window_hours()
    assert not frame["ready"].iloc[:need - 1].any() and frame["ready"].iloc[need - 1:].all()


def test_an_idea_that_reads_btc_refuses_windows_without_it(data):
    windows = _windows(data)
    windows.pop("BTCUSDT")
    with pytest.raises(ValueError, match="BTCUSDT"):
        ideas.IdeaStrategy("F1_btc_trend_beta", {"n": 10}).history(windows)
    assert set(ideas.IdeaStrategy("A1_tsmom_long", {"n": 20}).history(windows)) == {"ETHUSDT", "SOLUSDT"}


def _spec(name, params, symbols):
    return {"schema": 2, "kind": "target_strategy", "strategy": {"name": name, "params": params},
            "execution": {"band": 0.05, "execution": "market", "ttl_bars": 1}, "symbols": symbols,
            "allow_short": True, "passed": False}


def test_a_spec_can_name_an_idea_and_one_that_reads_btc_must_list_it():
    spec = parse_spec(_spec("B2_xs_reversal_ls", {"days": 2}, ["ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]))
    assert spec.strategy().cross_sectional is True and spec.strategy().params == {"days": 2}
    with pytest.raises(ValueError, match="BTCUSDT"):
        parse_spec(_spec("F1_btc_trend_beta", {"n": 10}, ["ETHUSDT"]))
    assert parse_spec(_spec("F1_btc_trend_beta", {"n": 10}, ["BTCUSDT", "ETHUSDT"])).symbols == ("BTCUSDT",
                                                                                                  "ETHUSDT")
    with pytest.raises(ValueError, match="exactly the parameters"):
        parse_spec(_spec("A1_tsmom_long", {"days": 2}, ["ETHUSDT"]))


def test_lab_frame_sums_the_settlements_of_an_hour_and_puts_zero_elsewhere():
    hourly = pd.DataFrame({"open_time": [T0, T0 + H, T0 + 2 * H], "available_at": [T0 + H, T0 + 2 * H, T0 + 3 * H],
                           "close": [1.0, 2.0, 3.0]})
    funding = pd.Series([1e-4, 2e-4, 5e-5], index=[T0 + 3, T0 + 2 * H + 7, T0 + 2 * H + 9])
    frame = ideas.lab_frame(Window(hourly, funding))
    assert frame.index.tolist() == [T0, T0 + H, T0 + 2 * H] and frame["close"].tolist() == [1.0, 2.0, 3.0]
    assert frame["funding"].tolist() == pytest.approx([1e-4, 0.0, 2.5e-4])
    assert "available_at" not in frame and "open_time" not in frame


# -- the R7 audit -------------------------------------------------------------------------------------------------
def test_a_cross_sectional_idea_does_not_depend_on_the_order_of_its_symbols(data):
    """F1: ties (funding sums of exactly the same rate are common) are broken by symbol name, not by list order."""
    symbols = SYMBOLS + MORE
    for name, params in (("C4_xs_funding_ls", {"window": 3}), ("C3_xs_funding_top_long", {"window": 3}),
                         ("B2_xs_reversal_ls", {"days": 1})):
        strategy = ideas.IdeaStrategy(name, params)
        forward = strategy.history(_windows(data, symbols=symbols))
        backward = strategy.history(_windows(data, symbols=tuple(reversed(symbols))))
        for symbol in symbols:
            pd.testing.assert_frame_equal(forward[symbol], backward[symbol], obj=f"{name} {symbol}")
    tied = {s: Window(w.hourly, pd.Series(1e-4, index=w.funding.index))
            for s, w in _windows(data, symbols=symbols).items()}
    a = ideas.IdeaStrategy("C3_xs_funding_top_long", {"window": 3}).history(tied)
    b = ideas.IdeaStrategy("C3_xs_funding_top_long", {"window": 3}).history(dict(reversed(list(tied.items()))))
    assert all(a[s]["target"].equals(b[s]["target"]) for s in symbols)              # every sum tied
    held = {s for s in symbols if (a[s]["target"] > 0).any()}
    assert held == {"ADAUSDT", "BTCUSDT", "DOGEUSDT"}                               # the first three names


@pytest.mark.parametrize("name,params,named", [
    ("A1_tsmom_long", {"n": 20.0}, "whole number"), ("A1_tsmom_long", {"n": 20.5}, "whole number"),
    ("E4_high_vol_off", {"q": 1.5}, "below 1"), ("A3_ma_cross_long", {"fast": 50, "slow": 20}, "shorter"),
    ("B5_flush_reversal", {"drop": 0.04, "hold": 24.0}, "whole number")])
def test_parameters_every_decision_would_fail_on_are_refused_when_read(name, params, named):
    """F6: refused when the spec is read, not at every bar."""
    with pytest.raises(ValueError, match=named):
        PARAMS[name](params)
    assert PARAMS["B5_flush_reversal"]({"drop": 0.04, "hold": 24})["hold"] == 24


def test_a_day_whose_kline_fields_are_unknown_gives_no_signal(data):
    """F7: an all-unknown day sums to unknown, not 0, so a volume surge is not read into the next known day."""
    hourly, funding = data["ETHUSDT"]
    hourly = hourly.iloc[-24 * 60:].reset_index(drop=True).copy()
    for column in ("quote_volume", "taker_buy_quote_volume", "taker_buy_volume", "trades"):
        hourly.loc[: len(hourly) - 49, column] = np.nan                     # known for the last two days only
    window = {"ETHUSDT": Window(hourly, funding)}
    for name, params in (("D3_surge_fade_short", {"m": 3, "hold": 1}), ("D1_volume_breakout", {"m": 2, "hold": 3}),
                         ("D2_taker_imbalance", {"x": 0.01})):
        assert ideas.IdeaStrategy(name, params).latest(window)["ETHUSDT"][0] == 0.0, name

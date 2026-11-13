"""Overlay study: the predeclared spec/gate are pinned, and only development data is used."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from lab import overlay_study as ov
from lab.data import DAY, HOUR
from lab.strategy import StrategySpec
from lab.study import BASE

T0 = 1_577_836_800_000  # 2020-01-01 00:00 UTC


def test_spec_and_gate_are_the_predeclared_ones():
    # Changing either of these after results exist invalidates the study.
    assert ov.SPEC.lookbacks == (10, 20, 30, 60, 90, 120, 180)
    assert replace(ov.SPEC, lookbacks=BASE.lookbacks, long_only=BASE.long_only) == BASE
    assert ov.SPEC.long_only and not (ov.SPEC.funding_filter or ov.SPEC.vol_spike_cut or ov.SPEC.confirm_4h)
    assert ov.GATE == {
        "calmar_vs_matched_benchmark_min_ratio": 1.5,
        "max_drawdown_at_most_matched_benchmark": True,
        "sharpe_shortfall_vs_benchmark_max": 0.10,
        "cost_x2_total_return_min_exclusive": 0.0,
        "fills_per_symbol_year_max": 150,
    }
    assert sorted(ov.ROBUSTNESS) == ["lookbacks_10_30_90", "lookbacks_20_60_120", "lookbacks_30_90_180"]


def test_stats_start_from_initial_capital():
    equity = pd.Series([100.0, 110.0, 99.0, 121.0], index=[9, 10, 11, 12])
    s = ov.stats(equity)
    assert s["days"] == 3
    assert math.isclose(s["total_return"], 0.21)
    assert math.isclose(s["max_drawdown"], 0.1)


def _gate_inputs(**overrides):
    strategy = {"calmar": 0.6, "max_drawdown": 0.2, "sharpe": 0.7}
    bench = {"calmar": 0.3, "max_drawdown": 0.25, "sharpe": 0.7}
    x2 = {"total_return": 0.4}
    fills = {"BTCUSDT": 60.0, "ETHUSDT": 70.0}
    for key, value in overrides.items():
        where, field = key.split("__")
        {"s": strategy, "b": bench, "x": x2, "f": fills}[where][field] = value
    return strategy, bench, x2, fills


def test_gate_requires_every_condition():
    assert all(c["ok"] for c in ov.gate_checks(*_gate_inputs()).values())
    for override, check in [({"s__calmar": 0.44}, "calmar_vs_matched_benchmark"),
                            ({"s__max_drawdown": 0.26}, "max_drawdown_vs_matched_benchmark"),
                            ({"s__sharpe": 0.59}, "sharpe_vs_benchmark"),
                            ({"x__total_return": 0.0}, "cost_x2_total_return"),
                            ({"f__ETHUSDT": 151.0}, "fills_per_symbol_year")]:
        checks = ov.gate_checks(*_gate_inputs(**override))
        assert not checks[check]["ok"], check
        assert sum(not c["ok"] for c in checks.values()) == 1


def _write_symbol(directory, symbol, n_hours, seed, shock_after=None):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0.00004, 0.006, n_hours)))
    with (directory / f"{symbol}_1h.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for i in range(n_hours):
            t = T0 + i * HOUR
            o, c = (price[i - 1] if i else price[0]), price[i]
            if shock_after is not None and t >= shock_after:
                o, c = o * 3, c * 0.2
            writer.writerow([t, f"{o:.4f}", f"{max(o, c) * 1.002:.4f}", f"{min(o, c) * 0.998:.4f}",
                             f"{c:.4f}", "1000"])
    with (directory / f"{symbol}_funding.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["funding_time", "rate", "mark_price", "mark_source", "observed_funding_time"])
        for t in range(T0, T0 + n_hours * HOUR, 8 * HOUR):
            writer.writerow([t, "0.0001", "100", "exchange", t])


def test_study_reads_only_the_development_window(tmp_path):
    eval_start, dev_end = T0 + 375 * DAY, T0 + 520 * DAY
    n_hours = 600 * 24
    results = []
    for name, shock in (("clean", None), ("shocked", dev_end)):
        data = tmp_path / f"data_{name}"
        data.mkdir()
        _write_symbol(data, "BTCUSDT", n_hours, seed=1, shock_after=shock)
        _write_symbol(data, "ETHUSDT", n_hours, seed=2, shock_after=shock)
        summary = ov.run_study(data, tmp_path / f"out_{name}", eval_start=eval_start, dev_end=dev_end)
        results.append(summary)
    # Prices after dev_end are wildly different, yet nothing in the study changes.
    assert json.dumps(results[0], default=float) == json.dumps(results[1], default=float)

    summary = results[0]
    assert summary["average_exposure"] > 0 and min(summary["fills_per_symbol_year"].values()) > 0
    assert summary["strategy"]["days"] == (dev_end - eval_start) // DAY
    assert set(summary["gate"]["checks"]) == {"calmar_vs_matched_benchmark", "max_drawdown_vs_matched_benchmark",
                                              "sharpe_vs_benchmark", "cost_x2_total_return", "fills_per_symbol_year"}
    frozen = json.loads((tmp_path / "out_clean" / "final_spec.json").read_text())
    assert frozen["kind"] == "btc_eth_multitimeframe_target" and frozen["trials"] == 1
    assert StrategySpec.from_record(frozen["spec"]) == ov.SPEC
    # The screen never unlocks mainnet: binance_target reads ``passed`` as the mainnet gate.
    assert frozen["passed"] is False and frozen["screen_passed"] == summary["gate"]["passed"]

"""Candidate research protocol: selection only on validation, frozen final test."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from data.quality.causal_bars import HOUR, Bar
from runner.offline import research
from runner.offline.context import METRICS_LAG_MS, DerivativesContext
from runner.offline.fixtures import synthetic_bars, synthetic_funding
from runner.offline.models import FEATURE_SETS, design, train_bundle, validate_bundle


def _with_extras(bars):
    return [Bar(b.open_time, b.open, b.high, b.low, b.close, b.volume, b.interval_ms,
                {"quote_volume": b.volume * b.close, "trades": D(50), "taker_buy_volume": b.volume / 2,
                 "taker_buy_quote_volume": b.volume * b.close / 2}) for b in bars]


def _context(bars):
    funding = [(t, 0.0001) for t in range(bars[0].open_time, bars[-1].available_at, 8 * HOUR)]
    metrics = [(t, {"open_interest": 1000 + (t // HOUR) % 7, "ls_ratio": 1.0 + ((t // HOUR) % 5) / 10,
                   "top_trader_ls_ratio": 1.2})
               for t in range(bars[0].open_time, bars[-1].available_at, 5 * 60_000)]
    return DerivativesContext(funding, metrics)


def test_context_is_causal():
    ctx = DerivativesContext([(0, 0.1), (8 * HOUR, 0.2)],
                             [(HOUR - METRICS_LAG_MS, {"open_interest": 1.0, "ls_ratio": None,
                                                       "top_trader_ls_ratio": None}),
                              (HOUR, {"open_interest": 2.0, "ls_ratio": None, "top_trader_ls_ratio": None})])
    at = ctx.at(HOUR)
    # The snapshot stamped exactly at the close is not yet usable; settled funding is.
    assert at["open_interest"] == 1.0 and at["funding_rate"] == 0.1
    assert ctx.at(8 * HOUR)["funding_rate"] == 0.2
    assert ctx.at(8 * HOUR - 1)["funding_rate"] == 0.1
    # Stale snapshots are dropped rather than carried forward.
    assert ctx.at(10 * HOUR)["open_interest"] is None


def test_horizon_labels_end_before_cutoff():
    bars = synthetic_bars(800)
    cutoff = bars[600].open_time
    for horizon in (1, 4, 24):
        samples = design(bars, cutoff, spec={"feature_set": "ohlcv", "horizon": horizon})
        assert samples and max(s[2] for s in samples) < cutoff


def test_derivative_bundle_is_symbol_bound_and_validated():
    bars = _with_extras(synthetic_bars(1200))
    ctx = _context(bars)
    spec = {"feature_set": "derivatives", "horizon": 4, "alpha": 10.0}
    bundle = train_bundle(bars, bars[900].open_time, symbol="ETHUSDT", context=ctx, spec=spec)
    assert bundle["models"]["1h"]["features"] == list(FEATURE_SETS["derivatives"])
    validate_bundle(bundle, bars[900].open_time, symbol="ETHUSDT")
    with pytest.raises(ValueError, match="different symbol"):
        validate_bundle(bundle, bars[900].open_time, symbol="BTCUSDT")


def test_research_selects_on_validation_and_freezes_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "CANDIDATES", {"feature_set": ("ohlcv",), "horizon": (1,),
                                                "alpha": (1.0,), "deadzone": (1.0,)})
    monkeypatch.setattr(research, "MIN_VALIDATION_FILLS", 1)
    monkeypatch.setattr(research, "WARMUP_BARS", 600)
    bars = synthetic_bars(5200)
    split = {"train_end_ms": bars[3120].open_time, "validation_end_ms": bars[4160].open_time}
    summary = research.run_research(bars, synthetic_funding(bars), tmp_path, symbol="BTCUSDT", synthetic=True,
                                    **split)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["candidate_count"] == 1
    assert protocol["split"]["final_test_start_ms"] == bars[4160].open_time
    assert protocol["split"]["train_end_ms"] == bars[3120].open_time
    if summary["chosen"] is None:
        assert not summary["stages"] and "NO_CANDIDATE" in protocol["outcome"]
    else:
        model = json.loads((tmp_path / "model.json").read_text())
        assert model["sha256"] == protocol["final_model_sha256"]
        assert model["cutoff_ms"] <= protocol["split"]["final_test_start_ms"]
        assert set(summary["stages"]) == {"final_test", "final_test_cost_x2", "buy_hold", "trend_50"}
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert gate["passed"] is False  # synthetic data must never pass the live gate trivially
    with pytest.raises(ValueError, match="empty output"):
        research.run_research(bars, synthetic_funding(bars), tmp_path, symbol="BTCUSDT", synthetic=True, **split)


def test_gate_requires_every_condition():
    good = {"net_return_pct": 10.0, "fills": 200, "average_gross_exposure": 0.5, "sharpe_daily": 1.5}
    summary = {"stages": {"final_test": good, "final_test_cost_x2": {**good, "net_return_pct": 2.0},
                          "buy_hold": {"net_return_pct": 5.0, "average_gross_exposure": 1.0},
                          "trend_50": {"net_return_pct": 1.0, "average_gross_exposure": 1.0}}}
    assert research.passes_gate(summary) == (True, [])
    summary["stages"]["final_test_cost_x2"]["net_return_pct"] = -1.0
    ok, reasons = research.passes_gate(summary)
    assert not ok and any("2x costs" in r for r in reasons)


def test_validation_only_never_reads_past_the_validation_window(tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(research, "CANDIDATES", {"feature_set": ("ohlcv",), "horizon": (1,),
                                                "alpha": (1.0,), "deadzone": (1.0,)})
    monkeypatch.setattr(research, "MIN_VALIDATION_FILLS", 1)
    monkeypatch.setattr(research, "WARMUP_BARS", 600)
    bars = synthetic_bars(5200)
    val_end = 4160
    split = {"train_end_ms": bars[3120].open_time, "validation_end_ms": bars[val_end].open_time}
    # Later data both changed AND appended (as a data refresh would): with fixed split
    # dates neither the windows nor any validation result may move.
    changed = [replace(b, open=b.open * 2, high=b.high * 2, low=b.low * 2, close=b.close * 2)
               if i >= val_end else b for i, b in enumerate(synthetic_bars(6000))]
    outputs = []
    for name, series in (("clean", bars), ("changed", changed)):
        out = tmp_path / name
        summary = research.run_research(series, synthetic_funding(series), out, symbol="BTCUSDT",
                                        synthetic=True, validation_only=True, **split)
        protocol = json.loads((out / "protocol.json").read_text())
        assert protocol["mode"] == "validation_only" and protocol["final_test_views"] == 0
        assert "final_model_sha256" not in protocol and not (out / "model.json").exists()
        assert not summary["stages"] and not list(out.glob("final_test*"))
        gate = json.loads((out / "gate.json").read_text())
        assert gate["kind"] == "validation_only" and gate["passed"] == summary["validation_gate"]["passed"]
        outputs.append((protocol["split"], protocol["validation_candidates"], protocol["validation_gate_result"],
                        summary["chosen"]))
    # Every bar after the validation window was doubled; nothing in the result changed.
    assert outputs[0] == outputs[1]


def test_validation_gate_requires_every_condition():
    chosen = {"validation_sharpe_daily": 0.8, "fills": 100}
    stress = {"daily_sharpe": 0.3}
    assert research.passes_validation_gate(chosen, stress, 1.0) == (True, [])
    for bad_chosen, bad_stress, years, text in [({**chosen, "validation_sharpe_daily": 0.5}, stress, 1.0, "Sharpe"),
                                                (chosen, {"daily_sharpe": 0.0}, 1.0, "2x costs"),
                                                (chosen, {"daily_sharpe": None}, 1.0, "2x costs"),
                                                ({**chosen, "fills": 151}, stress, 1.0, "fills per year")]:
        ok, reasons = research.passes_validation_gate(bad_chosen, bad_stress, years)
        assert not ok and len(reasons) == 1 and text in reasons[0]
    assert research.passes_validation_gate(None, None, 1.0)[0] is False
    assert research.VALIDATION_GATE == {"validation_sharpe_min_exclusive": 0.5,
                                        "validation_cost_x2_sharpe_min_exclusive": 0.0,
                                        "validation_fills_per_year_max": 150}


def test_split_dates_must_be_bar_open_times_in_order():
    bars = synthetic_bars(5200)
    assert research.split_indices(bars, bars[3120].open_time, bars[4160].open_time) == (3120, 4160)
    with pytest.raises(ValueError, match="not the open time"):
        research.split_indices(bars, bars[3120].open_time + 1, bars[4160].open_time)
    with pytest.raises(ValueError, match="not the open time"):
        research.split_indices(bars, bars[3120].open_time, bars[-1].open_time + HOUR)
    with pytest.raises(ValueError, match="split must satisfy"):
        research.split_indices(bars, bars[4160].open_time, bars[3120].open_time)

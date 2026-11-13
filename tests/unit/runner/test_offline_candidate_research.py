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
    summary = research.run_research(bars, synthetic_funding(bars), tmp_path, symbol="BTCUSDT", synthetic=True)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["candidate_count"] == 1
    assert protocol["split"]["final_test_start_ms"] == bars[int(5200 * 0.8)].open_time
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
        research.run_research(bars, synthetic_funding(bars), tmp_path, symbol="BTCUSDT", synthetic=True)


def test_gate_requires_every_condition():
    good = {"net_return_pct": 10.0, "fills": 200, "average_gross_exposure": 0.5, "sharpe_daily": 1.5}
    summary = {"stages": {"final_test": good, "final_test_cost_x2": {**good, "net_return_pct": 2.0},
                          "buy_hold": {"net_return_pct": 5.0, "average_gross_exposure": 1.0},
                          "trend_50": {"net_return_pct": 1.0, "average_gross_exposure": 1.0}}}
    assert research.passes_gate(summary) == (True, [])
    summary["stages"]["final_test_cost_x2"]["net_return_pct"] = -1.0
    ok, reasons = research.passes_gate(summary)
    assert not ok and any("2x costs" in r for r in reasons)

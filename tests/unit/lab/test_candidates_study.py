"""Development run of the three forward-test candidates on synthetic data."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab import candidates_study as cs
from lab import trials
from lab.pair import build, labels, load_inputs
from lab.pair_study import fit_predict
from tests.unit.lab.conftest import CANDIDATE_WINDOW as WINDOW
from tests.unit.lab.test_pair_study import write_snapshot

HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_609_459_200_000          # 2021-01-01
SPECS = ("pair_ridge_pd_7d", "dir_ridge_7d", "flush_reversal_72h")


def _spec(root, name):
    return json.loads((root / "out" / f"{name}.json").read_text())


def test_the_study_freezes_three_specs_and_records_two_trials_once(study):
    root, market, premium, registry, summary = study
    assert sorted(p.name for p in (root / "out").iterdir()) == sorted([f"{n}.json" for n in SPECS] + ["summary.json"])
    rows = trials.load(registry)
    assert [(r["family"], r["trial"], r["count"]) for r in rows] == [
        ("directional_btc_eth", "dir_ridge_7d", 1), ("directional_btc_eth", "flush_reversal_72h", 1)]
    assert rows[0]["window"] == [WINDOW["b_start"], WINDOW["limit"]]
    assert rows[1]["window"] == [WINDOW["c_start"], WINDOW["limit"]]
    with pytest.raises(ValueError, match="empty output directory"):
        cs.run_study(market, premium, root / "out", registry=registry, **WINDOW)
    with pytest.raises(ValueError, match="runs once"):
        cs.run_study(market, premium, root / "again", registry=registry, **WINDOW)
    assert not (root / "again").exists()


def test_every_spec_refuses_mainnet_and_registration_follows_the_floor(study):
    root, _, _, _, summary = study
    for name in SPECS:
        spec = _spec(root, name)
        assert spec["passed"] is False and spec["mainnet"].startswith("refused")
        assert spec["fit_limit_ms"] == WINDOW["limit"] and spec["forward_gate"] == cs.FORWARD_GATE
    assert _spec(root, "pair_ridge_pd_7d")["registered"] is True
    for name in ("dir_ridge_7d", "flush_reversal_72h"):
        floor = summary[name]["floor"]
        assert _spec(root, name)["registered"] is all(check["ok"] for check in floor.values())
    assert set(summary["flush_reversal_72h"]["floor"]) == {"base_total_return", "no_wipeout",
                                                           "fills_per_symbol_year", "events"}


def test_the_frozen_pair_model_is_the_study_fit_on_every_label_that_ended_by_the_limit(study):
    root, market, premium, _, _ = study
    limit = WINDOW["limit"]
    closes, feats = build(load_inputs(market, premium, limit))
    lab = labels(closes, 7)
    x = feats[list(cs.PAIR_FEATURES)].to_numpy(float)
    y = (lab["y"] / (feats["sigma_pair"] * np.sqrt(7))).to_numpy(float)
    rows = np.isfinite(x).all(axis=1) & np.isfinite(y) & (lab["label_end"].to_numpy() <= limit)
    spec = _spec(root, "pair_ridge_pd_7d")
    assert spec["model"]["rows"] == int(rows.sum()) and spec["features"] == list(cs.PAIR_FEATURES)
    assert (feats.index[rows] + 7 * DAY).max() == limit       # the last usable label ends exactly at the limit
    np.testing.assert_allclose(cs.predict_frozen(spec["model"], x[rows][:25]),
                               fit_predict("ridge", x[rows].copy(), y[rows], x[rows][:25]), rtol=1e-9)


def test_both_final_fits_use_no_label_that_ends_after_the_limit_even_when_later_data_is_loaded(study):
    root, market, premium, _, _ = study
    limit = WINDOW["limit"]
    later = load_inputs(market, premium, limit + 9 * DAY)      # nine more days of data than the study had
    assert cs.fit_pair(later, limit) == _spec(root, "pair_ridge_pd_7d")["model"]
    assert cs.fit_directional(cs.b_dataset(later), limit) == _spec(root, "dir_ridge_7d")["model"]
    assert cs.fit_pair(later, limit + 9 * DAY)["rows"] > _spec(root, "pair_ridge_pd_7d")["model"]["rows"]


def test_a_candidate_below_its_floor_is_frozen_as_not_registered(study, tmp_path, monkeypatch):
    _, market, premium, _, _ = study
    failing = {"base_total_return": {"value": -0.1, "ok": False}}
    monkeypatch.setattr(cs, "floor", lambda result, events=None: failing)
    summary = cs.run_study(market, premium, tmp_path / "out", registry=tmp_path / "trials.jsonl", **WINDOW)
    registered = {name: json.loads((tmp_path / "out" / f"{name}.json").read_text())["registered"] for name in SPECS}
    assert registered == {"pair_ridge_pd_7d": True, "dir_ridge_7d": False, "flush_reversal_72h": False}
    assert summary["dir_ridge_7d"]["registered"] is False and summary["flush_reversal_72h"]["registered"] is False
    assert len(trials.load(tmp_path / "trials.jsonl")) == 2      # a closed candidate is still a recorded trial


def test_walk_forward_predictions_ignore_every_price_after_the_decision(tmp_path):
    first, limit, cut = T0 + 300 * DAY, T0 + 440 * DAY, T0 + 360 * DAY       # the cut is inside the first block

    def predictions(name, wild_after):
        market, premium = write_snapshot(tmp_path / name, 450, seed=21, wild_after=wild_after)
        data = cs.b_dataset(load_inputs(market, premium, limit))
        return cs.b_walk_forward(data, first_test=first, limit=limit)[0]

    calm, wild = predictions("calm", None), predictions("wild", cut)
    for symbol in cs.SYMBOLS:
        before = calm[symbol].index <= cut
        assert np.isfinite(calm[symbol][before]).sum() >= 50
        pd.testing.assert_series_equal(calm[symbol][before], wild[symbol][before])
        assert not np.allclose(calm[symbol][~before].dropna(), wild[symbol][~before].dropna())


def _result(total=0.1, wiped=False, fills=20.0):
    side = {"total_return": total, "wiped": wiped, "fills_per_symbol_year": {"BTCUSDT": fills, "ETHUSDT": 1.0}}
    return {"base": side, "cost_x2": {**side, "wiped": False}}


@pytest.mark.parametrize("result,events,failed", [
    (_result(), None, set()),
    (_result(total=0.0), None, {"base_total_return"}),            # must be strictly positive
    (_result(total=-0.2), None, {"base_total_return"}),
    (_result(total=float("nan")), None, {"base_total_return"}),
    (_result(wiped=True), None, {"no_wipeout"}),
    (_result(fills=150.0), None, set()),
    (_result(fills=150.1), None, {"fills_per_symbol_year"}),
    (_result(), 10, set()),
    (_result(), 9, {"events"}),
])
def test_the_registration_floor(result, events, failed):
    checks = cs.floor(result, events=events)
    assert {name for name, check in checks.items() if not check["ok"]} == failed
    assert ("events" in checks) is (events is not None)


def test_a_wipeout_under_doubled_costs_also_fails_the_floor():
    result = _result()
    result["cost_x2"]["wiped"] = True
    assert cs.floor(result)["no_wipeout"]["ok"] is False


def test_flush_segments_measure_each_hold_from_entry_to_one_hour_after_the_target_returns_to_zero():
    times = [T0 + i * HOUR for i in range(1, 11)]
    signal = pd.DataFrame({"available_at": times,
                           "target": [0, .5, .5, 0, 0, .5, .5, .5, .5, .5],
                           "entry": [False, True, False, False, False, True, False, False, False, False]})
    curve = pd.DataFrame({"at": times, "equity": [100, 100, 104, 108, 110, 110, 99, 98, 97, 96.8]})
    segments = cs.flush_segments(SimpleNamespace(curve=curve), signal, start=T0, end=T0 + 20 * HOUR)
    assert [(s["entry"], s["closed"]) for s in segments] == [(times[1], times[4]), (times[5], times[9])]
    assert [s["return"] for s in segments] == pytest.approx([0.10, -0.12])   # the open hold is marked at the end
    late = cs.flush_segments(SimpleNamespace(curve=curve), signal, start=times[2], end=T0 + 20 * HOUR)
    assert [s["entry"] for s in late] == [times[5]]          # an entry before the window is not counted

"""Forward replay of the three frozen candidates on synthetic data."""
from __future__ import annotations

import json

import pytest

from lab import candidates_forward as cf
from lab import forward
from tests.unit.lab.test_pair_study import write_snapshot

HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_609_459_200_000          # 2021-01-01
SPECS = ("pair_ridge_pd_7d", "dir_ridge_7d", "flush_reversal_72h")


def _spec(root, name):
    return json.loads((root / "out" / f"{name}.json").read_text())


def test_a_forward_replay_ignores_everything_from_its_end_date_on(study, tmp_path):
    root, market, premium, _, _ = study
    start, end = T0 + 330 * DAY, T0 + 400 * DAY
    wild_market, wild_premium = write_snapshot(tmp_path / "wild", 420, wild_after=end)
    for name in SPECS:
        frozen = _spec(root, name)
        calm = cf.evaluate_forward(frozen, market, premium, start, end)
        wild = cf.evaluate_forward(frozen, wild_market, wild_premium, start, end)
        assert json.dumps(calm, sort_keys=True, default=float) == json.dumps(wild, sort_keys=True, default=float)
        assert calm["window"] == {"start_ms": start, "end_ms": end, "days": 70}
        assert calm["verdict"] in {"continue", "stop", "insufficient_sample"}
    assert set(cf.evaluate_forward(_spec(root, "dir_ridge_7d"), market, premium, start, end)["checks"]) == {
        "cost_x2_total_return", "ic", "max_drawdown_daily", "fills_per_symbol_year"}


def test_a_forward_replay_refuses_data_that_stops_before_the_end_date(study):
    root, market, premium, _, _ = study
    with pytest.raises(ValueError, match="data ends before"):
        cf.evaluate_forward(_spec(root, "dir_ridge_7d"), market, premium, T0 + 330 * DAY, T0 + 430 * DAY)
    with pytest.raises(ValueError, match="premium"):
        cf.evaluate_forward(_spec(root, "dir_ridge_7d"), market, None, T0 + 330 * DAY, T0 + 400 * DAY)


def test_the_forward_tool_refuses_a_candidate_closed_by_its_floor_and_unknown_kinds(study, tmp_path):
    root, market, premium, _, _ = study
    closed = {**_spec(root, "dir_ridge_7d"), "registered": False}
    (tmp_path / "closed.json").write_text(json.dumps(closed))
    with pytest.raises(PermissionError, match="development floor"):
        forward.evaluate(tmp_path / "closed.json", market, T0 + 400 * DAY, premium_dir=premium)
    (tmp_path / "odd.json").write_text(json.dumps({"kind": "something_else"}))
    with pytest.raises(ValueError, match="unsupported spec kind"):
        forward.evaluate(tmp_path / "odd.json", market, T0 + 400 * DAY, premium_dir=premium)


def test_forward_verdicts():
    ok, bad = {"value": 1, "ok": True}, {"value": 0, "ok": False}
    assert cf.verdict("directional_ridge_frozen", {"checks": {"a": ok, "b": ok}}) == "continue"
    assert cf.verdict("directional_ridge_frozen", {"checks": {"a": ok, "b": bad}}) == "stop"
    assert cf.verdict("flush_reversal_rule", {"checks": {"events": bad, "b": ok}}) == "insufficient_sample"
    assert cf.verdict("flush_reversal_rule", {"checks": {"events": ok, "b": bad}}) == "stop"
    assert cf.verdict("eth_btc_pair_frozen", {"checks": {"events": bad}}) == "stop"   # only C may defer

"""Exchange leverage / margin settings come only from human config; isolated stops must sit inside liquidation."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from runner.binance_main import check_stop_inside_liquidation, exchange_settings


def test_defaults_are_unchanged():
    assert exchange_settings({}) == (3, None)


def test_isolated_ten_x_from_env():
    assert exchange_settings({"BINANCE_EXCHANGE_LEVERAGE": "10", "BINANCE_MARGIN_TYPE": "isolated"}) == (10, "ISOLATED")


@pytest.mark.parametrize("env", [{"BINANCE_EXCHANGE_LEVERAGE": "11"}, {"BINANCE_EXCHANGE_LEVERAGE": "0"},
                                 {"BINANCE_MARGIN_TYPE": "portfolio"}])
def test_invalid_settings_are_refused(env):
    with pytest.raises(RuntimeError):
        exchange_settings(env)


def test_isolated_stop_must_be_inside_half_the_liquidation_distance():
    env = {"BINANCE_EXCHANGE_LEVERAGE": "10", "BINANCE_MARGIN_TYPE": "ISOLATED"}
    check_stop_inside_liquidation(D("0.04"), env)                       # 4% < 4.75%
    with pytest.raises(RuntimeError, match="stop-pct <= 0.0475"):
        check_stop_inside_liquidation(D("0.08"), env)                   # binance_target's default
    check_stop_inside_liquidation(D("0.08"), {"BINANCE_EXCHANGE_LEVERAGE": "10"})   # margin type unmanaged
    check_stop_inside_liquidation(D("0.08"), {"BINANCE_EXCHANGE_LEVERAGE": "10", "BINANCE_MARGIN_TYPE": "CROSSED"})


def test_target_runtime_refuses_an_isolated_stop_too_close_to_liquidation(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "10")
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-x")
    monkeypatch.setattr("runner.binance_target.load_spec", lambda path: {
        "symbols": ["ETHUSDT"], "spec": {"lookbacks": [10, 30, 90]}, "passed": False})
    with pytest.raises(RuntimeError, match="liquidation distance"):
        TargetRuntime(spec_path=tmp_path / "spec.json", state_dir=tmp_path / "state", mode="execution",
                      market=object(), account=object(), testnet=True, stop_pct=D("0.08"))

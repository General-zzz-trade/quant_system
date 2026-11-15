"""With equity unknown, only a whole-position close goes out (migration step 13d; R13, decided 2026-11-15).

Sizing needs equity, closing everything does not: when the target is 0 or on the other side, the session
closes the whole confirmed position. A smaller target on the same side cannot be sized, so nothing goes out,
and no entry ever does. The research plan is unchanged.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _start, _until_position
from tests.unit.lab.test_target_runtime import _advance
from trading.domain.rebalance import Intent, close_without_equity


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


@pytest.mark.parametrize("target,qty,expected", [
    (0.0, D("0.5"), [Intent("sell", D("0.5"), True)]),
    (-0.3, D("0.5"), [Intent("sell", D("0.5"), True)]),
    (0.3, D("-0.5"), [Intent("buy", D("0.5"), True)]),
    (0.2, D("0.5"), []),
    (-0.2, D("-0.5"), []),
    (0.5, D("0"), []),
])
def test_only_a_target_of_zero_or_the_other_side_closes_the_whole_position(target, qty, expected):
    assert close_without_equity(target, qty) == expected


def _equity_unknown(world):
    btc = world["fakes"]["BTCUSDT"]
    original = btc.request_signed

    def down(*, method, path, params=None):
        if path == "/fapi/v2/account":
            raise OSError("account endpoint down")
        return original(method=method, path=path, params=params)

    btc.request_signed = down


@pytest.mark.parametrize("target,closed", [(0.0, True), ("flip", True), ("smaller", False)])
def test_the_runtime_closes_only_what_it_can_without_equity(tmp_path, target, closed):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    position = btc.positions["BTCUSDT"]
    _equity_unknown(world)
    sign = 1 if position > 0 else -1
    weight = {0.0: 0.0, "flip": -0.5 * sign, "smaller": 0.01 * sign}[target]
    session._target = lambda: (weight, True)
    posts = len(btc.posts)
    _advance(world, runtime, i + 1)                       # an hour later: the equity reading is stale (unknown)
    assert session.equity.current() == 0
    new = btc.posts[posts:]
    assert all(p.get("reduceOnly") in (True, "true") for p in new)          # never an entry
    assert (btc.positions["BTCUSDT"] == 0) is closed
    assert len(new) == (1 if closed else 0)
    _tick(world, runtime)
    assert all(p.get("reduceOnly") in (True, "true") for p in btc.posts[posts:])
    reasons = [r["reason"] for r in _events(tmp_path, "intent") if r["symbol"] == "BTCUSDT"][-1:]
    assert reasons == (["rebalance"] if closed else reasons)

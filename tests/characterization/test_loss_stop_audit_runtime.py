"""The strategy loss stop in the running runtime after the independent audit of 2026-11-16 (PR #106): the auditor's
reproductions, asserting the decided behaviour. The fake account holds 10,000 USDT, the strategy capital is 300, so
the stop is at -45 USDT."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _tick, _until
from tests.characterization.test_runtime_decisions import _events
from tests.characterization.test_strategy_capital_sizing import _execution
from tests.characterization.test_strategy_loss_stop_runtime import _flat, _holding, _kill
from tests.support.state_db import state_dir_db
from tests.unit.binance.fake_venue import FakeBinance
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.adapters.sqlite.schema import STATE_DB

SCOPE = "binance-testnet-ut"


@pytest.fixture(autouse=True)
def _capital(monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    monkeypatch.delenv("STRATEGY_LOSS_STOP_PCT", raising=False)


def _under_water(fake, symbol, usdt):
    """Put the mark so that the fake's position on ``symbol`` is ``usdt`` under water."""
    qty, entry = fake.positions[symbol], fake.entry[symbol]
    fake.price[symbol] = entry - D(usdt) / qty


def _position_rows(world, answer):
    """Replace the venue's answer to position requests: ``answer(params, real)`` -> rows."""
    client = world["adapter"]._client
    real = client.request_signed

    def request_signed(*, method, path, params=None):
        if path == "/fapi/v2/positionRisk":
            return answer(params or {}, lambda: real(method=method, path=path, params=params))
        return real(method=method, path=path, params=params)
    client.request_signed = request_signed


def _loss_issues(tmp_path):
    return [r for r in _events(tmp_path, "issue") if r.get("issue_kind") == "strategy_loss_stop"]


# -- audit findings 1 and 2: the stop reads the strategy's own unrealized PnL --------------------------------------
def test_an_empty_whole_account_answer_does_not_hide_the_strategy_s_loss(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    _position_rows(world, lambda params, real: real() if params.get("symbol") else [])   # "no position anywhere"
    _under_water(btc, "BTCUSDT", 60)                                # past the 45 USDT limit
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]


def test_a_held_symbol_the_venue_reports_no_row_for_is_an_unknown_loss_not_zero(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    held = btc.positions["BTCUSDT"]
    _position_rows(world, lambda params, real: [] if params.get("symbol") == "BTCUSDT" else real())
    _under_water(btc, "BTCUSDT", 60)
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is False and btc.positions["BTCUSDT"] == held      # nothing is closed
    (finding,) = runtime.exits.loss_findings
    assert "strategy loss unknown" in finding and "BTCUSDT" in finding
    assert _loss_issues(tmp_path) and not any(s.executor.ready for s in runtime.sessions.values())


def test_a_manual_position_s_profit_on_a_strategy_symbol_does_not_hide_the_strategy_s_loss(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    eth = world["fakes"]["ETHUSDT"]
    assert eth.positions.get("ETHUSDT", D(0)) == 0 and runtime.sessions["ETHUSDT"].position_qty() == 0
    _under_water(btc, "BTCUSDT", 60)
    eth.positions["ETHUSDT"] = D("1")                    # the owner's manual long on the shared account, 100 USDT up
    eth.entry["ETHUSDT"] = eth.mark("ETHUSDT") - D("100")
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]


def test_a_manual_position_s_loss_does_not_trip_the_stop_and_is_not_closed_by_it(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    eth = world["fakes"]["ETHUSDT"]
    assert eth.positions.get("ETHUSDT", D(0)) == 0 and runtime.sessions["ETHUSDT"].position_qty() == 0
    held = btc.positions["BTCUSDT"]
    eth.positions["ETHUSDT"] = D("1")                    # the owner's manual long, 60 USDT under water
    eth.entry["ETHUSDT"] = eth.mark("ETHUSDT") + D("60")
    _tick(world, runtime, 61)
    _tick(world, runtime)
    _tick(world, runtime)
    assert _kill(tmp_path)["active"] is False and runtime.exits.loss_findings == []
    assert eth.positions["ETHUSDT"] == D("1") and btc.positions["BTCUSDT"] == held


def test_the_second_symbol_s_open_loss_counts(tmp_path):
    world, runtime = _execution(tmp_path)
    btc, eth = world["fakes"]["BTCUSDT"], world["fakes"]["ETHUSDT"]
    _until(world, runtime, lambda: eth.positions.get("ETHUSDT", D(0)) != 0)
    if btc.positions.get("BTCUSDT", D(0)) != 0:
        btc.price["BTCUSDT"] = btc.entry["BTCUSDT"]      # nothing lost on the first symbol
    _under_water(eth, "ETHUSDT", 50)
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]


# -- audit finding 7: an owner of an earlier spec in the same scope still counts ----------------------------------
def test_an_earlier_spec_s_owner_in_the_ledger_counts_for_the_base_and_for_the_stop(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    state_dir_db(tmp_path / "exec")
    ledger = SqliteOrderJournal(tmp_path / "exec" / STATE_DB, SCOPE)
    world, runtime = _execution(tmp_path)
    sol = world["fakes"]["SOLUSDT"] = FakeBinance(world["rows"]["ETHUSDT"], clock=lambda: world["clock"]["now"])
    # SOLUSDT was traded under an earlier spec: 30 USDT lost and realized, nothing held now.
    ledger.set_position_manually("target.SOLUSDT", {"qty": "0", "entry_price": "0", "realized_pnl": "-30", "fees": {}},
                                 venue={}, reason="test: an earlier spec's ledger", at=world["clock"]["now"] - 3600)
    sol.add_income("SOLUSDT", "-10", sol.now_ms() - 1_000)           # and its funding since then
    _tick(world, runtime, 601)
    assert runtime.sessions["BTCUSDT"].sleeve_equity() == (D("300") - D("30") - D("10")) / 2
    assert _kill(tmp_path)["active"] is False                         # 40 lost: short of 45
    sol.add_income("SOLUSDT", "-5", sol.now_ms())
    _tick(world, runtime, 601)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]
    assert _flat(world)

"""The strategy loss stop in the running runtime after the independent audit of 2026-11-16 (PR #106): the auditor's
reproductions, asserting the decided behaviour. The fake account holds 10,000 USDT, the strategy capital is 300, so
the stop is at -45 USDT."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from tests.characterization.test_exit_paths import _tick, _until
from tests.characterization.test_runtime_decisions import _events, _runtime
from tests.characterization.test_strategy_capital_sizing import _execution
from tests.characterization.test_strategy_loss_stop_runtime import _flat, _holding, _kill
from tests.support.state_db import state_dir_db
from tests.unit.binance.fake_venue import HOUR, FakeBinance
from tests.unit.lab.test_target_runtime import _advance
from tests.unit.trading.test_market_failures import _due, _market
from trading.adapters.binance.rest import BinanceRetryableError
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


def _manual_long(world, runtime, symbol, usdt):
    """A person's own long of 1 on ``symbol`` of the shared account, netted with whatever the strategy holds there;
    the venue's row for the symbol then shows ``usdt`` of unrealized PnL that is not the strategy's. Whether the
    strategy holds the second symbol at this point depends on the real-time order rate limit, so an entry still
    waiting for it is dropped first: the scenario is the same either way."""
    fake, session = world["fakes"][symbol], runtime.sessions[symbol]
    session.drop_deferred("test: no entry waiting")
    own = fake.positions.get(symbol, D(0))
    assert session.position_qty() == own
    fake.positions[symbol] = own + D("1")
    fake.entry[symbol] = fake.mark(symbol) - D(usdt) / fake.positions[symbol]
    return fake.positions[symbol]


def test_a_manual_position_s_profit_on_a_strategy_symbol_does_not_hide_the_strategy_s_loss(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    _under_water(btc, "BTCUSDT", 60)
    _manual_long(world, runtime, "ETHUSDT", 100)         # the owner's manual long, 100 USDT up
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]


def test_a_manual_position_s_loss_does_not_trip_the_stop_and_is_not_closed_by_it(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    eth, held = world["fakes"]["ETHUSDT"], btc.positions["BTCUSDT"]
    netted = _manual_long(world, runtime, "ETHUSDT", -60)             # the owner's manual long, 60 USDT under water
    _tick(world, runtime, 61)
    _tick(world, runtime)
    _tick(world, runtime)
    assert _kill(tmp_path)["active"] is False and runtime.exits.loss_findings == []
    assert eth.positions["ETHUSDT"] == netted and btc.positions["BTCUSDT"] == held


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


# -- audit finding 3: the stop is read while a restart waits for market data and in the warm-up -------------------
def test_the_flatten_happens_in_the_cycle_of_the_trip(tmp_path):
    world, runtime, btc, _ = _holding(tmp_path)
    _under_water(btc, "BTCUSDT", 60)
    _tick(world, runtime, 61)                                         # one cycle
    assert _kill(tmp_path)["active"] is True and _kill(tmp_path)["flatten"] is True and _flat(world)


def test_a_restart_waiting_for_market_data_reads_the_loss_stop_and_closes_in_the_same_hold(tmp_path):
    world, runtime, btc, i = _holding(tmp_path)
    qty, entry = btc.positions["BTCUSDT"], btc.entry["BTCUSDT"]
    runtime = _runtime(tmp_path, world)                               # a restart while market data is down
    _market(world, BinanceRetryableError("Network error: down"))
    btc.price["BTCUSDT"] = entry - D("20") / qty                      # 20 USDT under water: short of the 8 % stop
    btc.add_income("BTCUSDT", "-40", btc.now_ms() + 1_000)            # and 40 USDT of funding paid: 60 in all
    _due(world, i)
    assert runtime.warmup_or_hold() is False                          # the hold: no market data, the exit side runs
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]
    assert _flat(world)


def test_a_restart_with_the_switch_cleared_sets_it_again_in_the_warm_up_before_any_entry(tmp_path):
    world, runtime, btc, i = _holding(tmp_path)
    btc.add_income("BTCUSDT", "-50", btc.now_ms() + 1_000)
    _tick(world, runtime, 601)
    _tick(world, runtime)
    assert _kill(tmp_path)["active"] is True and _flat(world)
    (tmp_path / "exec" / "kill_switch.json").write_text(json.dumps({"active": False, "reason": "cleared"}))
    runtime = _runtime(tmp_path, world)                               # a person cleared the switch and restarted
    posts = sum(len(f.posts) for f in world["fakes"].values())
    runtime.warmup()
    assert _kill(tmp_path)["active"] is True and _kill(tmp_path)["flatten"] is True
    for k in range(3):                                                # the loss is still in the ledger: no entry
        _advance(world, runtime, i + k)
        assert _flat(world)
    assert sum(len(f.posts) for f in world["fakes"].values()) == posts


# -- audit finding 6: once reached, a cleared switch lets no entry through ----------------------------------------
def test_a_cleared_switch_lets_no_entry_through_and_is_set_again_at_the_next_reconciliation(tmp_path):
    world, runtime, btc, i = _holding(tmp_path)
    btc.add_income("BTCUSDT", "-50", btc.now_ms() + 1_000)
    _tick(world, runtime, 601)
    _tick(world, runtime)
    assert _kill(tmp_path)["active"] is True and _flat(world)
    _advance(world, runtime, i + 1)
    close = (world["rows"]["BTCUSDT"][i + 2]["start"] + HOUR) / 1000
    world["clock"]["now"] = close - 20                 # a cycle 20 s before the next bar closes
    runtime.cycle()
    (tmp_path / "exec" / "kill_switch.json").write_text(json.dumps({"active": False, "reason": "cleared"}))
    posts = sum(len(f.posts) for f in world["fakes"].values())
    _advance(world, runtime, i + 2)                    # 25 s later the bar closes: the market side decides first
    assert _flat(world) and sum(len(f.posts) for f in world["fakes"].values()) == posts
    assert _kill(tmp_path)["active"] is True and _kill(tmp_path)["flatten"] is True     # set again in that cycle
    for k in range(3, 6):
        _advance(world, runtime, i + k)
        assert _flat(world)
    reached = [r for r in _loss_issues(tmp_path) if "strategy loss stop reached" in r["issue"]]
    assert len(reached) == 1                           # a lasting finding is written once
    trips = [r for r in _events(tmp_path, "alert") if "strategy loss stop" in r.get("reason", "")]
    assert len(trips) == 2                             # the trip and the trip after the clear


# -- audit finding 4: funding that cannot be refreshed does not blind the stop ------------------------------------
def test_funding_unreadable_for_hours_sizes_to_zero_but_the_stop_still_knows_the_loss(tmp_path):
    world, runtime, btc, i = _holding(tmp_path)
    btc.add_income("BTCUSDT", "-10", btc.now_ms() + 1_000)
    _tick(world, runtime, 601)                                        # a funding total is read: -10
    for fake in world["fakes"].values():
        fake.income_error = True
    for step in range(3):                                             # three hours without a refresh
        _advance(world, runtime, i + step)
    assert runtime.sessions["BTCUSDT"].sleeve_equity() == 0           # the sizing base keeps its two-hour rule
    assert runtime.exits.loss_findings == [] and _loss_issues(tmp_path) == []      # the stop is not blind
    assert _kill(tmp_path)["active"] is False and btc.positions["BTCUSDT"] != 0
    _under_water(btc, "BTCUSDT", 40)                                  # 40 open + the 10 of funding read earlier
    _tick(world, runtime, 61)
    assert _kill(tmp_path)["active"] is True and "strategy loss stop" in _kill(tmp_path)["reason"]
    assert _flat(world)

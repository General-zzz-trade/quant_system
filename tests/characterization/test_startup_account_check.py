"""ADR-0010: execution start-up refuses an account that holds anything outside the strategy ledger, or whose
leverage or margin mode differs from the configuration, and the runtime never changes those settings itself.

Every refusal happens before the runtime is built and after read-only requests only. Driven through the entry
point (``runner.binance_target.main``) with the fake venue; never contacts Binance.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
from decimal import Decimal as D
from pathlib import Path

import pytest

from tests.unit.binance.fake_venue import FakeBinance, make_bars
from trading.adapters.binance import adapter as binance
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.infrastructure import bootstrap
from trading.infrastructure import state_dir as managed

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "binance-testnet-ut"
T0 = 1_794_700_000.0


class Recording(FakeBinance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[tuple[str, str]] = []

    def request_signed(self, *, method, path, params=None):
        self.requests.append((method, path))
        return super().request_signed(method=method, path=path, params=params)


class Started(Exception):
    """Raised by the stand-in runtime: the start-up got past every check."""


@pytest.fixture
def world(tmp_path, monkeypatch):
    from lab.strategy import StrategySpec
    import runner.binance_target as entry
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (("QUANT_ALLOW_UNSIGNED_MODELS", "1"), ("RISK_SCOPE", SCOPE), ("BINANCE_TESTNET", "1"),
                        ("BINANCE_TESTNET_API_KEY", "k"), ("BINANCE_TESTNET_API_SECRET", "s")):
        monkeypatch.setenv(name, value)
    fake = Recording(make_bars(10), clock=lambda: T0)

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)

    def stand_in(**kwargs):
        raise Started()

    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    monkeypatch.setattr(entry, "TargetRuntime", stand_in)
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1"}) == 0
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                "symbols": ["BTCUSDT", "ETHUSDT"], "spec": StrategySpec().record(),
                                "passed": False}))
    return {"fake": fake, "state": state, "main": entry.main,
            "argv": ["--spec", str(spec), "--enable-execution", "--state-dir", str(state)]}


def _refused(world, match):
    with pytest.raises(RuntimeError, match=match):
        world["main"](world["argv"])
    writes = [r for r in world["fake"].requests if r[0] != "GET"]
    assert writes == [], f"a refused start-up sent {writes}"
    assert world["fake"].leverage.keys() <= {"BTCUSDT", "ETHUSDT", "SOXLUSDT"}


def _started(world):
    with pytest.raises(Started):
        world["main"](world["argv"])
    assert [r for r in world["fake"].requests if r[0] != "GET"] == []
    assert not any(path in ("/fapi/v1/leverage", "/fapi/v1/marginType") for _, path in world["fake"].requests)


def _journal(world):
    return SqliteOrderJournal(world["state"] / "orders.sqlite", SCOPE)


# -- refusals ----------------------------------------------------------------------------------------------------
def test_a_manual_position_on_a_strategy_symbol_refuses_the_start(world):
    world["fake"].positions["ETHUSDT"] = D("-1")
    _refused(world, re.escape("ETHUSDT: position -1 is not in the strategy ledger"))


def test_a_manual_position_on_any_other_symbol_refuses_the_start(world):
    world["fake"].positions["SOXLUSDT"] = D("-3")
    world["fake"].price["SOXLUSDT"] = D("30")
    _refused(world, re.escape("SOXLUSDT: position -3 is not in the strategy ledger"))


def test_a_manual_open_order_refuses_the_start(world):
    world["fake"].orders["77"] = {"orderId": 77, "clientOrderId": "web_manual", "symbol": "ETHUSDT", "side": "SELL",
                                  "type": "LIMIT", "origQty": "1", "price": "2000", "status": "NEW"}
    _refused(world, re.escape("ETHUSDT: open order 77 is not in the strategy journal"))


def test_a_manual_conditional_order_refuses_the_start(world):
    world["fake"].algo["web_stop"] = {"algoId": 5, "clientAlgoId": "web_stop", "symbol": "ETHUSDT", "side": "BUY",
                                      "algoStatus": "NEW", "triggerPrice": "1950", "closePosition": True}
    _refused(world, re.escape("ETHUSDT: conditional order web_stop is not in the strategy journal"))


def test_a_leverage_other_than_configured_refuses_the_start_and_is_not_changed(world):
    world["fake"].leverage["BTCUSDT"] = 10
    _refused(world, re.escape("BTCUSDT: exchange leverage 10x, configured 3x"))
    assert world["fake"].leverage == {"BTCUSDT": 10}


def test_isolated_margin_without_a_configured_margin_type_refuses_the_start(world):
    """No configured margin type means cross: the start-up stop check assumes cross in that case."""
    world["fake"].margin_type["BTCUSDT"] = "isolated"
    _refused(world, re.escape("BTCUSDT: exchange margin isolated, configured cross"))
    assert world["fake"].margin_type == {"BTCUSDT": "isolated"}


def test_a_margin_type_other_than_configured_refuses_the_start(world, monkeypatch):
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    world["fake"].margin_type["BTCUSDT"] = "isolated"           # ETHUSDT is still cross
    _refused(world, re.escape("ETHUSDT: exchange margin cross, configured isolated"))


def test_an_account_that_cannot_be_read_refuses_the_start(world):
    world["fake"].position_error = True
    _refused(world, "account could not be read")


def test_every_finding_is_listed(world):
    world["fake"].positions["ETHUSDT"] = D("-1")
    world["fake"].leverage["BTCUSDT"] = 10
    with pytest.raises(RuntimeError) as refusal:
        world["main"](world["argv"])
    assert "ETHUSDT: position -1" in str(refusal.value) and "BTCUSDT: exchange leverage 10x" in str(refusal.value)


# -- what belongs to the strategy --------------------------------------------------------------------------------
def test_a_clean_account_with_the_configured_settings_starts_without_writing_anything(world):
    _started(world)


def test_the_configured_settings_are_checked_not_set(world, monkeypatch):
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "5")
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    world["fake"].leverage.update(BTCUSDT=5, ETHUSDT=5)
    world["fake"].margin_type.update(BTCUSDT="isolated", ETHUSDT="isolated")
    _started(world)


def test_the_strategy_s_own_position_order_and_stop_do_not_refuse_the_start(world):
    journal = _journal(world)
    with sqlite3.connect(world["state"] / "orders.sqlite") as db:
        db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                   (SCOPE, "target.BTCUSDT", json.dumps({"position": {"qty": "0.1"}})))
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner) "
                   "VALUES (?, 'i1', 'BTCUSDT', 'cid-1', '501', 1, 'target.BTCUSDT')", (SCOPE,))
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner) "
                   "VALUES (?, 'i2', 'BTCUSDT', 'cid-2', '', 0, 'target.BTCUSDT')", (SCOPE,))
    journal.ensure_stop_table()
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="stop-1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="SELL", trigger_price="90", created_at=T0)
    fake = world["fake"]
    fake.positions["BTCUSDT"] = D("0.1")
    fake.orders["501"] = {"orderId": 501, "clientOrderId": "cid-1", "symbol": "BTCUSDT", "side": "SELL",
                          "type": "LIMIT", "status": "NEW"}
    fake.orders["502"] = {"orderId": 502, "clientOrderId": "cid-2", "symbol": "BTCUSDT", "side": "SELL",
                          "type": "LIMIT", "status": "NEW"}          # known by client id: its order id is not saved
    fake.algo["stop-1"] = {"algoId": 9, "clientAlgoId": "stop-1", "symbol": "BTCUSDT", "side": "SELL",
                           "algoStatus": "NEW", "triggerPrice": "90", "closePosition": True}
    _started(world)


def test_a_difference_on_a_symbol_with_an_unsettled_strategy_order_is_left_to_recovery(world):
    """After a crash between a fill and its booking the venue is ahead of the ledger; the order recovery at start
    settles it, and reconciliation blocks entries until then. Refusing here would block that recovery."""
    with sqlite3.connect(world["state"] / "orders.sqlite") as db:
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner) "
                   "VALUES (?, 'i3', 'BTCUSDT', 'cid-3', '', 0, 'target.BTCUSDT')", (SCOPE,))
    world["fake"].positions["BTCUSDT"] = D("0.1")
    _started(world)


def test_a_settled_journal_does_not_excuse_a_difference(world):
    with sqlite3.connect(world["state"] / "orders.sqlite") as db:
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner) "
                   "VALUES (?, 'i4', 'BTCUSDT', 'cid-4', '600', 1, 'target.BTCUSDT')", (SCOPE,))
    world["fake"].positions["BTCUSDT"] = D("0.1")
    _refused(world, re.escape("BTCUSDT: position 0.1 is not in the strategy ledger (ledger 0)"))


# -- no code path changes exchange settings ----------------------------------------------------------------------
def test_no_production_code_sets_leverage_or_margin_type():
    files = [*sorted((ROOT / "trading").rglob("*.py")), *sorted((ROOT / "runner").rglob("*.py"))]
    offenders = [p.relative_to(ROOT).as_posix() for p in files
                 if re.search(r"/fapi/v1/(leverage|marginType)\b|configure_leverage|set_leverage|set_margin_type",
                              p.read_text(encoding="utf-8"))]
    assert offenders == []


# -- added after the mutation self-check (survivors N11, N12, N16) -----------------------------------------------
def test_the_strategy_owner_of_a_symbol_counts_even_without_order_rows(world):
    """A booked position belongs to the strategy whether or not its order rows are still in the journal."""
    with sqlite3.connect(world["state"] / "orders.sqlite") as db:
        db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                   (SCOPE, "target.ETHUSDT", json.dumps({"position": {"qty": "-0.5"}})))
    world["fake"].positions["ETHUSDT"] = D("-0.5")
    _started(world)


def test_a_ledger_position_the_venue_does_not_have_refuses_the_start(world):
    with sqlite3.connect(world["state"] / "orders.sqlite") as db:
        db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                   (SCOPE, "target.BTCUSDT", json.dumps({"position": {"qty": "0.1"}})))
    _refused(world, re.escape("BTCUSDT: the venue is flat but the strategy ledger holds 0.1"))


def test_an_unexpected_conditional_order_snapshot_refuses_the_start(world):
    fake = world["fake"]
    signed = fake.request_signed

    def odd(*, method, path, params=None):
        if path == "/fapi/v1/openAlgoOrders":
            fake.requests.append((method, path))
            return {"orders": []}                     # not the documented list: never read as "none"
        return signed(method=method, path=path, params=params)

    fake.request_signed = odd
    _refused(world, "account could not be read")

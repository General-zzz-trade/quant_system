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
                        ("BINANCE_TESTNET_API_KEY", "k"), ("BINANCE_TESTNET_API_SECRET", "s"),
                        # S7 (ADR-0016 points 1-3): execution needs alerts and a heartbeat; Telegram is a fake below.
                        ("TELEGRAM_BOT_TOKEN", "123:tok"), ("TELEGRAM_CHAT_ID", "42"),
                        ("HEARTBEAT_URL", "https://hc.example.invalid/ping/x")):
        monkeypatch.setenv(name, value)
    from trading.infrastructure.cli import target as cli

    class Confirming:                      # Telegram confirms every message; nothing leaves the host
        channel = "telegram"

        def deliver(self, title, detail="", *, timeout_s):
            return True

        def alert(self, level, title, detail=""):
            pass
    monkeypatch.setattr(cli, "make_notifier", lambda secrets: Confirming())
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
    return SqliteOrderJournal(world["state"] / "state.sqlite", SCOPE)


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
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
        db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                   (SCOPE, "target.BTCUSDT", json.dumps({"position": {"qty": "0.1"}})))
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner) "
                   "VALUES (?, 'i1', 'BTCUSDT', 'cid-1', '501', 1, 'target.BTCUSDT')", (SCOPE,))
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner) "
                   "VALUES (?, 'i2', 'BTCUSDT', 'cid-2', '', 0, 'target.BTCUSDT')", (SCOPE,))
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
    with sqlite3.connect(world["state"] / "state.sqlite") as db:     # a real intent has a side and a quantity
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner, side, qty) "
                   "VALUES (?, 'i3', 'BTCUSDT', 'cid-3', '', 0, 'target.BTCUSDT', 'buy', '0.1')", (SCOPE,))
    world["fake"].positions["BTCUSDT"] = D("0.1")
    _started(world)


def test_a_settled_journal_does_not_excuse_a_difference(world):
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
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
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
        db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                   (SCOPE, "target.ETHUSDT", json.dumps({"position": {"qty": "-0.5"}})))
    world["fake"].positions["ETHUSDT"] = D("-0.5")
    _started(world)


def test_a_ledger_position_the_venue_does_not_have_refuses_the_start(world):
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
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


# -- independent audit of ADR-0010 (findings 1-5): in-flight strategy activity is accounted for exactly -----------
def _row(world, intent, symbol, *, side=None, qty=None, settled=0, client_id="", order_id="", reducing=0):
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
        db.execute("INSERT INTO submissions (scope, intent_id, symbol, client_id, order_id, settled, owner, side, qty, "
                   "reducing) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (SCOPE, intent, symbol, client_id, order_id, settled, f"target.{symbol}", side, qty, reducing))


def _checkpoint(world, symbol, qty):
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
        db.execute("INSERT INTO execution_consumers VALUES (?, ?, ?)",
                   (SCOPE, f"target.{symbol}", json.dumps({"position": {"qty": qty}})))


def test_finding_1_a_fill_booked_but_not_yet_consumed_is_the_strategy_s(world):
    """Crash after the fill was stored and the order settled, before the ledger took it (P6)."""
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    fill = {"fill_id": "f1", "intent_id": "i1", "symbol": "BTCUSDT", "order_id": "501", "trade_id": "1",
            "side": "buy", "qty": "0.1", "price": "100", "fee": "0", "fee_asset": "USDT"}
    assert _journal(world).insert_fill(fill_id="f1", intent_id="i1", owner="target.BTCUSDT", payload=fill)
    world["fake"].positions["BTCUSDT"] = D("0.1")
    _started(world)


def test_finding_1_a_stop_that_fired_while_stopped_is_left_to_the_stop_sync(world):
    """The venue is flat because the strategy's own stop closed it; the runtime adopts that fill at start (P7)."""
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    journal = _journal(world)
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="stop-1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="sell", trigger_price="92", created_at=T0)
    journal.set_stop_status("stop-1", "active")
    world["fake"].algo["stop-1"] = {"algoId": 9, "clientAlgoId": "stop-1", "symbol": "BTCUSDT", "side": "SELL",
                                    "algoStatus": "FINISHED", "triggerPrice": "92", "actualOrderId": "777"}
    _started(world)


def test_a_flat_venue_without_a_fired_stop_still_refuses(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    journal = _journal(world)
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="stop-1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="sell", trigger_price="92", created_at=T0)
    journal.set_stop_status("stop-1", "active")
    world["fake"].algo["stop-1"] = {"algoId": 9, "clientAlgoId": "stop-1", "symbol": "BTCUSDT", "side": "SELL",
                                    "algoStatus": "CANCELED", "triggerPrice": "92", "actualOrderId": ""}
    _refused(world, re.escape("BTCUSDT: the venue is flat but the strategy ledger holds 0.1"))


def test_finding_2_a_resting_entry_does_not_excuse_a_position_against_it(world):
    _row(world, "i9", "ETHUSDT", side="buy", qty="0.5", client_id="cid-9", order_id="900")
    world["fake"].orders["900"] = {"orderId": 900, "clientOrderId": "cid-9", "symbol": "ETHUSDT", "side": "BUY",
                                   "type": "LIMIT", "status": "NEW"}
    world["fake"].positions["ETHUSDT"] = D("-1")
    _refused(world, re.escape("ETHUSDT: position -1 is not in the strategy ledger"))


def test_finding_2_a_resting_entry_excuses_at_most_its_own_quantity(world):
    _row(world, "i9", "ETHUSDT", side="buy", qty="0.5", client_id="cid-9", order_id="900")
    world["fake"].positions["ETHUSDT"] = D("0.6")
    _refused(world, re.escape("ETHUSDT: position 0.6 is not in the strategy ledger"))


def test_a_resting_entry_filled_in_part_while_stopped_starts(world):
    _row(world, "i9", "ETHUSDT", side="buy", qty="0.5", client_id="cid-9", order_id="900")
    world["fake"].positions["ETHUSDT"] = D("0.3")
    _started(world)


def test_a_reduction_in_flight_may_have_closed_the_position(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    _row(world, "i2", "BTCUSDT", side="sell", qty="0.1", client_id="cid-2")
    _started(world)                                   # venue already flat: the close went through


def test_finding_2_a_legacy_row_without_side_or_quantity_excuses_nothing(world):
    _row(world, "legacy", "BTCUSDT")
    world["fake"].positions["BTCUSDT"] = D("5")
    _refused(world, re.escape("BTCUSDT: position 5 is not in the strategy ledger"))


def test_finding_4_an_order_id_of_another_symbol_does_not_make_an_order_known(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="77")
    _checkpoint(world, "BTCUSDT", "0.1")
    world["fake"].positions["BTCUSDT"] = D("0.1")
    world["fake"].orders["77"] = {"orderId": 77, "clientOrderId": "web_x", "symbol": "SOLUSDT", "side": "SELL",
                                  "type": "LIMIT", "status": "NEW"}
    _refused(world, re.escape("SOLUSDT: open order 77 is not in the strategy journal"))


def test_finding_4_a_client_id_of_another_symbol_does_not_make_an_order_known(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    world["fake"].positions["BTCUSDT"] = D("0.1")
    world["fake"].orders["88"] = {"orderId": 88, "clientOrderId": "cid-1", "symbol": "ETHUSDT", "side": "SELL",
                                  "type": "LIMIT", "status": "NEW"}
    _refused(world, re.escape("ETHUSDT: open order 88 is not in the strategy journal"))


def test_finding_3_a_manual_order_on_another_symbol_refuses_the_start(world):
    world["fake"].orders["5"] = {"orderId": 5, "clientOrderId": "web", "symbol": "SOXLUSDT", "side": "SELL",
                                 "type": "LIMIT", "status": "NEW"}
    _refused(world, re.escape("SOXLUSDT: open order 5 is not in the strategy journal"))


def test_finding_3_a_manual_conditional_order_on_another_symbol_refuses_the_start(world):
    world["fake"].algo["web"] = {"algoId": 6, "clientAlgoId": "web", "symbol": "SOXLUSDT", "side": "BUY",
                                 "algoStatus": "NEW", "triggerPrice": "40"}
    _refused(world, re.escape("SOXLUSDT: conditional order web is not in the strategy journal"))


def test_finding_3_a_leverage_lower_than_configured_refuses_the_start(world):
    world["fake"].leverage["BTCUSDT"] = 2
    _refused(world, re.escape("BTCUSDT: exchange leverage 2x, configured 3x"))


def test_finding_3_an_unexpected_open_order_snapshot_refuses_the_start(world):
    fake = world["fake"]
    signed = fake.request_signed

    def odd(*, method, path, params=None):
        if path == "/fapi/v1/openOrders":
            fake.requests.append((method, path))
            return {"orders": []}
        return signed(method=method, path=path, params=params)

    fake.request_signed = odd
    _refused(world, "account could not be read")


def test_finding_3_an_unreadable_journal_refuses():
    from trading.application.account_ownership import foreign_holdings

    class Account:
        def get_positions(self, symbol="", *, strict=True):
            return ()

        def get_open_orders(self, symbol="", *, strict=True):
            return ()

        def get_open_algo_orders(self):
            return ()

        def get_settings_snapshot(self):
            return {"BTCUSDT": (3, "cross")}

        def get_algo_order(self, *, client_algo_id):
            return None

    class Journal:
        def orders(self, owner=None, *, unresolved=False):
            raise OSError("disk I/O error")

    findings = foreign_holdings(Account(), Journal(), symbols=("BTCUSDT",), leverage=3, margin="cross")
    assert findings and "strategy journal could not be read" in findings[0]


def test_finding_5_reconciliation_counts_the_same_booked_position_as_the_start_up_check(world):
    """After `manual position` on a symbol the strategy never traded there is a checkpoint but no order row; both
    checks must count it, or the start passes and reconciliation then blocks entries every cycle."""
    from trading.application.reconcile import reconcile_account
    _checkpoint(world, "BTCUSDT", "0.1")
    world["fake"].positions["BTCUSDT"] = D("0.1")
    adapter = binance.BinanceAdapter(binance.BinanceConfig(api_key="k", api_secret="s"), client=world["fake"])
    issues = reconcile_account(adapter, _journal(world), owner="target.BTCUSDT", symbol="BTCUSDT")
    assert not [i for i in issues if "venue qty" in i]
    _started(world)


# -- added after the mutation self-check of the audit fixes (survivors F5, F7, F8, F9, F11, F12, F14) --------------
def _fill(world, fill_id, intent, symbol, side, qty):
    payload = {"fill_id": fill_id, "intent_id": intent, "symbol": symbol, "order_id": "", "trade_id": fill_id,
               "side": side, "qty": qty, "price": "100", "fee": "0", "fee_asset": "USDT"}
    assert _journal(world).insert_fill(fill_id=fill_id, intent_id=intent, owner=f"target.{symbol}", payload=payload)


def test_a_resting_buy_excuses_nothing_on_the_short_side(world):
    _row(world, "i9", "ETHUSDT", side="buy", qty="0.5", client_id="cid-9")
    world["fake"].positions["ETHUSDT"] = D("-0.3")
    _refused(world, re.escape("ETHUSDT: position -0.3 is not in the strategy ledger"))


def test_the_filled_part_of_a_resting_entry_is_not_excused_twice(world):
    _row(world, "i9", "ETHUSDT", side="buy", qty="0.5", client_id="cid-9")
    _fill(world, "f9", "i9", "ETHUSDT", "buy", "0.2")          # stored, not yet consumed: the ledger counts 0.2
    world["fake"].positions["ETHUSDT"] = D("0.6")              # at most 0.2 + the unfilled 0.3
    _refused(world, re.escape("ETHUSDT: position 0.6 is not in the strategy ledger (ledger 0.2"))


def test_a_reduction_on_the_wrong_side_moves_nothing(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    _row(world, "i2", "BTCUSDT", side="buy", qty="0.1", client_id="cid-2", reducing=1)
    _refused(world, re.escape("BTCUSDT: the venue is flat but the strategy ledger holds 0.1"))


def test_a_reduction_never_crosses_zero(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    _row(world, "i2", "BTCUSDT", side="sell", qty="0.3", client_id="cid-2", reducing=1)
    world["fake"].positions["BTCUSDT"] = D("-0.2")
    _refused(world, re.escape("BTCUSDT: position -0.2 is not in the strategy ledger"))


def test_a_stop_id_of_another_symbol_does_not_make_a_conditional_order_known(world):
    journal = _journal(world)
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="stop-1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="sell", trigger_price="92", created_at=T0)
    journal.set_stop_status("stop-1", "canceled")
    world["fake"].algo["stop-1"] = {"algoId": 9, "clientAlgoId": "stop-1", "symbol": "ETHUSDT", "side": "BUY",
                                    "algoStatus": "NEW", "triggerPrice": "2100"}
    _refused(world, re.escape("ETHUSDT: conditional order stop-1 is not in the strategy journal"))


def test_a_stored_fill_that_cannot_be_read_refuses_the_start(world):
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _fill(world, "f1", "i1", "BTCUSDT", "buy", "not-a-number")
    _refused(world, "strategy journal could not be read")


def test_a_stop_whose_state_cannot_be_read_refuses_the_start(world):
    from trading.adapters.binance.rest import BinanceRetryableError
    _row(world, "i1", "BTCUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", "0.1")
    journal = _journal(world)
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="stop-1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="sell", trigger_price="92", created_at=T0)
    journal.set_stop_status("stop-1", "active")               # not among the open conditional orders: look it up
    fake = world["fake"]
    signed = fake.request_signed

    def failing(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder":
            fake.requests.append((method, path))
            raise BinanceRetryableError("HTTP 503: unavailable")
        return signed(method=method, path=path, params=params)

    fake.request_signed = failing
    _refused(world, "account could not be read")


# -- re-check of the audit fixes (mutants Z1-Z7 and Z9 survived): settled rows, fired stops, close-all -------------
def _strategy_long(world, qty="0.1"):
    _row(world, "i1", "BTCUSDT", side="buy", qty=qty, settled=1, client_id="cid-1", order_id="501")
    _checkpoint(world, "BTCUSDT", qty)


def _stop_row(world, status, venue_status=None):
    """A strategy stop on BTCUSDT in journal ``status``; at the venue absent, or with ``venue_status``."""
    journal = _journal(world)
    journal.add_stop(owner="target.BTCUSDT", client_algo_id="stop-1", symbol="BTCUSDT", revision="r", attempt=0,
                     side="sell", trigger_price="92", created_at=T0)
    journal.set_stop_status("stop-1", status)
    if venue_status is not None:
        world["fake"].algo["stop-1"] = {"algoId": 9, "clientAlgoId": "stop-1", "symbol": "BTCUSDT", "side": "SELL",
                                        "algoStatus": venue_status, "triggerPrice": "92", "actualOrderId": "777"}


def test_a_settled_entry_excuses_nothing_even_with_an_unfilled_remainder(world):
    """A maker entry canceled with its remainder unfilled is settled: the venue cannot be ahead because of it."""
    _row(world, "i9", "BTCUSDT", side="buy", qty="0.5", settled=1, client_id="cid-9", order_id="900")
    world["fake"].positions["BTCUSDT"] = D("0.3")
    _refused(world, re.escape("BTCUSDT: position 0.3 is not in the strategy ledger"))


def test_a_stop_the_venue_does_not_know_has_not_fired(world):
    _strategy_long(world)
    _stop_row(world, "active")                                   # the venue answers "order does not exist"
    _refused(world, re.escape("BTCUSDT: the venue is flat but the strategy ledger holds 0.1"))


def test_a_close_all_reduction_in_flight_may_have_flattened_the_position(world):
    _strategy_long(world)
    _row(world, "i2", "BTCUSDT", side="sell", qty="0", client_id="cid-2", reducing=1)   # e.g. an adopted stop order
    _started(world)


@pytest.mark.parametrize("status", ["active", "unknown", "canceling", "awaiting_order"])
def test_a_stop_that_fired_while_stopped_counts_in_every_live_status(world, status):
    _strategy_long(world)
    _stop_row(world, status, venue_status=None if status == "awaiting_order" else "FINISHED")
    _started(world)


@pytest.mark.parametrize("status", ["triggered", "canceled"])
def test_a_stop_row_that_is_no_longer_live_is_not_looked_up(world, status):
    """An adopted (triggered) or canceled stop row explains nothing by itself; its adopted order row would."""
    _strategy_long(world)
    _stop_row(world, status, venue_status="FINISHED")
    _refused(world, re.escape("BTCUSDT: the venue is flat but the strategy ledger holds 0.1"))


def test_a_fired_stop_explains_a_move_to_zero_not_past_it(world):
    _strategy_long(world)
    _stop_row(world, "active", venue_status="FINISHED")
    world["fake"].positions["BTCUSDT"] = D("-0.05")
    _refused(world, re.escape("BTCUSDT: position -0.05 is not in the strategy ledger"))

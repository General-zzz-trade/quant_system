"""Plan D2: every symbol's leverage and margin mode from one read of the whole account.

``GET /fapi/v2/positionRisk`` without a symbol answers a row for every symbol, flat ones included, with its leverage
and margin mode (checked live with account_check 2026-11-19: a flat BTCUSDT row carried leverage 10, isolated). The
whole-account check (ADR-0010 at start, ADR-0011 about once a minute) reads them there instead of one read of weight
5 per symbol. A symbol missing from the answer, or whose row cannot be read, is a finding: fail closed, as an
unreadable account always was. Which symbols must match the configuration does not change (plan D7 changes it).
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.support.state_db import state_db
from tests.unit.binance.fake_venue import FakeBinance
from trading.adapters.binance.adapter import BinanceAdapter, BinanceConfig
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.account_ownership import foreign_holdings
from trading.domain.issues import IssueKind as K

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


class Recording(FakeBinance):
    def __init__(self) -> None:
        super().__init__()
        self.listed, self.calls, self.answer = SYMBOLS, [], None

    def request_signed(self, *, method, path, params=None):
        self.calls.append((path, dict(params or {})))
        if self.answer is not None and path == "/fapi/v2/positionRisk":
            return self.answer
        return super().request_signed(method=method, path=path, params=params)


@pytest.fixture
def venue():
    fake = Recording()
    return fake, BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)


def test_every_symbol_s_settings_come_from_one_read_without_a_symbol(venue):
    fake, adapter = venue
    fake.leverage["BTCUSDT"], fake.margin_type["BTCUSDT"] = 10, "isolated"
    fake.positions["XRPUSDT"] = D("5")                         # a symbol outside the list is answered too
    assert adapter.get_settings_snapshot() == {"BTCUSDT": (10, "isolated"), "ETHUSDT": (3, "cross"),
                                               "SOLUSDT": (3, "cross"), "XRPUSDT": (3, "cross")}
    assert fake.calls == [("/fapi/v2/positionRisk", {})]


ROW = {"symbol": "BTCUSDT", "positionAmt": "0", "leverage": "5", "marginType": "isolated", "positionSide": "BOTH"}


@pytest.mark.parametrize("row", [dict(ROW, marginType="unknown"), dict(ROW, marginType=""),
                                 {k: v for k, v in ROW.items() if k != "marginType"},
                                 {k: v for k, v in ROW.items() if k != "leverage"}, dict(ROW, leverage="x"),
                                 dict(ROW, leverage="0"), dict(ROW, leverage="-3"), dict(ROW, leverage="2.5"),
                                 dict(ROW, leverage=None)])
def test_a_row_whose_settings_cannot_be_read_leaves_its_symbol_out(venue, row):
    fake, adapter = venue
    fake.answer = [row, dict(ROW, symbol="ETHUSDT")]
    assert adapter.get_settings_snapshot() == {"ETHUSDT": (5, "isolated")}


def test_rows_of_one_symbol_count_only_when_they_agree(venue):
    fake, adapter = venue
    fake.answer = [dict(ROW, positionSide="LONG"), dict(ROW, positionSide="SHORT")]
    assert adapter.get_settings_snapshot() == {"BTCUSDT": (5, "isolated")}
    fake.answer = [dict(ROW, positionSide="LONG"), dict(ROW, positionSide="SHORT", leverage="6"),
                   dict(ROW, symbol="ETHUSDT")]
    assert adapter.get_settings_snapshot() == {"ETHUSDT": (5, "isolated")}
    fake.answer = [dict(ROW, marginType="cross"), dict(ROW)]
    assert adapter.get_settings_snapshot() == {}


@pytest.mark.parametrize("answer", [{"code": -1}, "text", [["BTCUSDT"]], [{"positionAmt": "0"}],
                                    [dict(ROW, symbol="")]])
def test_an_answer_that_is_not_a_list_of_rows_raises(venue, answer):
    fake, adapter = venue
    fake.answer = answer
    with pytest.raises(RuntimeError, match="Invalid Binance position snapshot"):
        adapter.get_settings_snapshot()


# -- the whole-account check ------------------------------------------------------------------------------------------
@pytest.fixture
def journal(tmp_path):
    return SqliteOrderJournal(state_db(tmp_path / "state.sqlite"), "binance-testnet-ut")


def test_the_account_check_reads_no_symbol_alone(venue, journal):
    fake, adapter = venue
    assert foreign_holdings(adapter, journal, symbols=SYMBOLS, leverage=3, margin="cross") == []
    assert [params for path, params in fake.calls if path == "/fapi/v2/positionRisk"] == [{}, {}]
    assert all(not params.get("symbol") for _, params in fake.calls)


def test_settings_that_differ_from_the_configuration_are_still_found(venue, journal):
    fake, adapter = venue
    fake.leverage["ETHUSDT"], fake.margin_type["SOLUSDT"] = 2, "isolated"
    fake.leverage["XRPUSDT"] = 7                                # not a strategy symbol: not compared (until D7)
    findings = foreign_holdings(adapter, journal, symbols=SYMBOLS, leverage=3, margin="cross")
    assert [(i.kind, i.symbol) for i in findings] == [(K.ACCOUNT_SETTINGS, "ETHUSDT"), (K.ACCOUNT_SETTINGS, "SOLUSDT")]
    assert "ETHUSDT: exchange leverage 2x, configured 3x" in findings[0]
    assert "SOLUSDT: exchange margin isolated, configured cross" in findings[1]


def test_a_strategy_symbol_missing_from_the_answer_is_a_finding(venue, journal):
    fake, adapter = venue
    real = FakeBinance.request_signed

    def without_sol(self, *, method, path, params=None):
        rows = real(self, method=method, path=path, params=params)
        return [r for r in rows if r["symbol"] != "SOLUSDT"] if path == "/fapi/v2/positionRisk" else rows
    fake.request_signed = without_sol.__get__(fake)
    findings = foreign_holdings(adapter, journal, symbols=SYMBOLS, leverage=3, margin="cross")
    assert [(i.kind, i.symbol) for i in findings] == [(K.ACCOUNT_UNREADABLE, "SOLUSDT")]
    assert "SOLUSDT: leverage and margin mode could not be read" in findings[0]
    fake.margin_type["BTCUSDT"] = "unknown"                    # a row whose margin mode cannot be read: the same
    findings = foreign_holdings(adapter, journal, symbols=SYMBOLS, leverage=3, margin="cross")
    assert [(i.kind, i.symbol) for i in findings] == [(K.ACCOUNT_UNREADABLE, "BTCUSDT"),
                                                      (K.ACCOUNT_UNREADABLE, "SOLUSDT")]


def test_a_settings_read_that_fails_is_an_unreadable_account(venue, journal):
    fake, adapter = venue
    real = FakeBinance.request_signed
    reads = []

    def second_read_odd(self, *, method, path, params=None):       # the positions read passes, the settings' fails
        if path == "/fapi/v2/positionRisk":
            reads.append(path)
            if len(reads) == 2:
                return {"code": -1}
        return real(self, method=method, path=path, params=params)
    fake.request_signed = second_read_odd.__get__(fake)
    findings = foreign_holdings(adapter, journal, symbols=SYMBOLS, leverage=3, margin="cross")
    assert [(i.kind, i.symbol) for i in findings] == [(K.ACCOUNT_UNREADABLE, None)] and len(reads) == 2
    assert "Invalid Binance position snapshot" in findings[0]

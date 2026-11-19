"""Plan D1 (docs/arch/dynamic-universe-plan.md): execution start-up refuses a contract that is not a trading USDT
perpetual, has a delisting scheduled, does not allow the configured exchange leverage, or - with isolated margin - whose
own maintenance margin tier puts the protective stop beyond half the liquidation distance. Read-only, before the
runtime is built; driven through the entry point with the fake venue (``test_startup_account_check``'s world)."""
from __future__ import annotations

import io
import json
import re
from decimal import Decimal as D

import pytest

from tests.characterization.test_startup_account_check import (_checkpoint, _refused, _row,  # noqa: F401
                                                               _started, world)
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses

TIER = {"bracket": 1, "notionalFloor": 0, "cum": 0.0}


def _isolated(world, monkeypatch, stop="0.045"):  # noqa: F811
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "10")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        world["fake"].leverage[symbol] = 10
        world["fake"].margin_type[symbol] = "isolated"
    world["argv"] = world["argv"] + ["--stop-pct", stop]


def test_the_margin_tiers_are_read_at_start_up_and_a_normal_contract_starts(world):  # noqa: F811
    _started(world)
    assert ("GET", "/fapi/v1/leverageBracket") in world["fake"].requests


@pytest.mark.parametrize("fields, words", [
    ({"status": "SETTLING"}, "ETHUSDT: not trading (settling)"),
    ({"status": "TRADING_HALT"}, "ETHUSDT: not trading (halted)"),
    ({"status": "PENDING_TRADING"}, "ETHUSDT: not trading (pending)"),
    ({"contractType": "CURRENT_QUARTER"}, "ETHUSDT: not a USDT perpetual"),
    ({"deliveryDate": 1_795_000_000_000}, "ETHUSDT: delisting scheduled at 2026-11-18"),
])
def test_a_contract_that_is_not_a_trading_usdt_perpetual_refuses_the_start(world, fields, words):  # noqa: F811
    world["fake"].contract["ETHUSDT"] = fields
    _refused(world, re.escape(words))


def test_a_contract_that_does_not_allow_the_configured_leverage_refuses_the_start(world):  # noqa: F811
    world["fake"].brackets["BTCUSDT"] = [{**TIER, "initialLeverage": 2, "notionalCap": 5000, "maintMarginRatio": 0.01}]
    _refused(world, re.escape("BTCUSDT: the exchange allows at most 2x, the configuration asks 3x"))


def test_unreadable_margin_tiers_refuse_the_start(world):  # noqa: F811
    world["fake"].bracket_error = True
    _refused(world, "margin tiers could not be read")


def test_isolated_a_stop_beyond_half_the_liquidation_distance_at_the_contracts_own_tier_refuses(world,  # noqa: F811
                                                                                              monkeypatch):
    _isolated(world, monkeypatch)
    world["fake"].brackets["ETHUSDT"] = [{**TIER, "initialLeverage": 20, "notionalCap": 5000,
                                          "maintMarginRatio": 0.02}]
    _refused(world, re.escape("ETHUSDT: protective stop 4.50% is not inside 50% of the 10x isolated liquidation "
                              "distance at its maintenance margin rate 2.00%"))


def test_isolated_a_stop_inside_half_the_distance_at_each_contracts_tier_starts(world, monkeypatch):  # noqa: F811
    _isolated(world, monkeypatch)
    _started(world)


@pytest.mark.parametrize("capital, multiple, starts", [("999", "1", True), ("1000", "1", False), ("500", "1", True),
                                                       ("500", "2", False)])
def test_the_tier_is_read_for_twice_each_symbols_share_of_the_strategy_capital(world, monkeypatch,  # noqa: F811
                                                                               capital, multiple, starts):
    """1,000 USDT over two symbols at 1x is 500 a symbol; twice that is 1,000, the first notional of the 2 % tier.
    At a strategy leverage of 2, 500 USDT reaches it too."""
    _isolated(world, monkeypatch)
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", capital)
    monkeypatch.setenv("STRATEGY_LEVERAGE", multiple)
    world["fake"].brackets["ETHUSDT"] = [
        {**TIER, "initialLeverage": 50, "notionalCap": 1000, "maintMarginRatio": 0.004},
        {**TIER, "bracket": 2, "initialLeverage": 20, "notionalFloor": 1000, "notionalCap": 5000,
         "maintMarginRatio": 0.02}]
    if starts:
        _started(world)
    else:
        _refused(world, re.escape("ETHUSDT: protective stop 4.50%"))


# -- audit of D1 (2026-11-19) -----------------------------------------------------------------------------------
class Alerts:
    channel = "telegram"

    def __init__(self):
        self.sent = []

    def deliver(self, title, detail="", *, timeout_s):
        self.sent.append((title, detail))
        return True

    def alert(self, level, title, detail=""):
        self.sent.append((title, detail))


def _alerts(monkeypatch):
    from trading.infrastructure.cli import target as cli
    alerts = Alerts()
    monkeypatch.setattr(cli, "make_notifier", lambda secrets: alerts)
    return alerts


def _hold(world, symbol, qty):  # noqa: F811
    _checkpoint(world, symbol, qty)
    world["fake"].positions[symbol] = D(qty)


def _paused(world):  # noqa: F811
    pauses = SqliteEntryPauses(world["state"] / "state.sqlite")
    return {s: pauses.reason(s) for s in ("BTCUSDT", "ETHUSDT")}


def test_a_held_symbol_that_becomes_unfit_pauses_entries_and_the_start_goes_on(world, monkeypatch):  # noqa: F811
    """M1: refusing would leave the held position with no runtime to reduce, flatten or stop it."""
    alerts = _alerts(monkeypatch)
    _hold(world, "ETHUSDT", "-0.5")
    world["fake"].contract["ETHUSDT"] = {"deliveryDate": 1_795_500_000_000}
    _started(world)
    paused = _paused(world)
    assert paused["BTCUSDT"] is None
    assert paused["ETHUSDT"].startswith("contract not fit to trade (plan D1): ETHUSDT: delisting scheduled")
    titles = [title for title, _ in alerts.sent]
    assert "entries paused at start: contracts not fit to trade" in titles
    assert "execution start refused" not in titles


def test_with_a_held_position_unreadable_tiers_pause_every_symbol_instead_of_refusing(world,  # noqa: F811
                                                                                   monkeypatch):
    _alerts(monkeypatch)
    _hold(world, "BTCUSDT", "0.1")
    world["fake"].bracket_error = True
    _started(world)
    assert all(cause and "margin tiers unknown" in cause for cause in _paused(world).values())


def test_without_a_held_position_an_unfit_symbol_still_refuses_and_pauses_nothing(world):  # noqa: F811
    _checkpoint(world, "BTCUSDT", "0")                       # a flat ledger position is not a held one
    world["fake"].contract["ETHUSDT"] = {"status": "SETTLING"}
    _refused(world, re.escape("ETHUSDT: not trading (settling)"))
    assert _paused(world) == {"BTCUSDT": None, "ETHUSDT": None}


def test_a_symbol_missing_from_the_tiers_answer_refuses_the_start(world):  # noqa: F811
    fake = world["fake"]
    signed = fake.request_signed

    def without_eth(*, method, path, params=None):
        rows = signed(method=method, path=path, params=params)
        return [r for r in rows if r["symbol"] != "ETHUSDT"] if path == "/fapi/v1/leverageBracket" else rows
    fake.request_signed = without_eth
    _refused(world, re.escape("ETHUSDT: margin tiers unknown"))


def test_the_contract_check_runs_on_mainnet(tmp_path, monkeypatch):
    """A signed, passed spec on mainnet reaches the contract check (it is not a testnet-only check)."""
    from lab.strategy import StrategySpec
    import runner.binance_target as entry
    from infra.model_signing import sign_file
    from tests.characterization.test_startup_account_check import Recording, Started, T0
    from tests.unit.binance.fake_venue import make_bars
    from trading.adapters.binance import adapter as binance
    from trading.infrastructure import bootstrap
    from trading.infrastructure import state_dir as managed
    scope = "binance-main-ut"
    for name in ("QUANT_ALLOW_UNSIGNED_MODELS", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "BINANCE_TESTNET_API_KEY",
                 "BINANCE_TESTNET_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (("QUANT_MODEL_SIGN_KEY", "test-key"), ("RISK_SCOPE", scope), ("BINANCE_TESTNET", "0"),
                        ("BINANCE_API_KEY", "k"), ("BINANCE_API_SECRET", "s"), ("STRATEGY_CAPITAL_USDT", "150"),
                        ("TELEGRAM_BOT_TOKEN", "123:tok"), ("TELEGRAM_CHAT_ID", "42"),
                        ("HEARTBEAT_URL", "https://hc.example.invalid/ping/x")):
        monkeypatch.setenv(name, value)
    alerts = _alerts(monkeypatch)
    fake = Recording(make_bars(10), clock=lambda: T0)
    fake.contract["ETHUSDT"] = {"status": "SETTLING"}

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)

    def stand_in(**kwargs):
        raise Started()
    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    monkeypatch.setattr(entry, "TargetRuntime", stand_in)
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": scope, "BINANCE_TESTNET": "0"}) == 0
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                "symbols": ["BTCUSDT", "ETHUSDT"], "spec": StrategySpec().record(), "passed": True}))
    sign_file(spec)
    with pytest.raises(RuntimeError, match=re.escape("ETHUSDT: not trading (settling)")):
        entry.main(["--spec", str(spec), "--enable-execution", "--state-dir", str(state)])
    assert [r for r in fake.requests if r[0] != "GET"] == []
    assert ("execution start refused", ) == tuple(t for t, _ in alerts.sent if t == "execution start refused")


# -- re-audit of D1 (2026-11-19) --------------------------------------------------------------------------------
def test_a_fill_not_yet_booked_counts_as_held_so_recovery_and_its_stop_still_run(world, monkeypatch):  # noqa: F811
    """N1: a crash between an entry's fill and its booking leaves an unsettled strategy order and a venue position;
    refusing the start would leave that position without its stop."""
    _alerts(monkeypatch)
    _row(world, "i1", "ETHUSDT", side="buy", qty="0.1", settled=0, client_id="cid-1", order_id="501")
    world["fake"].positions["ETHUSDT"] = D("0.1")
    world["fake"].contract["ETHUSDT"] = {"deliveryDate": 1_795_500_000_000}
    _started(world)
    assert "delisting scheduled" in _paused(world)["ETHUSDT"]


def test_the_contract_pause_lasts_until_a_person_releases_it(world, monkeypatch):  # noqa: F811
    """N2: never a pause that ends by itself (that would raise the risk automatically)."""
    import sqlite3
    _alerts(monkeypatch)
    _hold(world, "ETHUSDT", "-0.5")
    world["fake"].contract["ETHUSDT"] = {"status": "SETTLING"}
    _started(world)
    with sqlite3.connect(world["state"] / "state.sqlite") as db:
        rows = db.execute("SELECT released_at, release_reason FROM entry_pauses WHERE symbol='ETHUSDT'").fetchall()
    assert rows == [(None, None)]
    later = SqliteEntryPauses(world["state"] / "state.sqlite", clock=lambda: 4_102_444_800.0)    # 2100-01-01
    assert later.reason("ETHUSDT").startswith("contract not fit to trade")


def test_a_pause_that_cannot_be_written_refuses_the_start_with_its_alert(world, monkeypatch):  # noqa: F811
    """N3: the refusal reaches the alert channel like every other start refusal."""
    import sqlite3
    alerts = _alerts(monkeypatch)
    _hold(world, "ETHUSDT", "-0.5")
    world["fake"].contract["ETHUSDT"] = {"status": "SETTLING"}

    def broken(self, symbol, cause, triggers=(), *, ends_at=None):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(SqliteEntryPauses, "pause", broken)
    _refused(world, re.escape("ETHUSDT: the entry pause could not be written (OperationalError: database is locked)"))
    assert "execution start refused" in [title for title, _ in alerts.sent]


def test_when_the_account_check_refuses_no_contract_pause_is_left_behind(world, monkeypatch):  # noqa: F811
    """N4: the account check runs first; a refused start writes no pause a person would have to release."""
    _alerts(monkeypatch)
    _hold(world, "ETHUSDT", "-0.5")
    world["fake"].contract["ETHUSDT"] = {"status": "SETTLING"}
    world["fake"].leverage["ETHUSDT"] = 10
    _refused(world, re.escape("ETHUSDT: exchange leverage 10x, configured 3x"))
    assert _paused(world) == {"BTCUSDT": None, "ETHUSDT": None}


def test_an_unsettled_strategy_order_alone_counts_as_held(world, monkeypatch):  # noqa: F811
    _alerts(monkeypatch)
    _row(world, "i1", "ETHUSDT", side="buy", qty="0.1", settled=0, client_id="cid-1", order_id="501")
    world["fake"].contract["ETHUSDT"] = {"status": "SETTLING"}
    _started(world)
    assert "not trading (settling)" in _paused(world)["ETHUSDT"]


def test_a_venue_position_of_a_fill_stored_but_not_yet_in_the_ledger_counts_as_held(world, monkeypatch):  # noqa: F811
    from tests.characterization.test_startup_account_check import _journal
    _alerts(monkeypatch)
    _row(world, "i1", "ETHUSDT", side="buy", qty="0.1", settled=1, client_id="cid-1", order_id="501")
    fill = {"fill_id": "f1", "intent_id": "i1", "symbol": "ETHUSDT", "order_id": "501", "trade_id": "1",
            "side": "buy", "qty": "0.1", "price": "100", "fee": "0", "fee_asset": "USDT"}
    assert _journal(world).insert_fill(fill_id="f1", intent_id="i1", owner="target.ETHUSDT", payload=fill)
    world["fake"].positions["ETHUSDT"] = D("0.1")
    world["fake"].contract["ETHUSDT"] = {"status": "SETTLING"}
    _started(world)
    assert "not trading (settling)" in _paused(world)["ETHUSDT"]

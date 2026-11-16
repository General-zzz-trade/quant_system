"""The strategy loss stop (owner, 2026-11-16): once the strategy has lost 15 % of its capital - realized PnL after
fees, funding and the open positions' unrealized PnL - every position is closed and entries stop until a person
releases them. It replaced "3 losses in a row".

It trips the operator kill switch with flatten, so the close goes through the audited kill path and the kill file
is the durable latch. The runtime only ever sets the switch; a person clears it. The loss stays in the ledger, so
clearing alone trips it again: going on means a new pilot (a new state directory). A loss that cannot be known closes
nothing and blocks every entry.
"""
from __future__ import annotations

import json
from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from trading.adapters.files.kill_switch import KillFile
from trading.application.risk.loss_stop import LOSS_STOP_SECONDS, StrategyLossStop, strategy_unrealized
from trading.domain.issues import Effect
from trading.infrastructure.config import load_config


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def _stop(tmp_path, pnl, *, limit="0.15"):
    clock = Clock()
    kill = KillFile(tmp_path / "kill_switch.json", missing_is_active=False)
    kill.write(active=False, reason="initialized")
    holder = {"pnl": pnl, "calls": 0}

    def read():
        holder["calls"] += 1
        value = holder["pnl"]
        if isinstance(value, Exception):
            raise value
        return value
    return StrategyLossStop(D("300"), D(limit), read, kill, clock=clock), kill, holder, clock


def test_a_loss_short_of_the_limit_does_nothing(tmp_path):
    stop, kill, _, _ = _stop(tmp_path, D("-44.99"))
    assert stop.check() == [] and stop.take_trip() is None
    assert kill.state().active is False


def test_at_15_per_cent_the_kill_switch_is_set_with_flatten_once(tmp_path):
    stop, kill, holder, clock = _stop(tmp_path, D("-45"))
    assert stop.check() == []
    state = kill.state()
    assert state.active is True and state.flatten is True
    assert "strategy loss stop" in state.reason and "45" in state.reason and "15" in state.reason
    assert "new pilot" in state.reason                              # what a person has to know before clearing
    text = stop.take_trip()
    assert text is not None and "45" in text and stop.take_trip() is None        # reported once
    written = (tmp_path / "kill_switch.json").read_text()
    clock.now += LOSS_STOP_SECONDS
    holder["pnl"] = D("-60")
    assert stop.check() == [] and stop.take_trip() is None
    assert (tmp_path / "kill_switch.json").read_text() == written                 # already set: left as it is


def test_a_switch_a_person_set_without_flatten_gets_flatten(tmp_path):
    stop, kill, _, _ = _stop(tmp_path, D("-50"))
    kill.write(active=True, flatten=False, reason="operator: looking at something")
    stop.check()
    state = kill.state()
    assert state.active and state.flatten and "strategy loss stop" in state.reason


def test_after_a_person_clears_the_switch_the_same_loss_sets_it_again(tmp_path):
    stop, kill, _, clock = _stop(tmp_path, D("-50"))
    stop.check()
    kill.write(active=False, reason="cleared by a person")
    clock.now += LOSS_STOP_SECONDS
    stop.check()
    assert kill.state().active and kill.state().flatten and stop.take_trip() is not None


@pytest.mark.parametrize("unknown", [OSError("ledger locked"), D("nan"), D("inf"), None])
def test_an_unknown_loss_closes_nothing_and_blocks_every_entry(tmp_path, unknown):
    stop, kill, holder, _ = _stop(tmp_path, unknown)
    (issue,) = stop.check()
    assert Effect.BLOCK_ALL_ENTRIES in issue.effects and "strategy loss" in issue
    assert kill.state().active is False and stop.take_trip() is None
    holder["pnl"] = D("-10")
    assert stop.check() == []                                       # asked again at once while unknown


def test_a_switch_that_cannot_be_set_blocks_every_entry_and_is_tried_again(tmp_path):
    stop, kill, _, _ = _stop(tmp_path, D("-50"))
    real, kill.trip = kill.trip, lambda reason: (_ for _ in ()).throw(OSError("read-only file system"))
    (issue,) = stop.check()
    assert Effect.BLOCK_ALL_ENTRIES in issue.effects and "kill switch" in issue
    assert stop.take_trip() is None
    kill.trip = real
    assert stop.check() == [] and kill.state().flatten is True     # no minute's wait


def test_the_loss_is_read_at_most_once_a_minute_while_it_is_known(tmp_path):
    stop, _, holder, clock = _stop(tmp_path, D("-10"))
    stop.check()
    clock.now += LOSS_STOP_SECONDS - 1
    stop.check()
    assert holder["calls"] == 1
    clock.now += 1
    stop.check()
    assert holder["calls"] == 2


def test_the_kill_file_trip_is_active_with_flatten(tmp_path):
    kill = KillFile(tmp_path / "k.json", missing_is_active=True)
    kill.trip("strategy loss stop: test")
    data = json.loads((tmp_path / "k.json").read_text())
    assert (data["active"], data["flatten"], data["reason"]) == (True, True, "strategy loss stop: test")


def _position(symbol, qty, entry, mark):
    return NS(symbol=symbol, qty=D(qty), entry_price=None if entry is None else D(entry),
              mark_price=None if mark is None else D(mark))


def test_unrealized_pnl_is_the_strategy_s_positions_at_the_venue_s_mark():
    positions = [_position("BTCUSDT", "0.01", "50000", "47000"), _position("ETHUSDT", "-2", "3000", "2900"),
                 _position("SOLUSDT", "5", "100", "1")]                         # not a strategy symbol
    venue = {p.symbol: p for p in positions}.get
    ledger = [(p.symbol, p.qty, p.entry_price) for p in positions[:2]]          # the strategy's ledger: BTC and ETH
    assert strategy_unrealized(ledger, venue) == D("-30") + D("200")
    assert strategy_unrealized([], venue) == 0


@pytest.mark.parametrize("entry,mark", [(None, "47000"), ("50000", None), ("0", "47000"), ("50000", "0"),
                                        ("50000", "nan")])
def test_a_position_without_a_usable_entry_or_mark_price_is_unknown(entry, mark):
    with pytest.raises(RuntimeError, match="BTCUSDT"):
        strategy_unrealized([("BTCUSDT", D("0.01"), None if entry is None else D(entry))],
                            lambda symbol: _position(symbol, "0.01", entry, mark))


ENV = {"RISK_SCOPE": "binance-ut", "BINANCE_TESTNET": "0", "TELEGRAM_BOT_TOKEN": "1:t", "TELEGRAM_CHAT_ID": "1",
       "HEARTBEAT_URL": "https://hc.example.invalid/x", "STRATEGY_CAPITAL_USDT": "300"}


def _load(**over):
    env = {k: v for k, v in dict(ENV, **over).items() if v is not None}
    return load_config(env, execution=True, stop_pct=D("0.08"))[0]


def test_the_limit_is_15_per_cent_with_a_capital_and_a_person_can_only_tighten_it():
    assert _load().strategy_loss_stop == D("0.15")
    assert json.loads(json.dumps(_load().record()))["strategy_loss_stop"] == "0.15"
    assert _load(STRATEGY_LOSS_STOP_PCT="10").strategy_loss_stop == D("0.10")
    assert _load(STRATEGY_LOSS_STOP_PCT=" ").strategy_loss_stop == D("0.15")
    for raw in ("15.01", "20", "0", "-5", "nan", "ten"):
        with pytest.raises(RuntimeError, match="STRATEGY_LOSS_STOP_PCT"):
            _load(STRATEGY_LOSS_STOP_PCT=raw)


def test_without_a_capital_there_is_no_loss_stop_and_setting_one_is_refused():
    testnet = dict(BINANCE_TESTNET="1", RISK_SCOPE="binance-testnet-ut", STRATEGY_CAPITAL_USDT=None)
    assert _load(**testnet).strategy_loss_stop is None
    with pytest.raises(RuntimeError, match="STRATEGY_CAPITAL_USDT"):
        _load(**testnet, STRATEGY_LOSS_STOP_PCT="10")
    paper = load_config({"STRATEGY_CAPITAL_USDT": "300", "STRATEGY_LOSS_STOP_PCT": "10"}, execution=False,
                        stop_pct=D("0.08"))[0]
    assert paper.strategy_loss_stop is None

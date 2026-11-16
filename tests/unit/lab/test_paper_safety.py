"""Paper safety (ADR-0013 "state unknown and failure direction", audit M6, S10 acceptance).

A paper run - the runtime as the tests build it, and the entry point's ``main`` - constructs no credentialed adapter,
sends no signed request and reads no account-related environment variable. The one environment read it may make is
BINANCE_TESTNET: the spec loader reads it to refuse an unsigned spec on mainnet (it can only make paper stricter;
S10 decision 3), and ``main`` reads it once in the configuration.

Market data fully down: the kill switch still flattens and the stops are still kept, in paper as in execution.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal as D

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.binance.fake_venue import FakeBinance, make_bars
from tests.unit.lab.test_target_runtime import _advance, _world
from tests.unit.lab.test_target_runtime_resilience import _outage
from trading.adapters.binance import adapter as binance
from trading.adapters.simulated.exchange import SimulatedExchange

ACCOUNT_ENV_PREFIXES = ("BINANCE_", "TELEGRAM_")
ACCOUNT_ENV_NAMES = {"RISK_SCOPE", "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"}
ALLOWED_ACCOUNT_ENV = {"BINANCE_TESTNET"}


@pytest.fixture(autouse=True)
def _dev_signing(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "RISK_SCOPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")


def _spec(execution="market", ttl=1):
    return StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution=execution,
                        ttl_bars=ttl)


def _paper(world, state_dir):
    return TargetRuntime(spec_path=world["spec_path"], state_dir=state_dir, mode="paper", market=world["adapter"],
                         clock=lambda: world["clock"]["now"])


def _run(world, runtime, bars):
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + bars):
        _advance(world, runtime, i)


class Recording(dict):
    """os.environ that records every name read."""

    def __init__(self, base, read):
        super().__init__(base)
        self.read = read

    def __getitem__(self, key):
        self.read.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.append(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.append(key)
        return super().__contains__(key)


def _account_reads(read):
    return sorted({k for k in read if k.startswith(ACCOUNT_ENV_PREFIXES) or k in ACCOUNT_ENV_NAMES}
                  - ALLOWED_ACCOUNT_ENV)


def test_paper_constructs_no_credentialed_adapter_and_sends_no_signed_request(tmp_path, monkeypatch):
    world = _world(tmp_path, _spec("maker", 4))            # the test's market-data venue is built before watching
    built, signed = [], []
    original = binance.BinanceAdapter.__init__

    def recording(self, config, *, client=None):
        built.append(config)
        original(self, config, client=client)
    monkeypatch.setattr(binance.BinanceAdapter, "__init__", recording)
    client = world["adapter"]._client
    sign = client.request_signed

    def counting(*, method, path, params=None):
        signed.append(path)
        return sign(method=method, path=path, params=params)
    client.request_signed = counting

    runtime = _paper(world, tmp_path / "paper")
    _run(world, runtime, 120)
    assert sum(len(s.sleeve.broker.fills) for s in runtime.sessions.values()) > 0   # it traded, on the venue
    assert built == [] and signed == []
    assert isinstance(runtime.account, SimulatedExchange)


def test_paper_reads_no_account_environment_variable(tmp_path, monkeypatch):
    world = _world(tmp_path, _spec())
    read: list[str] = []
    for name in ("BINANCE_API_KEY", "BINANCE_TESTNET_API_KEY", "TELEGRAM_BOT_TOKEN", "RISK_SCOPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "BINANCE_MAX_ORDER_NOTIONAL", "BINANCE_EXCHANGE_LEVERAGE"):
        monkeypatch.setenv(name, "must-not-be-read")
    monkeypatch.setattr(os, "environ", Recording(os.environ, read))
    runtime = _paper(world, tmp_path / "paper")
    _run(world, runtime, 30)
    assert read                                            # the recorder does see reads (the spec signature switch)
    assert _account_reads(read) == []


def test_the_paper_entry_point_builds_only_the_public_market_adapter(tmp_path, monkeypatch):
    from runner.binance_target import main
    from trading.infrastructure import bootstrap
    spec = _spec()
    rows = make_bars(spec.window_hours() + 60)
    fake = FakeBinance(rows, clock=lambda: (rows[-1]["start"] + 3_600_000 + 5_000) / 1000)
    built, signed = [], []

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            built.append(config)
            super().__init__(config, client=fake)
    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    sign = fake.request_signed

    def counting(*, method, path, params=None):
        signed.append(path)
        return sign(method=method, path=path, params=params)
    fake.request_signed = counting
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                     "symbols": ["BTCUSDT", "ETHUSDT"], "spec": spec.record(), "passed": False}))
    read: list[str] = []
    monkeypatch.setenv("BINANCE_API_KEY", "must-not-be-read")
    monkeypatch.setattr(os, "environ", Recording(os.environ, read))
    main(["--spec", str(spec_path), "--state-dir", str(tmp_path / "paper"), "--max-cycles", "2"])
    assert [(c.api_key, c.api_secret) for c in built] == [("", "")]
    assert signed == [] and _account_reads(read) == []
    status = json.loads((tmp_path / "paper/target_status.json").read_text())
    assert status["mode"] == "paper"


# -- market data fully down ----------------------------------------------------------------------------------------
def _holding(world, runtime, symbol="BTCUSDT"):
    session = runtime.sessions[symbol]
    i = world["warm"]
    while session.position_qty() == 0 or not session.stops.has_active_stop():
        assert i < world["warm"] + 300, "no position with a stop"
        _advance(world, runtime, i)
        i += 1
    return session, i


def test_paper_keeps_its_stops_and_flattens_on_the_kill_switch_while_market_data_is_down(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    session, i = _holding(world, runtime)
    venue = runtime.account
    held = session.position_qty()
    on = {"value": True}
    _outage(world, on)
    (stop,) = [s for s in venue.get_open_algo_orders() if s.symbol == "BTCUSDT"]
    venue.cancel_algo_order(client_algo_id=stop.client_algo_id)    # the venue lost the stop
    _advance(world, runtime, i)                                      # no bar can be fetched ...
    assert runtime.status()["market_failures"] == 1
    assert [s.symbol for s in venue.get_open_algo_orders()] == ["BTCUSDT"]   # ... and the stop is placed again
    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, i + 1)
    closing = [o for o in venue.get_open_orders("BTCUSDT")]
    assert len(closing) == 1 and closing[0].side == ("sell" if held > 0 else "buy")   # the flatten is at the venue
    on["value"] = False
    _advance(world, runtime, i + 2)                                  # market data back: the next bar fills it
    assert session.position_qty() == 0 and venue.broker("BTCUSDT").position.qty == 0
    assert venue.get_open_algo_orders() == ()


def test_execution_keeps_its_stops_while_market_data_is_down(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")
    world = _world(tmp_path, _spec())
    btc = world["fakes"]["BTCUSDT"]
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                            testnet=True)
    runtime.warmup()
    session, i = _holding(world, runtime)
    _outage(world, {"value": True})
    for algo in btc.algo.values():
        if algo["algoStatus"] == "NEW":
            algo["algoStatus"] = "CANCELED"                          # the venue lost the stop
    _advance(world, runtime, i)
    assert runtime.status()["market_failures"] == 1
    assert [a for a in btc.algo.values() if a["algoStatus"] == "NEW"]
    assert session.position_qty() == btc.positions["BTCUSDT"] != D(0)

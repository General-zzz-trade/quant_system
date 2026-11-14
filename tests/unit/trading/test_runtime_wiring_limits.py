"""Limits and guards as the execution runtime is really wired. Each test closes a gap the M2 audit found by
injecting a defect that no test caught (docs/arch/execution-log.md, audit after step 10b)."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from execution.safety.limits import OrderLimitsConfig
from runner.binance_target import TargetRuntime, main
from tests.characterization.test_runtime_decisions import MARKET, _events, _start, _until_position
from tests.unit.binance.fake_venue import HOUR, FakeBinance, make_bars
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.binance.rest import BinanceRateLimitError, BinanceRetryableError
from trading.application.execution.executor import OwnerExecutor
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure import bootstrap


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET",
                 "BINANCE_MAX_LEVERAGE", "BINANCE_MAX_ORDER_NOTIONAL", "BINANCE_MAX_DAILY_NOTIONAL",
                 "BINANCE_MAX_DAILY_ORDERS", "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _limiter(runtime):
    return runtime.sessions["BTCUSDT"].gate.inner._limiter


# -- limits a person sets ---------------------------------------------------------------------------------
def test_a_leverage_setting_below_the_hard_cap_is_the_cap_that_applies(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_MAX_LEVERAGE", "1.5")
    _, runtime = _start(tmp_path)
    assert runtime.sessions["BTCUSDT"].gate.inner._portfolio_reader.max_leverage == D("1.5")


def test_the_order_limits_reach_the_limiter_each_in_its_own_place(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_MAX_ORDER_NOTIONAL", "300")
    monkeypatch.setenv("BINANCE_MAX_DAILY_NOTIONAL", "900")
    monkeypatch.setenv("BINANCE_MAX_DAILY_ORDERS", "7")
    _, runtime = _start(tmp_path)
    limiter = _limiter(runtime)
    assert limiter._cfg == OrderLimitsConfig(max_order_notional=D("300"), max_daily_notional=D("900"),
                                             max_daily_orders=7, max_orders_per_second=1.0)
    assert limiter is runtime.sessions["ETHUSDT"].gate.inner._limiter        # one limiter for the account
    refused = limiter.check_order(qty=D("1"), price=D("301"))
    assert refused.allowed is False and refused.violated_rule == "max_order_notional"
    assert limiter.check_order(qty=D("1"), price=D("300")).allowed is True


def test_the_default_order_limits_are_unchanged(tmp_path):
    _, runtime = _start(tmp_path)
    assert _limiter(runtime)._cfg == OrderLimitsConfig(max_order_notional=D("5000"), max_daily_notional=D("50000"),
                                                       max_daily_orders=50, max_orders_per_second=1.0)


def test_the_stop_distance_given_at_start_up_is_the_distance_of_the_exchange_stop(tmp_path):
    world = _world(tmp_path, StrategySpec(**MARKET, execution="market"))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                            testnet=True, stop_pct=D("0.05"))
    runtime.warmup()
    i = _until_position(world, runtime)
    _advance(world, runtime, i)
    session, btc = runtime.sessions["BTCUSDT"], world["fakes"]["BTCUSDT"]
    assert session.stops.stop_pct == D("0.05")
    qty, entry = session.consumer.position.qty, session.consumer.position.entry_price
    (stop,) = [a for a in btc.algo.values() if a["algoStatus"] == "NEW"]
    expected = entry * (D("0.95") if qty > 0 else D("1.05"))
    assert abs(D(stop["triggerPrice"]) - expected) <= D("0.1")                # floored to the 0.1 price tick


# -- kill switch, readiness, catch-up -----------------------------------------------------------------------
def test_a_kill_switch_without_flatten_blocks_entries_and_leaves_the_position_alone(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    _advance(world, runtime, i)
    btc = world["fakes"]["BTCUSDT"]
    position, posts = btc.positions["BTCUSDT"], len(btc.posts)
    runtime.kill.write(active=True, reason="pause only", flatten=False)
    for k in range(1, 4):
        _advance(world, runtime, i + k)
    assert not [r for r in _events(tmp_path, "intent") if r["reason"] == "kill_switch"]
    assert btc.positions["BTCUSDT"] != 0 and (btc.positions["BTCUSDT"] > 0) == (position > 0)
    assert all(p.get("reduceOnly") in (True, "true") for p in btc.posts[posts:])   # nothing new was opened


def test_execution_sends_nothing_while_the_target_is_not_ready(tmp_path):
    world, runtime = _start(tmp_path)
    for session in runtime.sessions.values():
        session._target = lambda: (0.9, False)                                # a large target that is not ready
    for i in range(world["warm"], world["warm"] + 5):
        _advance(world, runtime, i)
    assert all(f.posts == [] for f in world["fakes"].values())
    assert not (tmp_path / "exec/target_events.jsonl").exists() or _events(tmp_path, "intent") == []
    assert runtime.sessions["BTCUSDT"].last_target == (0.9, False)


def test_after_an_outage_execution_decides_on_the_newest_bar_only(tmp_path):
    world, runtime = _start(tmp_path)
    i = world["warm"]
    _advance(world, runtime, i)
    calls = []
    for symbol, session in runtime.sessions.items():
        original = session.process_bar

        def spy(bar, *, live, now_ms, _original=original, _symbol=symbol):
            calls.append((_symbol, bar.open_time, live))
            return _original(bar, live=live, now_ms=now_ms)

        session.process_bar = spy
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][i + 4]["start"] + HOUR + 5_000) / 1000   # four bars missed
    runtime.cycle()
    btc = [(t, live) for symbol, t, live in calls if symbol == "BTCUSDT"]
    assert len(btc) == 4 and [live for _, live in btc] == [False, False, False, True]
    assert btc[-1][0] == world["rows"]["BTCUSDT"][i + 4]["start"]


def test_a_network_error_below_the_exchange_client_is_a_market_data_failure_too(tmp_path):
    world, runtime = _start(tmp_path)
    for fake in world["fakes"].values():
        def failing(*, method, path, params=None):
            raise OSError("connection reset")
        fake.request_public = failing
    _advance(world, runtime, world["warm"])                                   # must not end the process
    assert runtime.status()["market_failures"] == 1
    assert "market data unavailable: OSError" in _events(tmp_path, "issue")[-1]["issue"]


# -- start-up order (R11) -----------------------------------------------------------------------------------
def _main_world(tmp_path, monkeypatch):
    fake = FakeBinance(make_bars(10), clock=lambda: 1_800_000_000.0)

    class Adapter(BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)

    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    for name in ("BINANCE_TESTNET", "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET"):
        monkeypatch.setenv(name, "1")
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                     "symbols": ["BTCUSDT", "ETHUSDT"], "spec": StrategySpec().record(),
                                     "passed": False}))
    monkeypatch.chdir(tmp_path)
    return fake, ["--spec", str(spec_path), "--enable-execution", "--state-dir", str(tmp_path / "state")]


def _nothing_written(fake, tmp_path):
    return fake.leverage == {} and fake.margin_type == {} and fake.posts == [] and not (tmp_path / "state").exists()


def test_a_wrong_risk_scope_refuses_before_any_account_setting_is_written(tmp_path, monkeypatch):
    fake, argv = _main_world(tmp_path, monkeypatch)
    monkeypatch.setenv("RISK_SCOPE", "binance-main-ut")                       # testnet run, scope without 'testnet'
    with pytest.raises(RuntimeError, match="RISK_SCOPE"):
        main(argv)
    assert _nothing_written(fake, tmp_path)


def test_a_stop_beyond_the_liquidation_distance_refuses_before_any_account_setting_is_written(tmp_path, monkeypatch):
    fake, argv = _main_world(tmp_path, monkeypatch)
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "10")
    monkeypatch.setenv("BINANCE_MARGIN_TYPE", "ISOLATED")
    with pytest.raises(RuntimeError, match="liquidation distance"):
        main(argv)                                                            # default --stop-pct 0.08
    assert _nothing_written(fake, tmp_path)


class _RecordingAdapter:
    instances: list = []

    def __init__(self, config):
        self.config, self.connected_with, self.blockers = config, None, []
        _RecordingAdapter.instances.append(self)

    def connect(self, **kw):
        self.connected_with = kw
        return True

    def execution_blockers(self):
        return list(type(self).pending_blockers)


def test_the_account_is_connected_with_the_configured_leverage_and_blockers_stop_the_start(monkeypatch):
    monkeypatch.setattr(bootstrap, "BinanceAdapter", _RecordingAdapter)
    for name, value in (("BINANCE_TESTNET_API_KEY", "k"), ("BINANCE_TESTNET_API_SECRET", "s"),
                        ("BINANCE_EXCHANGE_LEVERAGE", "5"), ("BINANCE_MARGIN_TYPE", "ISOLATED")):
        monkeypatch.setenv(name, value)
    _RecordingAdapter.instances, _RecordingAdapter.pending_blockers = [], []
    market, account = bootstrap.build_adapters(execution=True, testnet=True, symbols=["BTCUSDT", "ETHUSDT"])
    assert account.connected_with == {"symbols": ("BTCUSDT", "ETHUSDT"), "configure_leverage": True, "leverage": 5,
                                      "margin_type": "ISOLATED"}
    assert market.connected_with is None and market.config.testnet is False    # public data: mainnet, no keys
    _RecordingAdapter.pending_blockers = ["hedge mode is enabled"]
    with pytest.raises(RuntimeError, match="Execution blocked: hedge mode is enabled"):
        bootstrap.build_adapters(execution=True, testnet=True, symbols=["BTCUSDT"])
    assert bootstrap.build_adapters(execution=False, testnet=True, symbols=["BTCUSDT"])[1] is None


# -- smaller guards -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("scope", ["", " binance-testnet-ut", "binance-testnet-ut "])
def test_the_executor_refuses_a_blank_or_padded_scope(scope):
    with pytest.raises(ValueError, match="explicit account/environment scope"):
        OwnerExecutor(service=None, scope=scope)


@pytest.mark.parametrize("error", [BinanceRateLimitError("HTTP 429: too many requests"),
                                   BinanceRetryableError("HTTP 503: unavailable"), OSError("reset")])
def test_a_stop_refused_or_lost_for_any_other_reason_is_never_reported_as_breached(error):
    fake = FakeBinance(make_bars(10), clock=lambda: 1_800_000_000.0)
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    original = fake.request_signed

    def failing(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            raise error
        return original(method=method, path=path, params=params)

    fake.request_signed = failing
    result = adapter.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1")
    assert result.get("breached", False) is False
    assert result["status"] == ("rejected" if isinstance(error, BinanceRateLimitError) else "error")


"""The strategy capital cap in the running runtime and at the entry point (owner, 2026-11-16).

- With STRATEGY_CAPITAL_USDT set, an execution runtime sizes from the cap (plus the strategy's own realized PnL),
  not from the whole account: the fake account holds 10,000 USDT, the cap is 300.
- Mainnet execution refuses to start without a cap, before any account request.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.characterization.test_exit_paths import _tick, _until
from tests.unit.lab.test_target_runtime import _advance
from tests.support.state_db import state_dir_db
from tests.unit.lab.test_target_runtime import _world


def _execution(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    world = _world(tmp_path, spec)
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    return world, runtime


def _first_entry_notional(tmp_path):
    world, runtime = _execution(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    return abs(btc.positions["BTCUSDT"]) * btc.mark("BTCUSDT"), runtime


def test_the_runtime_sizes_from_the_cap_not_the_whole_account(tmp_path, monkeypatch):
    whole, _ = _first_entry_notional(tmp_path / "whole")
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    capped, runtime = _first_entry_notional(tmp_path / "capped")
    assert runtime.config.strategy_capital == D("300")
    assert capped <= D("300") / 2 * D("1.01")             # half of the cap per symbol, at most a full weight
    assert whole > 10 * capped                             # the same signal on the whole 10,000 account


def test_mainnet_execution_without_a_cap_is_refused_before_any_account_request(tmp_path, monkeypatch):
    from tests.characterization.test_startup_account_check import Recording
    from tests.unit.binance.fake_venue import make_bars
    from trading.adapters.binance import adapter as binance
    from trading.infrastructure import bootstrap
    from trading.infrastructure.cli import target as cli
    import runner.binance_target as entry

    fake = Recording(make_bars(10), clock=lambda: 1_794_700_000.0)

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)
    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    monkeypatch.setattr(cli, "preflight", lambda *a, **k: {"symbols": ["BTCUSDT"], "passed": True})
    for name, value in (("BINANCE_TESTNET", "0"), ("RISK_SCOPE", "binance-ut"), ("BINANCE_API_KEY", "k"),
                        ("BINANCE_API_SECRET", "s"), ("TELEGRAM_BOT_TOKEN", "1:t"), ("TELEGRAM_CHAT_ID", "1"),
                        ("HEARTBEAT_URL", "https://hc.example.invalid/x")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STRATEGY_CAPITAL_USDT", raising=False)
    with pytest.raises(RuntimeError, match="STRATEGY_CAPITAL_USDT"):
        entry.main(["--spec", str(tmp_path / "spec.json"), "--enable-execution", "--state-dir", str(tmp_path / "s")])
    assert fake.requests == []


# -- funding in the base (tech lead 2026-11-16) -------------------------------------------------------------------
def test_funding_since_the_first_fill_moves_the_capped_base(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    world, runtime = _execution(tmp_path)
    btc, eth = world["fakes"]["BTCUSDT"], world["fakes"]["ETHUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    session = runtime.sessions["BTCUSDT"]
    before = session.sleeve_equity()
    now = btc.now_ms()
    btc.add_income("BTCUSDT", "-40", now + 1_000)                  # after the first fill: the strategy's
    btc.add_income("BTCUSDT", "-500", now - 24 * 3_600_000)        # before it: not the strategy's
    btc.add_income("BTCUSDT", "-700", now + 2_000, asset="BNB")    # not USDT
    _tick(world, runtime, 601)                                      # the next refresh is due
    assert session.sleeve_equity() == before - D("40") / 2         # half of the base per symbol
    assert btc.income_requests and {r["incomeType"] for r in btc.income_requests} == {"FUNDING_FEE"}
    eth_traded = any(t["symbol"] == "ETHUSDT" for t in eth.trades)
    assert bool(eth.income_requests) == eth_traded                  # a symbol never traded is not asked


def test_funding_that_cannot_be_read_sizes_to_zero_after_two_hours_and_the_runtime_runs_on(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    world, runtime = _execution(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    session = runtime.sessions["BTCUSDT"]
    for fake in world["fakes"].values():
        fake.income_error = True
    for step in range(3):                                           # 1 h, 2 h: the last total; 3 h: unknown
        _advance(world, runtime, i + step)
        assert (session.sleeve_equity() > 0) == (step < 2)
    assert D(runtime.status()["symbols"]["BTCUSDT"]["sleeve_equity"]) == 0
    for fake in world["fakes"].values():
        fake.income_error = False
    _advance(world, runtime, i + 3)
    assert session.sleeve_equity() > 0


def test_a_cap_without_the_account_s_funding_income_is_refused_when_the_sessions_are_built(tmp_path):
    from trading.application.equity import EquitySource
    from trading.infrastructure import target
    from trading.infrastructure.config import risk_limits
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
    world = _world(tmp_path, spec)
    state_dir_db(tmp_path / "exec")
    inputs = target.ExecutionInputs(account=world["adapter"], scope="binance-testnet-ut", stop_pct=D("0.08"),
                                    limits=risk_limits({}), kill=SimpleNamespace(state=None), leverage=3, margin=None,
                                    strategy_capital=D("300"))
    with pytest.raises(ValueError, match="funding income"):
        target.execution_sessions(spec, ["BTCUSDT", "ETHUSDT"], state_dir=tmp_path / "exec", journal=None,
                                  equity=EquitySource(lambda: 0.0), inputs=inputs)


def test_without_a_cap_no_funding_is_asked(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATEGY_CAPITAL_USDT", raising=False)
    world, runtime = _execution(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _tick(world, runtime, 601)
    assert all(fake.income_requests == [] for fake in world["fakes"].values())


def test_the_runtime_keeps_the_funding_rows_in_the_state_database(tmp_path, monkeypatch):
    """Tech lead 2026-11-17: the rows read are kept (version 2), so a later refresh reads only a short window and a
    restart's loss stop knows the total before any read."""
    import sqlite3
    from trading.adapters.sqlite import schema
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    world, runtime = _execution(tmp_path)
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    now = btc.now_ms()
    btc.add_income("BTCUSDT", "-40", now + 1_000)
    for _ in range(27):                                            # 27 hours on: past the 24-hour overlap
        _tick(world, runtime, 3600)
    with sqlite3.connect(tmp_path / "exec" / schema.STATE_DB) as db:
        kept = db.execute("SELECT symbol, amount FROM funding_rows").fetchall()
        scopes = {r[0] for r in db.execute("SELECT scope FROM funding_rows")}
        span = db.execute("SELECT from_ms, through_ms FROM funding_coverage WHERE symbol='BTCUSDT'").fetchone()
    assert ("BTCUSDT", "-40") in kept and span is not None
    assert scopes == {runtime.config.scope}                        # audit of #147 (mutant A12): the runtime's scope
    first, last = btc.income_requests[0], btc.income_requests[-1]
    assert last["startTime"] > first["startTime"]                  # later reads start near the kept span's end
    assert last["endTime"] - last["startTime"] <= 25 * 3_600_000   # 24 hours of overlap and one hour since the last

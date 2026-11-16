"""The strategy capital cap in the running runtime and at the entry point (owner, 2026-11-16).

- With STRATEGY_CAPITAL_USDT set, an execution runtime sizes from the cap (plus the strategy's own realized PnL),
  not from the whole account: the fake account holds 10,000 USDT, the cap is 300.
- Mainnet execution refuses to start without a cap, before any account request.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.characterization.test_exit_paths import _until
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

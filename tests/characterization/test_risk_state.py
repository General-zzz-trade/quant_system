"""Risk state, kill file, lease and start-up order of runner.binance_target."""
from __future__ import annotations

import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from execution.safety.limits import OrderLimiter, OrderLimitsConfig
from shared.drawdown_guard import DrawdownGuard
from shared.execution_lease import ExecutionLease
from shared.kill_file import DEFAULT_PATH, KillFile

T0 = 1_794_700_000.0


def _tripped_guard(path: Path, scope: str) -> DrawdownGuard:
    guard = DrawdownGuard(path, scope, daily_pct=5, weekly_pct=12)
    guard.observe(1000.0, now=T0)
    allowed, reason = guard.observe(940.0, now=T0 + 60)
    assert not allowed and "drawdown latched" in reason
    return guard


def test_drawdown_latch_survives_a_restart_with_the_same_file_and_scope(tmp_path):
    _tripped_guard(tmp_path / "drawdown.sqlite", "binance-testnet-ut")
    restarted = DrawdownGuard(tmp_path / "drawdown.sqlite", "binance-testnet-ut", daily_pct=5, weekly_pct=12)
    assert restarted.latched(now=T0 + 120)


@pytest.mark.known_defect("R6")
def test_current_drawdown_latch_is_lost_when_the_file_is_missing_or_the_scope_changes(tmp_path):
    path = tmp_path / "drawdown.sqlite"
    _tripped_guard(path, "binance-testnet-ut")
    renamed = DrawdownGuard(path, "binance-testnet-renamed", daily_pct=5, weekly_pct=12)
    assert not renamed.latched(now=T0 + 120)
    assert renamed.observe(940.0, now=T0 + 130) == (True, "")
    path.unlink()
    recreated = DrawdownGuard(path, "binance-testnet-ut", daily_pct=5, weekly_pct=12)
    assert not recreated.latched(now=T0 + 140)
    assert recreated.observe(940.0, now=T0 + 150) == (True, "")


@pytest.mark.known_defect("R12")
def test_current_drawdown_limit_has_no_hard_ceiling(tmp_path):
    assert DrawdownGuard(tmp_path / "d.sqlite", "s", daily_pct=99.9, weekly_pct=99.9).limits == {
        "day": 99.9, "week": 99.9}


@pytest.mark.known_defect("R12")
def test_current_daily_order_count_restarts_from_zero_in_a_new_limiter():
    config = OrderLimitsConfig(max_daily_orders=2, max_orders_per_second=1000.0)
    limiter = OrderLimiter(config)
    assert [limiter.check_order(qty=D("0.001"), price=D("100")).allowed for _ in range(3)] == [True, True, False]
    assert OrderLimiter(config).check_order(qty=D("0.001"), price=D("100")).allowed   # after a restart


def test_kill_file_semantics(tmp_path):
    kill = KillFile(tmp_path / "kill_switch.json")
    assert kill.allow_order(symbol="BTCUSDT") == (True, "")              # missing file: inactive
    kill.write(active=True, reason="test", flatten=True)
    assert kill.state().active and kill.state().flatten
    (tmp_path / "kill_switch.json").write_text("{not json")
    state = kill.state()
    assert state.active and not state.flatten                            # unreadable: active, no flatten


@pytest.mark.known_defect("R5")
def test_current_kill_cli_default_path_differs_from_the_target_runtime_path(tmp_path, monkeypatch):
    import runner.binance_target as target
    seen = {}

    class Stop(Exception):
        pass

    def capture(journal_path, scope):
        seen["state_dir"] = Path(journal_path).parent
        raise Stop

    from lab.strategy import StrategySpec
    monkeypatch.setattr("shared.execution_lease.ExecutionLease", capture)
    monkeypatch.chdir(tmp_path)
    # Since step 10 the spec is verified before anything else, so the run needs a real (unsigned, dev-mode) one.
    Path("spec.json").write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                             "symbols": ["BTCUSDT", "ETHUSDT"], "spec": StrategySpec().record(),
                                             "passed": False}))
    with pytest.raises(Stop):
        target.main(["--spec", "spec.json"])
    assert seen["state_dir"] == Path("data/runtime/binance_target")       # runtime reads <state_dir>/kill_switch.json
    assert DEFAULT_PATH == Path("data/runtime/binance/kill_switch.json")  # the CLI writes here by default
    assert not DEFAULT_PATH.is_absolute()


@pytest.mark.known_defect("R20")
def test_current_lease_allows_the_same_scope_in_two_state_dirs(tmp_path):
    first = ExecutionLease(tmp_path / "a" / "orders.sqlite", "binance-testnet-ut")
    second = ExecutionLease(tmp_path / "b" / "orders.sqlite", "binance-testnet-ut")
    with pytest.raises(RuntimeError, match="already has an execution process"):
        ExecutionLease(tmp_path / "a" / "orders.sqlite", "binance-testnet-ut")
    first.close()
    second.close()


def test_the_mainnet_gate_refuses_before_any_account_setting_is_written(tmp_path, monkeypatch):
    """R11, fixed in migration step 10 (invariant L6): every local start-up check runs before the account is
    connected. Until then this test pinned the defect: leverage was set first, then the gate refused."""
    import execution.adapters.binance as binance
    from infra.model_signing import sign_file
    from lab.strategy import StrategySpec
    from runner.binance_target import main
    from tests.unit.binance.fake_venue import FakeBinance, make_bars

    fake = FakeBinance(make_bars(10), clock=lambda: T0)

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)

    from trading.infrastructure import bootstrap
    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)       # where build_adapters looks it up (step 10)
    monkeypatch.delenv("QUANT_ALLOW_UNSIGNED_MODELS")
    for name, value in (("BINANCE_TESTNET", "0"), ("RISK_SCOPE", "binance-main-ut"), ("BINANCE_API_KEY", "k"),
                        ("BINANCE_API_SECRET", "s"), ("QUANT_MODEL_SIGN_KEY", "test-key")):
        monkeypatch.setenv(name, value)
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target",
                                     "symbols": ["BTCUSDT", "ETHUSDT"], "spec": StrategySpec().record(),
                                     "passed": False}))
    sign_file(spec_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="research gate not passed"):
        main(["--spec", str(spec_path), "--enable-execution", "--state-dir", str(tmp_path / "state")])
    assert fake.leverage == {} and fake.margin_type == {} and fake.posts == []   # nothing was written
    assert not (tmp_path / "state").exists()                               # not even the state directory

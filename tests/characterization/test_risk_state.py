"""Risk state, kill file, lease and start-up order of runner.binance_target."""
from __future__ import annotations

import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from lab.strategy import StrategySpec

from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard as DrawdownGuard
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.files.kill_switch import KillFile
from tests.support.state_db import state_db

T0 = 1_794_700_000.0


def _tripped_guard(path: Path, scope: str) -> DrawdownGuard:
    guard = DrawdownGuard(path, scope, daily_pct=5, weekly_pct=12)
    guard.observe(1000.0, now=T0)
    allowed, reason = guard.observe(940.0, now=T0 + 60)
    assert not allowed and "drawdown latched" in reason
    return guard


def test_drawdown_latch_survives_a_restart_with_the_same_file_and_scope(tmp_path):
    _tripped_guard(state_db(tmp_path / "drawdown.sqlite"), "binance-testnet-ut")
    restarted = DrawdownGuard(tmp_path / "drawdown.sqlite", "binance-testnet-ut", daily_pct=5, weekly_pct=12)
    assert restarted.latched(now=T0 + 120)


def test_execution_refuses_to_start_when_a_store_is_missing_or_the_scope_changed(tmp_path, monkeypatch):
    """R6, fixed in migration step 13c (ADR-0007 1c): execution starts only from a state directory a person
    initialized, for the same scope, with every store present. Until then this test pinned the defect: a
    renamed scope or a deleted drawdown file silently started a fresh, unlatched drawdown store. Since ADR-0009
    the stores never create anything either; the refusal is at start-up, before any is built."""
    import io

    from lab.strategy import StrategySpec
    from runner.binance_target import preflight
    from trading.infrastructure import state_dir as managed
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT"],
                                "spec": StrategySpec().record(), "passed": False}))
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(),
                        env={"RISK_SCOPE": "binance-testnet-ut", "BINANCE_TESTNET": "1"}) == 0
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-renamed")
    with pytest.raises(RuntimeError, match="belongs to scope binance-testnet-ut"):
        preflight(spec, execution=True, testnet=True, stop_pct=D("0.08"), state_dir=state)
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")
    (state / "state.sqlite").unlink()                       # the one state database (ADR-0009)
    with pytest.raises(RuntimeError, match="state.sqlite is missing"):
        preflight(spec, execution=True, testnet=True, stop_pct=D("0.08"), state_dir=state)
    assert not (state / "state.sqlite").exists()


def test_a_drawdown_limit_above_the_hard_ceiling_refuses_to_start():
    """R12 (first half), fixed in migration step 13b (tech lead, 2026-11-15): daily at most 10 %, weekly at
    most 25 %, checked where the target runtime reads them. Until then this test pinned the defect: the
    drawdown guard accepted 99.9 % for both."""
    from trading.infrastructure.settings import risk_limits
    with pytest.raises(RuntimeError, match="drawdown"):
        risk_limits({"MAX_DAILY_DRAWDOWN_PCT": "99.9", "MAX_WEEKLY_DRAWDOWN_PCT": "99.9"})
    assert (risk_limits({"MAX_DAILY_DRAWDOWN_PCT": "10", "MAX_WEEKLY_DRAWDOWN_PCT": "25"}).weekly_drawdown_pct
            == 25.0)


def test_the_daily_order_count_survives_a_restart(tmp_path):
    """R12 (second half), fixed in migration step 13b (ADR-0007 2b): the count is stored per UTC day. Until
    then this test pinned the defect: a new limiter started again from zero."""
    from trading.adapters.sqlite.limit_store import SqliteLimitStore
    from trading.application.risk.order_limiter import OrderLimiter as StoredLimiter
    from trading.domain.risk.order_limits import OrderLimitsConfig as Config
    config = Config(max_daily_orders=2, max_orders_per_second=1000.0)

    def limiter():
        return StoredLimiter(config, store=SqliteLimitStore(state_db(tmp_path / "order_limits.sqlite"), "s"),
                             clock=lambda: 1_800_000_000.0, monotonic=lambda: 0.0)

    first = limiter()
    assert [first.check_order(qty=D("0.001"), price=D("100")).allowed for _ in range(3)] == [True, True, False]
    assert limiter().check_order(qty=D("0.001"), price=D("100")).allowed is False   # after a restart


def test_kill_file_semantics(tmp_path):
    kill = KillFile(tmp_path / "kill_switch.json", missing_is_active=False)
    assert kill.allow_order(symbol="BTCUSDT") == (True, "")              # missing file: inactive
    kill.write(active=True, reason="test", flatten=True)
    assert kill.state().active and kill.state().flatten
    (tmp_path / "kill_switch.json").write_text("{not json")
    state = kill.state()
    assert state.active and not state.flatten                            # unreadable: active, no flatten


def test_the_kill_command_writes_exactly_where_the_runtime_reads(tmp_path):
    """R5, fixed in migration step 13c (ADR-0005 5b): one function gives the kill file's path from the state
    directory, the command requires the state directory, and in an initialized directory a missing kill file
    reads as active. Until then this test pinned the defect: the command's default path was another runtime's
    directory, and a missing file read as inactive."""
    import io

    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import _world
    from trading.infrastructure import kill as kill_cli
    from trading.infrastructure import state_dir as managed
    from trading.infrastructure.state_dir import kill_file_path
    state = tmp_path / "state"
    managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(),
                 env={"RISK_SCOPE": "binance-testnet-ut", "BINANCE_TESTNET": "1"})
    with pytest.raises(SystemExit):
        kill_cli.main(["activate"], stdout=io.StringIO())                     # no default path any more
    world = _world(tmp_path, StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="execution",
                            market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                            testnet=True)
    assert runtime.kill.path == kill_file_path(state)
    kill_cli.main(["--state-dir", str(state), "activate", "--reason", "test"], stdout=io.StringIO())
    assert runtime.kill.state().active is True
    kill_file_path(state).unlink()
    assert runtime.kill.state().active is True


def test_the_same_account_identity_in_two_state_dirs_is_refused(tmp_path):
    """R20, fixed in S6 (ADR-0016 point 5). Until then this test pinned the defect: the lease was taken per state
    directory, so one account could run in two directories. The directory lease stays as it was; the host lock
    by account identity is what refuses the second process."""
    from trading.adapters.files.host_lock import HostLock
    from trading.infrastructure.config import account_identity
    identity = account_identity(testnet=True, api_key="k")
    first = ExecutionLease(tmp_path / "a" / "orders.sqlite", "binance-testnet-ut")
    held = HostLock(identity, holder="the runtime of directory a")
    second = ExecutionLease(tmp_path / "b" / "orders.sqlite", "binance-testnet-ut")   # the directory lease allows it
    with pytest.raises(RuntimeError, match="already has an execution process"):
        HostLock(identity, holder="the runtime of directory b")                      # the host lock does not
    with pytest.raises(RuntimeError, match="already has an execution process"):
        ExecutionLease(tmp_path / "a" / "orders.sqlite", "binance-testnet-ut")
    held.close()
    first.close()
    second.close()


def test_the_mainnet_gate_refuses_before_any_account_setting_is_written(tmp_path, monkeypatch):
    """R11, fixed in migration step 10 (invariant L6): every local start-up check runs before the account is
    connected. Until then this test pinned the defect: leverage was set first, then the gate refused."""
    from trading.adapters.binance import adapter as binance
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

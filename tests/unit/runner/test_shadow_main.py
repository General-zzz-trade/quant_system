"""Native shadow orchestration with synthetic inputs and all networking forbidden."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
import socket
import sys
import time
from types import SimpleNamespace

import pytest

from runner import shadow_main as shadow
from runner.builders import alpha_builder as builder
from runner.run_manifest import load_run_manifest
from runner.runtime_monitor import RuntimeMonitor
from shared.runtime_state import initialize_runtime_state, RuntimeStateLease


HOUR = 3_600_000
INFO = {"zscore_window": 10, "zscore_warmup": 2, "horizon_models": [], "config": {},
        "deadzone": 0.5, "min_hold": 1, "max_hold": 10}


def forbidden(*args, **kwargs):
    raise AssertionError("Shadow touched a private account, execution, notification, or network interface")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    from alpha import model_loader_prod
    from data import oi_cache
    from decision.modules.alpha import AlphaDecisionModule
    from execution.adapters.bybit.adapter import BybitAdapter
    from monitoring import notify

    monkeypatch.chdir(tmp_path)
    now = int(time.time() * 1000) // (4 * HOUR) * (4 * HOUR) + 60_000
    registry = {
        "BTCUSDT": {"symbol": "BTCUSDT", "interval": "60", "model_dir": "BTC_model", "warmup": 2},
        "BTCUSDT_4h": {"symbol": "BTCUSDT", "interval": "240", "model_dir": "BTC_4h", "warmup": 2},
    }
    monkeypatch.setattr(builder, "SYMBOL_CONFIG", registry)
    data = tmp_path / "data_files"
    data.mkdir()
    monkeypatch.setattr(builder, "DATA_DIR", data)
    stamps = [now - 60_000 - i * HOUR for i in reversed(range(17))]
    (data / "BTCUSDT_open_interest.csv").write_text(
        "timestamp,sum_open_interest\n" + "".join(f"{ts},{100 + i}\n" for i, ts in enumerate(stamps)))
    (data / "BTCUSDT_ls_ratio.csv").write_text(
        "timestamp,long_short_ratio\n" + "".join(f"{ts},{1 + i / 100}\n" for i, ts in enumerate(stamps)))
    (data / "BTCUSDT_1h.csv").write_text(
        "open_time,open,high,low,close,volume\n" + "".join(f"{ts},100,102,99,101,10\n" for ts in stamps))
    for row in registry.values():
        directory = tmp_path / "models_v8" / row["model_dir"]
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({"models": ["fixture.pkl"]}))
        (directory / "fixture.pkl").write_text("never deserialized: model loader is injected")
    config = tmp_path / "config" / "shadow.json"
    config.parent.mkdir()
    config.write_text(json.dumps({
        "version": 1, "mode": "shadow", "venue": "bybit", "currency": "USDT",
        "shadow_balance": "10000", "models_root": "models_v8", "data_root": "data_files",
        "runners": list(registry), "disabled_runners": [],
    }))
    manifest = load_run_manifest(config, symbol_config=registry)
    state = tmp_path / "data" / "runtime"
    initialize_runtime_state(state, "shadow-fixture")
    monkeypatch.setenv("RISK_SCOPE", "shadow-fixture")
    monkeypatch.setenv("RUNTIME_STATE_DIR", str(state))
    monkeypatch.setenv("ORDER_JOURNAL_PATH", str(state / "orders.sqlite"))
    monkeypatch.setenv("DRAWDOWN_STATE_PATH", str(state / "drawdown.sqlite"))
    for name in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "QUANT_RELEASE_REVISION",
                 "QUANT_RELEASE_IMAGE_ID", "QUANT_RELEASE_INPUT_DIGEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_loader_prod, "load_model", lambda path: dict(INFO))
    monkeypatch.setattr(model_loader_prod, "create_adapter", forbidden)
    monkeypatch.setattr(BybitAdapter, "__init__", forbidden)
    monkeypatch.setattr(builder, "get_initial_balance", forbidden)
    monkeypatch.setattr(AlphaDecisionModule, "on_fill", forbidden)
    monkeypatch.setattr(notify, "send_alert", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    repairs = []
    monkeypatch.setitem(sys.modules, "scripts.run_full_backtest",
                        SimpleNamespace(_fix_oi_file=lambda symbol: repairs.append(symbol)))
    real_build = builder.build_coordinator

    def build(*args, **kwargs):
        coord, module = real_build(*args, **kwargs)
        coord._data_readiness.clock = lambda: now / 1000
        return coord, module

    monkeypatch.setattr(builder, "build_coordinator", build)
    real_cache = oi_cache.BinanceOICache
    monkeypatch.setattr(oi_cache, "BinanceOICache", lambda *args, **kwargs:
                        real_cache(*args, **kwargs, clock=lambda: now / 1000))
    predictions = []
    monkeypatch.setattr(shadow, "_batch_prediction", lambda *args: predictions.append(args) or 1.5)

    class Public:
        venue = "bybit"
        broken = False
        initial_latest = False
        advance = 0

        def __init__(self, symbols=()):
            self.calls = []

        def get_closed_bars(self, symbol, *, interval, limit, after_ms=None):
            self.calls.append((symbol, interval, after_ms))
            size = int(interval) * 60_000
            latest = (now + self.advance) // size * size - size
            bars = [{"start": latest - i * size, "open": "100", "high": "102", "low": "99",
                     "close": "101", "volume": "10"}
                    for i in reversed(range(3 if after_ms is None else limit))]
            if after_ms is None:
                return bars[-limit:] if self.initial_latest else bars[:limit]
            return [bar for bar in bars if bar["start"] > after_ms]

        def fetch_oi(self, path, params):
            if self.broken:
                raise ValueError("deliberate public data failure")
            field = "sumOpenInterest" if path == oi_cache.OI_PATH else "longShortRatio"
            return [{"symbol": "BTCUSDT", "timestamp": ts + self.advance, field: "123"}
                    for ts in stamps[-8:]]

        def __getattr__(self, name):
            forbidden()

    from runner import public_market_data
    monkeypatch.setattr(public_market_data, "BybitPublicMarketData", Public)
    return SimpleNamespace(manifest=manifest, state=state, now=now, public=Public(),
                           repairs=repairs, predictions=predictions, real_build=real_build)


@pytest.fixture
def session(environment):
    env = environment
    monitor = RuntimeMonitor(None, execution=False, execution_runner_keys=[])
    journal = shadow.ShadowJournal(env.state / "shadow-observations.jsonl", env.manifest)
    result = shadow.ShadowSession(env.manifest, env.public, monitor, journal)
    result.start()
    try:
        yield result
    finally:
        result.close()


def records(env):
    return [json.loads(line) for line in (env.state / "shadow-observations.jsonl").read_text().splitlines()]


def test_real_native_warmup_and_closed_bars_preserve_runner_bindings(environment, session):
    env = environment
    one, four = session.modules["BTCUSDT"], session.modules["BTCUSDT_4h"]
    assert one._symbol == four._symbol == "BTCUSDT"
    assert one._runner_key == "BTCUSDT" and four._runner_key == "BTCUSDT_4h"
    assert not one._signal_only and four._signal_only
    assert one._consensus is four._consensus
    assert one._audit_enabled and four._audit_enabled
    assert all(coord._execution_bridge is None for coord in session.coordinators.values())
    assert [coord._data_readiness.interval_ms for coord in session.coordinators.values()] == [HOUR, 4 * HOUR]
    previous = dict(session.cursors)
    session.cycle()
    assert session.cursors == {key: value + (HOUR if key == "BTCUSDT" else 4 * HOUR)
                               for key, value in previous.items()}
    assert env.predictions == [("BTCUSDT", "BTC_model", session.cursors["BTCUSDT"])]
    assert env.repairs == []
    observations = [record for record in records(env) if record["type"] == "signal_observation"]
    assert len(observations) == 2
    assert all(row["inputs"]["ready"] and row["balance_source"] == "synthetic" for row in observations)
    assert all(row["manifest_digest"] == env.manifest.digest for row in records(env))


def test_unsubmitted_native_order_releases_pending_without_fills(environment, session):
    module, coord = session.modules["BTCUSDT"], session.coordinators["BTCUSDT"]
    before = coord.get_state_view()["account"]
    for _ in range(2):
        order = module._make_open_order(100, 1, Decimal("1"))[0]
        assert order.order_id in module._pending_orders
        coord.emit(order, actor="shadow")
        assert not module._pending_orders
    assert module._signal == 0 and module._current_qty == 0
    assert coord.get_state_view()["account"] == before
    intents = [record for record in records(environment) if record["type"] == "intent"]
    assert len(intents) == 2 and all(row["disposition"] == "not_submitted" for row in intents)


@pytest.mark.parametrize("defect", ["batch", "public_inputs"])
def test_unavailable_inputs_do_not_advance_cursor_or_emit_market(environment, session, monkeypatch, defect):
    if defect == "batch":
        monkeypatch.setattr(shadow, "_batch_prediction", lambda *args: None)
    else:
        environment.public.broken = True
    key = "BTCUSDT"
    previous = session.cursors[key]
    bars = session.modules[key]._bars_processed
    session.cycle()
    assert session.cursors[key] == previous
    assert session.modules[key]._bars_processed == bars
    blocked = [row for row in records(environment) if row["type"] == "inputs_blocked" and row["runner_key"] == key]
    assert blocked and blocked[0]["inputs"]["ready"] is False


def test_injected_execution_bridge_stops_before_market_or_order(environment, session):
    coord = session.coordinators["BTCUSDT"]
    coord._execution_bridge = SimpleNamespace(on_event=forbidden)
    previous = dict(session.cursors)
    with pytest.raises(RuntimeError, match="execution bridge"):
        session.cycle()
    assert session.cursors == previous


def test_stale_pending_bar_fails_instead_of_silently_skipping_or_blocking_forever(environment, session):
    coord = session.coordinators["BTCUSDT"]
    coord._data_readiness.clock = lambda: environment.now / 1000 + 3600
    previous = dict(session.cursors)
    with pytest.raises(RuntimeError, match="expired"):
        session.cycle()
    assert session.cursors == previous


def test_startup_consumes_latest_warmup_then_waits_for_new_closed_bar(environment):
    env = environment
    env.public.initial_latest = True
    monitor = RuntimeMonitor(None, execution=False, execution_runner_keys=[])
    journal = shadow.ShadowJournal(env.state / "shadow-observations.jsonl", env.manifest)
    session = shadow.ShadowSession(env.manifest, env.public, monitor, journal)
    session.start()
    try:
        initial = dict(session.cursors)
        session.cycle()
        assert session.cursors == initial and env.predictions == []
        assert not [row for row in records(env) if row["type"] == "inputs_blocked"]
        for hour in range(1, 5):
            env.public.advance = hour * HOUR
            for coord in session.coordinators.values():
                coord._data_readiness.clock = lambda: (env.now + env.public.advance) / 1000
            for cache in session.caches.values():
                cache._clock = lambda: (env.now + env.public.advance) / 1000
            session.cycle()
        assert session.cursors == {key: value + 4 * HOUR for key, value in initial.items()}
    finally:
        session.close()


def test_complete_run_publishes_shadow_identity_and_releases_managed_lease(environment, monkeypatch):
    from shared.runtime_status import RuntimeStatusWriter
    phases = []
    original = RuntimeStatusWriter.publish

    def publish(self, phase, **kwargs):
        phases.append(phase)
        return original(self, phase, **kwargs)

    monkeypatch.setattr(RuntimeStatusWriter, "publish", publish)
    shadow.run(environment.manifest, max_cycles=1)
    status = json.loads((environment.state / "runtime-status.json").read_text())
    assert phases == ["starting", "running", "stopping", "stopped"]
    assert status["mode"] == "shadow" and status["phase"] == "stopped"
    assert status["manifest_digest"] == environment.manifest.digest
    assert status["execution"] is None and status["execution_runner_keys"] == []
    with RuntimeStateLease(environment.state, "shadow-fixture"):
        pass
    assert environment.repairs == []


def test_missing_managed_state_fails_before_public_access(environment):
    (environment.state / "orders.sqlite").unlink()
    with pytest.raises(ValueError, match="missing"):
        shadow.run(environment.manifest, max_cycles=1)
    assert environment.public.calls == []


def test_unvalidated_manifest_cannot_start(environment):
    with pytest.raises(ValueError, match="validated local inputs"):
        shadow.ShadowSession(replace(environment.manifest, files_validated=False), environment.public,
                             None, None)
    assert environment.public.calls == []


@pytest.mark.parametrize("args", [[], ["--dry-run", "--enable-execution"], ["--dry-run", "--max-cycles", "0"]])
def test_cli_rejects_missing_acknowledgement_execution_and_invalid_limits(args, monkeypatch):
    monkeypatch.setattr(shadow, "load_run_manifest", forbidden)
    with pytest.raises(SystemExit) as failure:
        shadow.main(args)
    assert failure.value.code == 2


@pytest.mark.parametrize("balance,dry_run", [(Decimal("1"), False), (Decimal("0"), True),
                                           (Decimal("NaN"), True), (Decimal("90000000001"), True)])
def test_synthetic_balance_is_bounded_and_only_allowed_without_execution(environment, balance, dry_run):
    with pytest.raises(ValueError, match="synthetic balance"):
        environment.real_build("BTCUSDT", "BTCUSDT", INFO, environment.public,
                               dry_run=dry_run, starting_balance=balance, repair_inputs=False)

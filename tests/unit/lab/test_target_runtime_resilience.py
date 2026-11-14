"""Target runtime keeps working through restarts and market-data outages (migration step 1b)."""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from execution.adapters.binance.rest import BinanceRetryableError
from lab.data import frame_from_bars, funding_series
from lab.simulate import simulate
from lab.strategy import StrategySpec, compute_targets
from runner.binance_main import bar_from_kline
from runner.binance_target import PAPER_STATE, TargetRuntime
from tests.unit.binance.fake_venue import HOUR
from tests.unit.lab.test_target_runtime import LIVE, _advance, _world

MARKET_PATHS = ("/fapi/v1/klines", "/fapi/v1/fundingRate")


@pytest.fixture(autouse=True)
def _dev_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BYBIT_BASE_URL", "BINANCE_TESTNET", "OKX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _spec(execution="market", ttl=1):
    return StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05,
                        execution=execution, ttl_bars=ttl)


def _paper(world, state_dir, **kw):
    return TargetRuntime(spec_path=world["spec_path"], state_dir=state_dir, mode="paper",
                         market=world["adapter"], clock=lambda: world["clock"]["now"], **kw)


def _outage(world, on: dict) -> None:
    """While on["value"] is true, public market-data requests fail like a network error."""
    for fake in world["fakes"].values():
        original = fake.request_public

        def failing(*, method, path, params=None, _original=original):
            if on["value"] and path in MARKET_PATHS:
                raise BinanceRetryableError("Network error: injected")
            return _original(method=method, path=path, params=params)

        fake.request_public = failing


def _assert_matches_lab(world, runtime, spec):
    traded = 0
    for symbol in ("BTCUSDT", "ETHUSDT"):
        bars = [bar_from_kline(r) for r in world["rows"][symbol]]
        funding = {t: (D(str(r)), D("1")) for t, r in world["funding"]}
        targets = compute_targets(frame_from_bars(bars), funding_series(funding), spec)
        lab = simulate(symbol, bars, targets, funding, spec, start_ms=bars[world["warm"]].open_time)
        broker = runtime.sessions[symbol].sleeve.broker
        key = lambda f: (f["at"], f["side"], f["qty"], f["price"], f["fee"])  # noqa: E731
        assert [key(f) for f in broker.fills] == [key(f) for f in lab.fills]
        assert float(broker.equity) == pytest.approx(lab.curve["equity"].iloc[-1], abs=1e-9)
        traded += len(lab.fills)
    assert traded > 0


@pytest.mark.parametrize("execution,ttl", [("market", 1), ("maker", 4)])
def test_paper_restart_after_downtime_reproduces_lab_simulation(tmp_path, execution, ttl):
    spec = _spec(execution, ttl)
    world = _world(tmp_path, spec)
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    stop, resume = world["warm"] + 150, world["warm"] + 157
    for i in range(world["warm"], stop):
        _advance(world, runtime, i)
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][resume - 1]["start"] + HOUR + 5_000) / 1000
    runtime = _paper(world, state)            # process restarted after seven closed bars
    runtime.warmup()                          # restores the saved sleeves and replays the missed bars
    for i in range(resume, world["warm"] + LIVE):
        _advance(world, runtime, i)
    _assert_matches_lab(world, runtime, spec)


def test_paper_market_outage_is_survived_and_caught_up(tmp_path):
    spec = _spec()
    world = _world(tmp_path, spec)
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    on = {"value": False}
    _outage(world, on)
    for i in range(world["warm"], world["warm"] + LIVE):
        on["value"] = world["warm"] + 100 <= i < world["warm"] + 104
        _advance(world, runtime, i)           # never raises; missed bars are processed on recovery
    _assert_matches_lab(world, runtime, spec)
    issues = [json.loads(line) for line in (tmp_path / "paper" / "target_events.jsonl").read_text().splitlines()
              if '"issue"' in line]
    assert [row["failures"] for row in issues] == [1, 2, 3, 4]


def test_market_failure_backs_off_before_retrying(tmp_path):
    world = _world(tmp_path, _spec())
    runtime = _paper(world, tmp_path / "paper")
    runtime.warmup()
    calls = []
    for fake in world["fakes"].values():
        def failing(*, method, path, params=None):
            calls.append(path)
            raise BinanceRetryableError("Network error: injected")
        fake.request_public = failing
    _advance(world, runtime, world["warm"])
    assert len(calls) == 1 and runtime.status()["market_failures"] == 1
    world["clock"]["now"] += 4                # inside the first 5 s backoff: no request
    runtime.cycle()
    assert len(calls) == 1
    world["clock"]["now"] += 2
    runtime.cycle()
    assert len(calls) == 2 and runtime._market_retry_at - world["clock"]["now"] == pytest.approx(10)


def test_kill_flatten_runs_while_market_data_is_unavailable(tmp_path):
    world = _world(tmp_path, _spec())
    btc = world["fakes"]["BTCUSDT"]
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    i = world["warm"]
    while btc.positions.get("BTCUSDT", D(0)) == 0:
        assert i < world["warm"] + LIVE - 10
        _advance(world, runtime, i)
        i += 1
    _outage(world, {"value": True})
    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, i)               # the bar cannot be fetched, the kill switch still flattens
    assert btc.positions["BTCUSDT"] == 0
    assert runtime.status()["market_failures"] == 1


def test_saved_paper_state_of_another_spec_is_refused_until_reset(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + 5):
        _advance(world, runtime, i)
    saved = json.loads((state / PAPER_STATE).read_text())
    saved["spec"]["band"] = 0.2
    (state / PAPER_STATE).write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="another spec"):
        _paper(world, state).warmup()
    fresh = _paper(world, state, reset_paper=True)
    fresh.warmup()
    assert all(s.sleeve.broker.fills == [] for s in fresh.sessions.values())
    assert len(list(state.glob("paper_state.*.json"))) == 1   # the old record is archived, not deleted


def test_paper_state_older_than_the_available_history_is_refused(tmp_path):
    world = _world(tmp_path, _spec())
    state = tmp_path / "paper"
    runtime = _paper(world, state)
    runtime.warmup()
    saved = json.loads((state / PAPER_STATE).read_text())
    for symbol, snap in saved["symbols"].items():
        snap["cursor"] = world["rows"][symbol][0]["start"] - 24 * HOUR   # before the venue's first bar
    (state / PAPER_STATE).write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        _paper(world, state).warmup()

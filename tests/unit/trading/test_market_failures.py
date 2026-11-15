"""Market-data failures reach a person, and the exit side keeps working without market data (step 14, ADR-0007 4).

- Five failures in a row raise one alert; a success resets the count.
- An error the exchange calls final (a delisted symbol, a bad parameter) is alerted at once and blocks
  entries until market data works again; it is still retried, at the longest backoff.
- In execution mode a warm-up without market data does not end the process: reconciliation, stop upkeep and
  risk actions (a kill switch's close) run, and the warm-up is tried again.
"""
from __future__ import annotations

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import _events, _runtime, _start, _until_position
from tests.unit.binance.fake_venue import HOUR
from trading.adapters.binance.rest import BinanceNonRetryableError, BinanceRetryableError
from trading.application.runtime import MARKET_ALERT_AFTER


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def _market(world, error=None):
    """Make public market data fail with ``error`` (None restores it)."""
    for fake in world["fakes"].values():
        if error is None:
            fake.__dict__.pop("request_public", None)
            continue

        def down(*, method, path, params=None, _error=error):
            raise _error

        fake.request_public = down


def _alerts(tmp_path):
    return [r for r in _events(tmp_path, "alert") if "market data" in r["reason"]]


def _due(world, i):
    """Put the clock just past the close of bar ``i`` (so the runtime asks for it)."""
    world["clock"]["now"] = (world["rows"]["BTCUSDT"][i]["start"] + HOUR + 5_000) / 1000


def test_five_failures_in_a_row_raise_one_alert_and_a_success_resets_the_count(tmp_path):
    assert MARKET_ALERT_AFTER == 5
    world, runtime = _start(tmp_path)
    _due(world, world["warm"])
    _market(world, BinanceRetryableError("Network error: down"))
    for _ in range(12):
        _tick(world, runtime, 301)                        # past the longest backoff each time
    assert runtime.status()["market_failures"] >= 6 and len(_alerts(tmp_path)) == 1
    _market(world)
    _tick(world, runtime, 301)
    assert runtime.status()["market_failures"] == 0
    _market(world, BinanceRetryableError("Network error: down again"))
    world["clock"]["now"] += HOUR                         # the next bar is due, so market data is asked for
    for _ in range(5):
        _tick(world, runtime, 301)
    assert len(_alerts(tmp_path)) == 2                    # a new outage alerts again


def test_an_error_the_exchange_calls_final_is_alerted_at_once_and_blocks_entries_until_data_returns(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    session = runtime.sessions["BTCUSDT"]
    _market(world, BinanceNonRetryableError('HTTP 400: {"code":-1121,"msg":"Invalid symbol."}', http_status=400,
                                            code=-1121))
    _due(world, i)
    runtime.cycle()
    (alert,) = _alerts(tmp_path)
    assert "-1121" in alert["reason"] and session.executor.ready is False
    _tick(world, runtime, 6)
    assert session.executor.ready is False                # still blocked at the next reconciliation
    assert any("market data refused" in r["issue"] for r in _events(tmp_path, "issue"))
    _market(world)
    _tick(world, runtime, 301)                            # retried at the longest backoff, and it works again
    _tick(world, runtime, 6)
    assert runtime.status()["market_failures"] == 0 and session.executor.ready is True


def test_a_warm_up_without_market_data_keeps_the_exit_side_running(tmp_path):
    world, runtime = _start(tmp_path)
    i = _until_position(world, runtime)
    btc = world["fakes"]["BTCUSDT"]
    assert btc.positions["BTCUSDT"] != 0
    runtime = _runtime(tmp_path, world)                   # a restart while market data is down
    _market(world, BinanceRetryableError("Network error: down"))
    runtime.kill.write(active=True, reason="test", flatten=True)
    _due(world, i)
    assert runtime.warmup_or_hold() is False              # no exit from the process
    assert btc.positions["BTCUSDT"] == 0                  # and the kill switch's close went out
    _market(world)
    runtime.kill.write(active=False)
    world["clock"]["now"] += 301
    assert runtime.warmup_or_hold() is True               # warm-up done once data is back
    assert len(runtime.sessions["BTCUSDT"].bars) >= runtime.spec.window_hours()


def test_a_warm_up_retry_after_a_partial_warm_up_starts_clean(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    runtime = _runtime(tmp_path, world)
    eth = world["fakes"]["ETHUSDT"]

    def eth_down(*, method, path, params=None):
        raise BinanceRetryableError("Network error: down")

    eth.request_public = eth_down                         # BTC warms up, ETH does not
    assert runtime.warmup_or_hold() is False
    eth.__dict__.pop("request_public")
    world["clock"]["now"] += 301
    assert runtime.warmup_or_hold() is True               # no "non-contiguous bar" from the first attempt


def test_paper_warm_up_failures_still_end_the_process(tmp_path):
    from runner.binance_target import TargetRuntime
    from tests.unit.lab.test_target_runtime import _world
    from trading.domain.strategy.overlay import StrategySpec
    world = _world(tmp_path, StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    _market(world, BinanceRetryableError("Network error: down"))
    with pytest.raises(BinanceRetryableError):
        runtime.warmup_or_hold()

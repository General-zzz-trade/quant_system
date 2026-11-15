"""Every part of the execution runtime reads the runtime's clock (migration step 13a).

Before, the order lifecycle, the stop manager and the drawdown latch read the system clock while the
runtime read the injected one: in production they agree, but in tests (and in a future replay) the
"not found" grace, a maker entry's lifetime and the daily drawdown period ignored the simulated time.
"""
from __future__ import annotations

import pytest

from tests.characterization.test_runtime_decisions import _start, _until_position
from tests.characterization.test_exit_paths import _tick


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


def test_every_part_reads_the_runtime_clock(tmp_path):
    world, runtime = _start(tmp_path)
    now = world["clock"]["now"]
    session = runtime.sessions["BTCUSDT"]
    lifecycle = session.executor._service._orders                    # wiring check: private on purpose
    assert lifecycle.clock() == now
    assert session.executor._service._clock() == now
    assert session.stops.clock() == now
    assert runtime.drawdown.clock() == now
    assert runtime.pauses.clock() == now


def test_an_order_the_venue_never_got_settles_after_the_grace_in_simulated_time(tmp_path):
    world, runtime = _start(tmp_path)
    _until_position(world, runtime)
    btc, session = world["fakes"]["BTCUSDT"], runtime.sessions["BTCUSDT"]
    btc.fault = "error_5xx"                          # the next order's outcome is unknown; the venue has none
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime, 31)
    (row,) = [r for r in session.executor.journal.orders(unresolved=True) if r["symbol"] == "BTCUSDT"]
    created = row["created_at"]
    assert created == pytest.approx(world["clock"]["now"], abs=1)    # reserved at the simulated time
    for _ in range(3):
        _tick(world, runtime, 31)                     # 93 s after the order was reserved: past the 90 s grace
    settled = next(r for r in session.executor.journal.orders() if r["intent_id"] == row["intent_id"])
    assert (settled["settled"], settled["status"]) == (1, "not_found")
    assert btc.positions["BTCUSDT"] == 0             # and the next attempt closed the position

"""ADR-0011 (S5a): while running, the exit side checks the whole account with the start-up check's function, at
most every 60 seconds. A finding blocks every entry and cancels the strategy's resting entries; its result is
re-applied at every reconciliation until the next check; a holding that went away unblocks at the next check; an
account that cannot be read counts as a finding. Exits never depend on the check."""
from __future__ import annotations

import json
from decimal import Decimal as D

from tests.characterization.test_exit_paths import _execution, _tick, _until
from trading.adapters.binance.rate_limit import endpoint_weight


def _issues(tmp_path, kind):
    rows = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == "issue" and r.get("issue_kind") == kind]


def _ready(runtime):
    return {s: session.executor.ready for s, session in runtime.sessions.items()}


def _foreign_position(world):
    world["fakes"]["BTCUSDT"].positions["SOLUSDT"] = D("3")              # a manual position on another symbol


def test_a_foreign_position_found_while_running_blocks_every_entry(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    _tick(world, runtime)
    assert all(_ready(runtime).values())
    _foreign_position(world)
    _tick(world, runtime, seconds=61)                                    # the next whole-account check is due
    assert not any(_ready(runtime).values())
    assert _issues(tmp_path, "foreign_holding")


def test_the_block_holds_between_checks_and_lifts_at_the_check_after_the_holding_went(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    _foreign_position(world)
    _tick(world, runtime, seconds=61)
    for _ in range(8):                                                   # 48 s: no new check, the result holds
        _tick(world, runtime)
        assert not any(_ready(runtime).values())
    del world["fakes"]["BTCUSDT"].positions["SOLUSDT"]
    _tick(world, runtime)                                                # 54 s after the finding: still blocked
    assert not any(_ready(runtime).values())
    _tick(world, runtime, seconds=10)                                    # 64 s: checked again, nothing found
    assert all(_ready(runtime).values())


def test_an_account_that_cannot_be_read_counts_as_a_finding(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")

    def unreadable():
        raise ConnectionError("algo orders unavailable")
    monkeypatch.setattr(world["adapter"], "get_open_algo_orders", unreadable)
    _tick(world, runtime, seconds=61)
    assert not any(_ready(runtime).values())
    assert _issues(tmp_path, "account_unreadable")


def test_checks_are_at_least_sixty_seconds_apart(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    times = []
    get_algo = world["adapter"].get_open_algo_orders

    def counted():
        times.append(world["clock"]["now"])
        return get_algo()
    monkeypatch.setattr(world["adapter"], "get_open_algo_orders", counted)
    for _ in range(40):                                                  # 240 s of 6 s cycles
        _tick(world, runtime)
    assert 3 <= len(times) <= 4 and all(b - a >= 60 for a, b in zip(times, times[1:]))


def test_a_finding_cancels_the_strategy_s_resting_entries(tmp_path, monkeypatch):
    world, runtime = _execution(tmp_path, execution="market")
    cancelled = []
    for symbol, session in runtime.sessions.items():
        monkeypatch.setattr(session, "cancel_entries", lambda symbol=symbol: cancelled.append(symbol))
    _foreign_position(world)
    _tick(world, runtime, seconds=61)
    assert set(cancelled) == {"BTCUSDT", "ETHUSDT"}


def test_the_kill_switch_still_flattens_the_venue_position_with_a_foreign_holding(tmp_path):
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _foreign_position(world)
    _tick(world, runtime, seconds=61)
    runtime.kill.write(active=True, reason="test", flatten=True)
    _tick(world, runtime)
    assert btc.positions["BTCUSDT"] == 0


def test_account_wide_order_lists_weigh_what_binance_publishes():
    # ADR-0011 point 7: open orders without a symbol weigh 40, with one 1; the conditional-order list 40.
    assert endpoint_weight("/fapi/v1/openOrders", {"symbol": "BTCUSDT"}) == 1.0
    assert endpoint_weight("/fapi/v1/openOrders", {}) == 40.0
    assert endpoint_weight("/fapi/v1/openAlgoOrders", {}) == 40.0

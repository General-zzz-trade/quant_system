"""4h breakout strategy and its study harness, on synthetic data only."""
from __future__ import annotations

import csv
import json
import math
from decimal import Decimal as D
from types import SimpleNamespace as NS

import numpy as np
import pytest

from data.quality.causal_bars import HOUR, Bar
from lab.breakout import BreakoutSpec, desired_side, features, stop_distance
from lab.breakout_study import GATE, gate_checks, run, run_study, trade_ledger
from lab.data import DAY, frame_from_bars

T0 = 1_609_459_200_000          # 2021-01-01 00:00 UTC


def row(close, hi_entry=110, lo_entry=90, hi_exit=105, lo_exit=95):
    return NS(close=close, hi_entry=hi_entry, lo_entry=lo_entry, hi_exit=hi_exit, lo_exit=lo_exit)


@pytest.mark.parametrize("side, close, want", [
    (0, 111, 1), (0, 89, -1), (0, 100, 0),
    (1, 100, 1), (1, 94, 0), (1, 89, -1), (1, 111, 1),
    (-1, 100, -1), (-1, 106, 0), (-1, 111, 1), (-1, 89, -1),
])
def test_desired_side(side, close, want):
    assert desired_side(side, row(close)) == want


def bars_from_closes(closes, start=T0, spread=0.002):
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        hi, lo = max(o, c) * (1 + spread), min(o, c) * (1 - spread)
        out.append(Bar(start + i * HOUR, D(f"{o:.4f}"), D(f"{hi:.4f}"), D(f"{lo:.4f}"), D(f"{c:.4f}"), D(50000)))
        prev = c
    return out


def random_closes(n, seed=7, vol=0.008, start=2000.0):
    rng = np.random.default_rng(seed)
    return list(start * np.exp(np.cumsum(rng.normal(0, vol, n))))


def test_features_are_causal_and_channels_exclude_the_decision_bar():
    spec = BreakoutSpec()
    frame = frame_from_bars(bars_from_closes(random_closes(24 * 40)))
    full = features(frame, spec)
    assert not full["ready"].iloc[: spec.vol_bars].any() and full["ready"].iloc[-1]
    i = len(full) - 1
    four = full.iloc[i - spec.entry_bars:i]
    assert full["hi_entry"].iloc[i] == pytest.approx(max(
        frame["high"].iloc[(i - k) * 4:(i - k + 1) * 4].max() for k in range(1, spec.entry_bars + 1)))
    assert len(four) == spec.entry_bars
    # trailing window of window_hours() gives the same last row as full history
    tail = features(frame.iloc[-spec.window_hours():].reset_index(drop=True), spec)
    for col in ("atr", "vol", "hi_entry", "lo_entry", "hi_exit", "lo_exit"):
        assert tail[col].iloc[-1] == pytest.approx(full[col].iloc[-1], rel=1e-12)


def test_trade_ledger_attributes_fees_and_funding():
    fills = [{"at": 10 * HOUR, "side": "buy", "qty": "0.1", "price": "2000", "fee": "0.1", "order": 1},
             {"at": 20 * HOUR, "side": "sell", "qty": "0.1", "price": "2100", "fee": "0.105", "order": 2},
             {"at": 20 * HOUR, "side": "sell", "qty": "0.2", "price": "2100", "fee": "0.21", "order": 3},
             {"at": 30 * HOUR, "side": "buy", "qty": "0.2", "price": "2200", "fee": "0.22", "order": "protective_stop"}]
    orders = [{"id": 1, "reason": "entry"}, {"id": 2, "reason": "exit"}, {"id": 3, "reason": "entry"}]
    funding = [{"at": 10 * HOUR, "cashflow": "0"}, {"at": 16 * HOUR, "cashflow": "-0.02"},
               {"at": 20 * HOUR, "cashflow": "-0.03"}, {"at": 24 * HOUR, "cashflow": "0.04"}]
    trades, open_trade = trade_ledger(fills, orders, funding)
    assert open_trade is None and len(trades) == 2
    long, short = trades
    assert long["side"] == 1 and long["exit_kind"] == "signal_exit"
    assert long["funding"] == D("-0.05") and long["net"] == D("10") - D("0.205") - D("0.05")
    assert short["side"] == -1 and short["exit_kind"] == "protective_stop"
    assert short["funding"] == D("0.04") and short["net"] == D("-20") - D("0.43") + D("0.04")
    assert short["net_over_notional"] == short["net"] / D("420")


def synthetic_run(closes, **kwargs):
    bars = bars_from_closes(closes)
    frame = frame_from_bars(bars)
    rows = features(frame, BreakoutSpec())
    funding = {b.open_time: (D("0.0001"), b.open) for b in bars if b.open_time % (8 * HOUR) == 0}
    return run(bars, funding, rows, desired_side, start_ms=T0 + 10 * DAY,
               end_ms=bars[-1].open_time + HOUR, **kwargs)


def test_breakout_enters_with_a_stop_and_the_stop_exits():
    calm = [2000 * (1 + 0.004 * math.sin(i / 3)) for i in range(24 * 12)]
    rally = [calm[-1] * (1.002 ** (i + 1)) for i in range(24)]
    crash = [rally[-1] * (0.99 ** (i + 1)) for i in range(12)]
    result = synthetic_run(calm + rally + crash)
    assert result.entries and result.entries[0]["side"] == 1
    first = result.entries[0]
    assert 0 < first["stop_distance"] <= 0.0475 and 0 < first["exposure"] <= 3
    assert result.trades[0]["exit_kind"] == "protective_stop"
    assert not result.liquidations


def test_equity_reconciles_with_the_trade_ledger():
    result = synthetic_run(random_closes(24 * 90, seed=3, vol=0.01))
    assert len(result.trades) >= 3
    last = result.curve.iloc[-1]
    legs = result.trades + ([result.open_trade] if result.open_trade else [])
    cash = sum(float(t["cash"]) for t in legs)
    assert sum(float(t["fees"]) for t in legs) == pytest.approx(result.fees, abs=1e-9)
    assert last["equity"] == pytest.approx(167 + cash + last["qty"] * last["close"] - result.fees + result.funding,
                                           abs=1e-6)
    if result.open_trade is None:
        assert last["equity"] == pytest.approx(167 + sum(float(t["net"]) for t in result.trades), abs=1e-6)
    assert all(abs(float(e["exposure"])) <= 3 for e in result.entries)


def test_gate_checks():
    good = {"total_return": 0.5, "sharpe": 0.8, "max_drawdown_hourly": 0.4, "liquidations": 0,
            "round_trips_per_year": 60, "mean_net_per_round_trip": 0.01}
    assert all(c["ok"] for c in gate_checks(good, good).values())
    for key, bad in (("sharpe", 0.49), ("max_drawdown_hourly", 0.61), ("liquidations", 1),
                     ("round_trips_per_year", 151), ("mean_net_per_round_trip", 0.0029)):
        checks = gate_checks({**good, key: bad}, good)
        assert sum(not c["ok"] for c in checks.values()) == 1
    assert not gate_checks(good, {**good, "total_return": 0.0})["cost_x2_total_return"]["ok"]
    assert GATE["mean_net_per_round_trip_min"] == 0.003


def write_dataset(path, closes):
    path.mkdir()
    bars = bars_from_closes(closes)
    with (path / "ETHUSDT_1h.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.open_time, b.open, b.high, b.low, b.close, b.volume])
    with (path / "ETHUSDT_funding.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["funding_time", "rate", "mark_price"])
        for b in bars:
            if b.open_time % (8 * HOUR) == 0:
                w.writerow([b.open_time, "0.0001", b.open])


def test_study_never_uses_data_from_dev_end_on(tmp_path):
    closes = random_closes(24 * 70, seed=11, vol=0.01)
    wild = closes[: 24 * 60] + [c * 3 for c in closes[24 * 60:]]
    write_dataset(tmp_path / "a", closes)
    write_dataset(tmp_path / "b", wild)
    window = {"eval_start": T0 + 20 * DAY, "dev_end": T0 + 60 * DAY}
    a = run_study(tmp_path / "a", tmp_path / "out_a", **window)
    b = run_study(tmp_path / "b", tmp_path / "out_b", **window)
    assert json.dumps(a, default=float) == json.dumps(b, default=float)
    frozen = json.loads((tmp_path / "out_a" / "final_spec.json").read_text())
    assert frozen["passed"] is False and frozen["trials"] == 1 and "screen_passed" in frozen
    assert (tmp_path / "out_a" / "trades.csv").exists()
    with pytest.raises(ValueError, match="empty output"):
        run_study(tmp_path / "a", tmp_path / "out_a", **window)


def test_stop_distance():
    assert stop_distance(NS(atr=20.0, close=2000.0), BreakoutSpec()) == pytest.approx(0.015)

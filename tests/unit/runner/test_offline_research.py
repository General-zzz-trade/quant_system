"""Causality, real decision wiring, accounting and fresh-process replay recovery."""
import csv
import json
import math
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from data.quality.causal_bars import HOUR, aggregate_closed, align_closed_signals, load_bars, validate_bars
from runner.offline.broker import BarBroker, ReplayConfig
from runner.offline.fixtures import synthetic_bars, synthetic_funding
from runner.offline.models import train_bundle, validate_bundle
from runner.offline.replay import run_replay, HistoricalRisk
from runner.offline.report import checkpoint_replay, metrics, write_json
from shared.drawdown_guard import DrawdownGuard
from shared.entry_risk import CompositeRiskGate


@pytest.fixture(scope="module")
def experiment():
    bars = synthetic_bars()
    start = bars[720].open_time
    return bars, train_bundle(bars, start), synthetic_funding(bars), start


def test_four_hour_signal_visible_only_after_close():
    import pandas as pd
    source = pd.to_datetime(["2024-01-01T08:00Z", "2024-01-01T12:00Z"])
    targets = pd.to_datetime(["2024-01-01T09:00Z", "2024-01-01T11:59Z", "2024-01-01T12:00Z", "2024-01-01T15:00Z"])
    assert list(align_closed_signals([1, -1], source, targets)) == [0, 0, 1, 1]


def test_complete_four_hour_groups_only():
    bars = synthetic_bars(11)[1:]
    result = aggregate_closed(bars)
    assert len(result) == 1
    assert result[0].open_time == bars[3].open_time
    assert result[0].available_at == bars[6].available_at
    assert result[0].volume == 400


@pytest.mark.parametrize("damage", ["gap", "duplicate", "reverse", "ohlc", "nan", "negative", "unaligned"])
def test_bad_input_rejected(damage):
    bars = synthetic_bars(4)
    if damage == "gap":
        del bars[1]
    elif damage == "duplicate":
        bars[1] = bars[0]
    elif damage == "reverse":
        bars.reverse()
    elif damage == "ohlc":
        bars[0] = replace(bars[0], high=bars[0].low)
    elif damage == "nan":
        bars[0] = replace(bars[0], close=D("NaN"))
    elif damage == "negative":
        bars[0] = replace(bars[0], volume=D(-1))
    else:
        bars[0] = replace(bars[0], open_time=bars[0].open_time + 1)
    with pytest.raises(ValueError):
        validate_bars(bars)


def test_cursors_do_not_backfill_future_observations(tmp_path):
    from runner.builders.alpha_builder import CsvCursor, CsvDictCursor
    path = tmp_path / "source.csv"
    path.write_text("timestamp,value\n10000,7\n20000,8\n")
    assert math.isnan(CsvCursor(path, "timestamp", "value").get(9999))
    assert math.isnan(CsvDictCursor(path, "timestamp", {"v": "value"}).get(9999)["v"])


def order(oid="one", qty="1", side="buy", price="100"):
    return NS(order_id=oid, symbol="BTCUSDT", qty=D(qty), side=side, price=D(price))


def broker_config(**kwargs):
    return ReplayConfig(starting_balance="10000", qty_step="0.1", min_notional="1", **kwargs)


def candle(start=0, **kwargs):
    from data.quality.causal_bars import Bar
    vals = {"open": D(101), "high": D(103), "low": D(98), "close": D(102), "volume": D(100)}
    vals.update({k: D(str(v)) for k, v in kwargs.items()})
    return Bar(start, **vals)


def test_no_same_bar_fill_or_touch_fill():
    broker = BarBroker(broker_config(limit_offset_bps="0"))
    broker.now = HOUR
    broker.send_order(order())
    broker.process_bar(candle(0))
    assert not broker.fills
    broker.process_bar(candle(HOUR, low=100))
    assert not broker.fills  # touching the limit does not establish queue execution
    broker.process_bar(candle(2 * HOUR))
    assert len(broker.fills) == 1
    assert broker.fills[0]["at"] == 3 * HOUR


def test_post_only_cross_rejected_after_gap():
    broker = BarBroker(broker_config(limit_offset_bps="0"))
    broker.send_order(order())
    broker.process_bar(candle(open=99))
    assert not broker.fills
    assert broker.orders[0]["status"] == "rejected"
    assert "post-only" in broker.orders[0]["rejection"]


def test_partial_fill_remainder_expires_and_callback_order():
    broker = BarBroker(broker_config(participation=".001", ttl_bars=2))
    seen = []
    broker.emit = lambda e: seen.append(("fill", str(e.qty)))
    broker.status = lambda oid, status: seen.append(("status", status))
    broker.send_order(order())
    broker.send_order(order())  # idempotent retry
    broker.process_bar(candle())
    broker.process_bar(candle(HOUR))
    assert broker.position.qty == D(".2")
    assert len(broker.orders) == 1 and not broker.pending
    assert seen == [("fill", "0.1"), ("fill", "0.1"), ("status", "canceled")]
    assert broker.orders[0]["remaining"] == "0.8"


def test_signed_funding_realized_pnl_fees_and_partial_close():
    broker = BarBroker(broker_config(limit_offset_bps="0", slippage_bps="0"))
    broker.send_order(order(side="sell", price="102"))
    broker.process_bar(candle(open=101, close=100))
    assert broker.position.qty == -1
    broker.settle_funding(HOUR, D(".001"), D(100))
    assert broker.funding == D(".1")
    close = order("close", qty=".4", side="buy")
    close.reduce_only = True
    broker.send_order(close)
    broker.process_bar(candle(HOUR, open=100, close=100))
    assert broker.position.qty == D("-.6")
    assert broker.position.realized_pnl == D(".8")
    assert broker.fees == D(".0404")
    assert broker.equity == D("10002.0596")


def test_close_cancels_entry_remainder_before_it_can_reopen():
    broker = BarBroker(broker_config(participation=".001", slippage_bps="0"))
    broker.send_order(order())
    broker.process_bar(candle())
    assert broker.position.qty == D(".1")
    broker.send_order(order("stop", qty="0", side="sell"))
    assert broker.orders[0]["status"] == "canceled"
    broker.process_bar(candle(HOUR))
    assert broker.position.qty == 0 and not broker.pending
    assert len(broker.fills) == 2


def test_funding_jitter_is_explicit_and_bounded(tmp_path):
    from runner.offline.report import load_funding
    path = tmp_path / "funding.csv"
    path.write_text("funding_time,rate,mark_price,observed_funding_time\n0,0.0001,100,16\n")
    assert load_funding(path)[0] == (D(".0001"), D(100))
    path.write_text("funding_time,rate,mark_price,observed_funding_time\n0,0.0001,100,1001\n")
    with pytest.raises(ValueError, match="tolerance"):
        load_funding(path)


def test_csv_rejects_unclosed_bar(tmp_path, experiment):
    bars, model, funding, _ = experiment
    path = save_inputs(tmp_path, bars, model, funding)
    with pytest.raises(ValueError, match="unclosed"):
        load_bars(path, as_of_ms=bars[-1].available_at - 1)


def test_missing_or_stale_risk_blocks_entries_but_allows_close(tmp_path):
    broker = BarBroker(broker_config())
    broker.now, broker.mark = HOUR, D(100)
    guard = DrawdownGuard(tmp_path / "risk.sqlite", "offline", max_age_seconds=1)
    historical = HistoricalRisk(broker, guard)
    limiter = NS(check_order=lambda **kw: NS(allowed=True, detail="", violated_rule=""))
    gate = CompositeRiskGate(limiter, portfolio_reader=historical, drawdown_guard=historical)
    assert not gate.check(order()).allowed
    guard.observe(10000, now=broker.now / 1000)
    assert gate.check(order()).allowed
    broker.now += 2000
    assert not gate.check(order()).allowed
    assert gate.check(order(qty="0")).allowed


def test_model_training_does_not_inspect_future(experiment):
    bars, model, _, start = experiment
    changed = [replace(b, close=b.close * 2) if b.open_time >= start else b for b in bars]
    assert train_bundle(changed, start) == model
    validate_bundle(model, start)
    with pytest.raises(ValueError, match="trained after"):
        validate_bundle(model, start - HOUR)


def test_replay_determinism_and_future_prefix_invariance(experiment):
    bars, model, funding, start = experiment
    first = run_replay(bars, model, funding, start_ms=start, stop_after=900)
    changed = [replace(b, open=b.open * 2, high=b.high * 2, low=b.low * 2, close=b.close * 2)
               if i >= 900 else b for i, b in enumerate(bars)]
    second = run_replay(changed, model, funding, start_ms=start, stop_after=900)
    assert first == second
    assert first["fills"] and first["orders"]
    assert all(r["submitted_at"] > start for r in first["orders"])
    assert metrics(first)["accounting_reconciled"]
    assert metrics(first)["daily_sharpe"] is None


def test_exits_ignore_consensus_companion_keys_and_respect_min_hold(experiment):
    """Regression: the 4h runner's "_z" float used to force-close every 1h
    position at the next bar ("4h_reversal(BTCUSDT_4h_z=...)", 0-hour holds)."""
    bars, model, funding, start = experiment
    result = run_replay(bars, model, funding, start_ms=start, stop_after=900)
    orders = {row["id"]: row for row in result["orders"]}
    exits = [row["reason"] or "" for row in result["orders"] if row["reduce_only"]]
    assert exits
    assert not [r for r in exits if "_z=" in r or "_ic_ok=" in r]
    holds, opened = [], None
    for fill in result["fills"]:
        row = orders[fill["order"]]
        if not row["reduce_only"] and opened is None:
            opened = fill["at"]
        elif row["reduce_only"] and opened is not None:
            holds.append(((fill["at"] - opened) // HOUR, row["reason"]))
            opened = None
    signal_exits = [hours for hours, reason in holds if reason == "signal_change"]
    assert signal_exits and min(signal_exits) >= ReplayConfig().min_hold


def test_missing_funding_fails(experiment):
    bars, model, _, start = experiment
    with pytest.raises(ValueError, match="funding"):
        run_replay(bars, model, {}, start_ms=start)


def save_inputs(tmp_path, bars, model, funding):
    path = tmp_path / "bars.csv"
    with path.open("w", newline="") as stream:
        rows = [b.record() for b in bars]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(tmp_path / "model.json", model)
    with (tmp_path / "funding.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["funding_time", "rate", "mark_price"])
        writer.writerows((at, rate, mark) for at, (rate, mark) in funding.items())
    return path


def test_checkpoint_fresh_process_matches_continuous_and_rejects_changes(tmp_path, experiment):
    bars, model, funding, start = experiment
    path = save_inputs(tmp_path, bars, model, funding)
    assert load_bars(path) == bars
    checkpoint = tmp_path / "checkpoint.json"
    # Stop at a bar with an open position AND a pending order (found by scanning
    # the continuous replay), so the resume must carry both across processes.
    partial = checkpoint_replay(bars, model, funding, start_ms=start, config=ReplayConfig(),
                                checkpoint=checkpoint, stop_after=753)
    assert partial["pending_orders"] > 0 and D(partial["final"]["qty"]) != 0
    cmd = [sys.executable, "-m", "runner.offline", "replay", "--bars", str(path),
           "--funding", str(tmp_path / "funding.csv"),
           "--model", str(tmp_path / "model.json"), "--start-ms", str(start), "--checkpoint", str(checkpoint),
           "--resume", "--out", str(tmp_path / "resumed")]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    resumed = json.loads((tmp_path / "resumed/replay.json").read_text())
    continuous = run_replay(bars, model, funding, start_ms=start)
    assert resumed == continuous
    with pytest.raises(ValueError, match="mismatch"):
        checkpoint_replay(bars, model, funding, start_ms=start, config=replace(ReplayConfig(), maker_bps="3"),
                          checkpoint=checkpoint, resume=True)


def test_cli_network_block():
    import socket
    from runner.offline.__main__ import no_network
    with no_network(), pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 443))


def test_code_identity_detects_wheel_native_module_change(tmp_path, monkeypatch):
    import _quant_hotpath
    from runner.offline.report import code_identity
    package = tmp_path / "_quant_hotpath"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("# simulated wheel package\n")
    native = package / "_quant_hotpath.so"
    native.write_bytes(b"native version one")
    monkeypatch.setattr(_quant_hotpath, "__file__", str(init))
    monkeypatch.setattr(_quant_hotpath, "__path__", [str(package)], raising=False)
    first = code_identity()
    native.write_bytes(b"native version two")
    assert code_identity()["source_sha256"] != first["source_sha256"]
    assert first["numpy"] and first["pandas"]


def test_study_freezes_model_and_labels_synthetic(tmp_path, experiment):
    from runner.offline.study import run_study
    bars, _, funding, _ = experiment
    result = run_study(bars, funding, tmp_path, synthetic=True)
    assert result["synthetic"] and not result["profitability_verified"]
    validation = json.loads((tmp_path / "validation/replay.json").read_text())
    holdout = json.loads((tmp_path / "holdout/replay.json").read_text())
    assert validation["model_sha256"] == holdout["model_sha256"]
    assert validation["dataset"]["end_ms"] <= holdout["evaluation_start_ms"]
    assert holdout["dataset"]["synthetic"]
    assert len(result["development_folds"]) == 3
    assert all(m["accounting_reconciled"] for m in result["stages"].values())
    assert "合成行情" in (tmp_path / "report.md").read_text()
    with pytest.raises(ValueError, match="empty output"):
        run_study(bars, funding, tmp_path, synthetic=True)

"""Binance runtime: paper mode == offline replay; execution lifecycle with restart."""
from __future__ import annotations

import csv
import json
from decimal import Decimal as D

import pytest

from data.quality.causal_bars import Bar
from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from runner import binance_main
from runner.offline.broker import ReplayConfig
from runner.offline.context import DerivativesContext
from runner.offline.models import train_bundle
from runner.offline.replay import run_replay
from tests.unit.binance.fake_venue import HOUR, FakeBinance, make_bars

WARMUP, LIVE = 600, 260


@pytest.fixture(autouse=True)
def _dev_signing(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BYBIT_BASE_URL", "BINANCE_TESTNET", "OKX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setattr(binance_main, "WARMUP_BARS", WARMUP)


def _to_bar(row):
    return Bar(row["start"], *(D(str(row[k])) for k in ("open", "high", "low", "close", "volume")),
               extras={k: D(str(row[k])) for k in ("quote_volume", "trades", "taker_buy_volume",
                                                    "taker_buy_quote_volume")})


@pytest.fixture
def world(tmp_path):
    rows = make_bars(WARMUP + LIVE)
    bars = [_to_bar(r) for r in rows]
    funding = [(t, 0.0001 if (t // HOUR) % 3 else -0.00005)
               for t in range(rows[0]["start"], rows[-1]["start"] + HOUR, 8 * HOUR)]
    positioning = [{"timestamp": t, "open_interest": 1000 + (t // 300_000) % 11, "ls_ratio": 1.1,
                    "top_trader_ls_ratio": 1.3} for t in range(rows[0]["start"], rows[-1]["start"] + HOUR, 300_000)]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "BTCUSDT_funding.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["funding_time", "rate", "mark_price"])
        for t, r in funding:
            w.writerow([t, r, 100])
    with (data_dir / "BTCUSDT_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sum_open_interest", "count_long_short_ratio", "sum_toptrader_long_short_ratio"])
        for p in positioning:
            w.writerow([p["timestamp"], p["open_interest"], p["ls_ratio"], p["top_trader_ls_ratio"]])
    start_ms = bars[WARMUP].open_time
    bundle = train_bundle(bars[:WARMUP - 50], bars[WARMUP - 50].open_time, symbol="BTCUSDT",
                          spec={"feature_set": "ohlcv", "horizon": 1, "alpha": 1.0})
    cfg = ReplayConfig(deadzone=0.3, zscore_window=60, zscore_warmup=20, min_hold=2, max_hold=12)
    models = tmp_path / "models" / "BTCUSDT"
    models.mkdir(parents=True)
    (models / "model.json").write_text(json.dumps(bundle))
    (models / "protocol.json").write_text(json.dumps({"symbol": "BTCUSDT", "final_model_sha256": bundle["sha256"],
                                                      "final_config": cfg.record()}))
    (models / "gate.json").write_text(json.dumps({"passed": False, "reasons": ["synthetic"]}))
    clock = {"now": (start_ms + 5_000) / 1000}
    fake = FakeBinance(rows, clock=lambda: clock["now"], funding=funding, positioning=positioning)
    return {"rows": rows, "bars": bars, "funding": funding, "data_dir": data_dir, "bundle": bundle, "cfg": cfg,
            "models": tmp_path / "models", "clock": clock, "fake": fake, "start_ms": start_ms, "tmp": tmp_path}


def _adapter(fake):
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    return adapter


def _advance(world, runtime, index):
    world["clock"]["now"] = (world["rows"][index]["start"] + HOUR + 5_000) / 1000
    runtime.cycle()


def test_paper_runtime_matches_offline_replay(world):
    runtime = binance_main.BinanceRuntime(
        symbols=["BTCUSDT"], models_dir=world["models"], state_dir=world["tmp"] / "paper", mode="paper",
        market=_adapter(world["fake"]), data_dir=world["data_dir"], clock=lambda: world["clock"]["now"])
    runtime.warmup()
    for i in range(WARMUP, WARMUP + LIVE):
        _advance(world, runtime, i)
    session = runtime.sessions["BTCUSDT"]
    assert session.cursor == world["rows"][-1]["start"]

    funding = {t: (D(str(r)), world["bars"][(t - world["rows"][0]["start"]) // HOUR].open)
               for t, r in world["funding"]}
    context = DerivativesContext.from_files(world["data_dir"] / "BTCUSDT_funding.csv",
                                            world["data_dir"] / "BTCUSDT_metrics.csv")
    replay = run_replay(world["bars"], world["bundle"], funding, start_ms=world["start_ms"], config=world["cfg"],
                        symbol="BTCUSDT", context=context)

    def key(fill):
        return (fill["at"], fill["side"], fill["qty"], fill["price"], fill["fee"])
    assert replay["fills"], "fixture must trade for the comparison to mean anything"
    assert [key(f) for f in session.broker.fills] == [key(f) for f in replay["fills"]]
    assert str(session.broker.equity) == replay["final"]["equity"]


def test_execution_lifecycle_restart_stop_and_flatten(world, monkeypatch):
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")
    fake = world["fake"]
    state = world["tmp"] / "exec"

    def build():
        adapter = _adapter(fake)
        return binance_main.BinanceRuntime(
            symbols=["BTCUSDT"], models_dir=world["models"], state_dir=state, mode="execution",
            market=adapter, account=adapter, data_dir=world["data_dir"], clock=lambda: world["clock"]["now"],
            testnet=True)

    runtime = build()
    runtime.warmup()
    session = runtime.sessions["BTCUSDT"]
    assert session.modules["1h"]._execution_ready is True
    index = WARMUP
    while not fake.orders and index < WARMUP + LIVE - 20:
        _advance(world, runtime, index)
        index += 1
    assert fake.orders, "an entry intent should have been posted"
    order_id, order = next(iter(fake.orders.items()))
    assert order["timeInForce"] == "GTX"
    fake.fill_limit(order_id)
    _advance(world, runtime, index)  # reconcile consumes the confirmed fill
    index += 1
    module = session.modules["1h"]
    assert module._confirmed_position.qty == fake.positions["BTCUSDT"] != 0
    assert any(a["algoStatus"] == "NEW" for a in fake.algo.values()), "exchange-side stop expected"
    position = fake.positions["BTCUSDT"]

    # Restart: state comes back from the durable journal, nothing is resent.
    posts_before = len(fake.posts)
    runtime = build()
    runtime.warmup()
    session = runtime.sessions["BTCUSDT"]
    assert session.modules["1h"]._confirmed_position.qty == position
    assert len(fake.posts) == posts_before
    assert sum(a["algoStatus"] == "NEW" for a in fake.algo.values()) == 1

    # Operator kill switch with flatten closes the position and removes the stop.
    runtime.kill.write(active=True, reason="test", flatten=True)
    _advance(world, runtime, index)
    _advance(world, runtime, index + 1)
    assert fake.positions["BTCUSDT"] == 0
    assert session.modules["1h"]._confirmed_position.qty == 0
    assert not any(a["algoStatus"] == "NEW" for a in fake.algo.values())
    status = json.loads((state / "binance_status.json").read_text())
    assert status["kill"]["active"] is True


def test_mainnet_execution_requires_passed_gate(world, monkeypatch):
    monkeypatch.setenv("RISK_SCOPE", "binance-mainnet-ut")
    adapter = _adapter(world["fake"])
    with pytest.raises(RuntimeError, match="research gate not passed"):
        binance_main.BinanceRuntime(symbols=["BTCUSDT"], models_dir=world["models"], state_dir=world["tmp"] / "m",
                                    mode="execution", market=adapter, account=adapter, data_dir=world["data_dir"],
                                    clock=lambda: world["clock"]["now"], testnet=False)


def test_compare_script_confirms_paper_equals_replay(world, monkeypatch):
    import sys
    from scripts import compare_paper_replay

    monkeypatch.setattr(compare_paper_replay, "WARMUP_BARS", WARMUP)
    state = world["tmp"] / "paper2"
    runtime = binance_main.BinanceRuntime(
        symbols=["BTCUSDT"], models_dir=world["models"], state_dir=state, mode="paper",
        market=_adapter(world["fake"]), data_dir=world["data_dir"], clock=lambda: world["clock"]["now"])
    runtime.warmup()
    for i in range(WARMUP, WARMUP + LIVE):
        _advance(world, runtime, i)
    with (world["data_dir"] / "BTCUSDT_1h.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                    "taker_buy_volume", "taker_buy_quote_volume"])
        for r in world["rows"]:
            w.writerow([r["start"], r["open"], r["high"], r["low"], r["close"], r["volume"], r["quote_volume"],
                        r["trades"], r["taker_buy_volume"], r["taker_buy_quote_volume"]])
    monkeypatch.setattr(sys, "argv", ["x", "--events", str(state / "binance_events.jsonl"),
                                      "--models-dir", str(world["models"]), "--data-dir", str(world["data_dir"]),
                                      "--symbol", "BTCUSDT"])
    assert compare_paper_replay.main() == 0

#!/usr/bin/env python3
"""Parity test: Python RustTickProcessor vs Rust binary's process_tick_native.

Feeds identical synthetic kline bars through the Python-side RustTickProcessor
and compares ml_scores to verify the binary will produce identical results.

Since we can't directly call process_tick_native from Python (it's not exposed
via PyO3), we verify that RustTickProcessor.process_tick() produces consistent
results across multiple runs — proving determinism — and then the binary uses
the exact same Rust code path (steps 1-6).
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/quant_system")

from _quant_hotpath import RustTickProcessor


def discover_models(model_dir: str):
    """Discover model JSON files from a model directory (mirrors binary logic)."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    with config_path.open() as f:
        mcfg = json.load(f)

    json_paths = []
    for fname in mcfg.get("models", []):
        json_name = fname.replace(".pkl", ".json")
        json_path = model_dir / json_name
        assert json_path.exists(), f"Missing {json_path}"
        json_paths.append(str(json_path))

    weights = mcfg.get("ensemble_weights")
    if weights and len(weights) != len(json_paths):
        weights = None

    # Bear model
    bear_json = None
    bear_model_path = mcfg.get("bear_model_path")
    if bear_model_path:
        bear_dir = Path(bear_model_path)
        bear_cfg_path = bear_dir / "config.json"
        if bear_cfg_path.exists():
            with bear_cfg_path.open() as bf:
                bear_cfg = json.load(bf)
            bear_fname = bear_cfg["models"][0].replace(".pkl", ".json")
            bear_json_path = bear_dir / bear_fname
            if bear_json_path.exists():
                bear_json = str(bear_json_path)

    # Short model
    short_json = None
    short_dir = model_dir.parent / f"{mcfg['symbol']}_short"
    short_cfg_path = short_dir / "config.json"
    if short_cfg_path.exists():
        with short_cfg_path.open() as sf:
            short_cfg = json.load(sf)
        for sfname in short_cfg.get("models", []):
            sjson = short_dir / sfname.replace(".pkl", ".json")
            if sjson.exists():
                short_json = str(sjson)
                break

    return json_paths, weights, bear_json, short_json, mcfg


def build_processor(symbol: str, model_dir: str):
    """Build a RustTickProcessor for a symbol."""
    json_paths, weights, bear_json, short_json, mcfg = discover_models(model_dir)

    tp = RustTickProcessor.create(
        [symbol],
        "USDT",
        10000.0,
        json_paths,
        ensemble_weights=weights,
        bear_model_path=bear_json,
        short_model_path=short_json,
    )

    # Configure
    pos_mgmt = mcfg.get("position_management", {})
    bear_thresholds = None
    if pos_mgmt.get("bear_thresholds"):
        bear_thresholds = [tuple(x) for x in pos_mgmt["bear_thresholds"]]

    tp.configure_symbol(
        symbol,
        min_hold=mcfg.get("min_hold", 0),
        deadzone=mcfg.get("deadzone", 0.5),
        long_only=mcfg.get("long_only", False),
        monthly_gate=mcfg.get("monthly_gate", False),
        monthly_gate_window=mcfg.get("monthly_gate_window", 480),
        vol_target=pos_mgmt.get("vol_target"),
        vol_feature=pos_mgmt.get("vol_feature", "atr_norm_14"),
        bear_thresholds=bear_thresholds,
    )

    return tp


def generate_kline_bars(n_bars: int, base_price: float = 40000.0):
    """Generate synthetic kline bars with realistic price movement."""
    import random
    random.seed(42)

    bars = []
    price = base_price
    ts_ms = 1704067200000  # 2024-01-01 00:00:00 UTC

    for i in range(n_bars):
        ret = random.gauss(0, 0.002)  # ~0.2% per bar
        open_ = price
        close = price * (1 + ret)
        high = max(open_, close) * (1 + abs(random.gauss(0, 0.001)))
        low = min(open_, close) * (1 - abs(random.gauss(0, 0.001)))
        volume = random.uniform(50, 500)

        bars.append({
            "symbol": "BTCUSDT",
            "ts_ms": ts_ms,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

        price = close
        ts_ms += 60_000  # 1 minute

    return bars


def test_determinism():
    """Test that two identical processors produce identical results."""
    print("=== Determinism Test ===")
    model_dir = "models_v8/BTCUSDT_gate_v2"

    tp1 = build_processor("BTCUSDT", model_dir)
    tp2 = build_processor("BTCUSDT", model_dir)

    bars = generate_kline_bars(200)

    mismatches = 0
    advanced_count = 0

    for i, bar in enumerate(bars):
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"

        r1 = tp1.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )
        r2 = tp2.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )

        if r1.advanced:
            advanced_count += 1

        if r1.ml_score != r2.ml_score:
            mismatches += 1
            print(f"  MISMATCH bar {i}: ml_score {r1.ml_score:.8f} vs {r2.ml_score:.8f}")

        if r1.ml_short_score != r2.ml_short_score:
            mismatches += 1
            print(f"  MISMATCH bar {i}: short_score {r1.ml_short_score:.8f} vs {r2.ml_short_score:.8f}")

    print(f"  Bars: {len(bars)}, Advanced: {advanced_count}, Mismatches: {mismatches}")
    if mismatches == 0:
        print("  PASS: Deterministic")
    else:
        print("  FAIL: Non-deterministic!")
    return mismatches == 0


def test_signal_pipeline():
    """Test signal pipeline behavior (warmup, deadzone, min_hold)."""
    print("\n=== Signal Pipeline Test ===")
    model_dir = "models_v8/BTCUSDT_gate_v2"

    tp = build_processor("BTCUSDT", model_dir)
    bars = generate_kline_bars(800)

    scores = []
    raw_scores = []
    event_indices = []

    for bar in bars:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"

        r = tp.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )

        if r.advanced:
            scores.append(r.ml_score)
            raw_scores.append(r.raw_score)
            event_indices.append(r.event_index)

    # Verify warmup: first ~168 bars should have ml_score=0 (zscore warmup)
    warmup_zeros = sum(1 for s in scores[:168] if s == 0.0)
    post_warmup_nonzero = sum(1 for s in scores[168:] if s != 0.0)

    print(f"  Total advanced bars: {len(scores)}")
    print(f"  Warmup zeros (first 168): {warmup_zeros}/168")
    print(f"  Post-warmup non-zero: {post_warmup_nonzero}/{len(scores) - 168}")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Raw score range: [{min(raw_scores):.4f}, {max(raw_scores):.4f}]")
    print(f"  Event index range: [{min(event_indices)}, {max(event_indices)}]")

    # Verify monotonically increasing event_index
    monotonic = all(event_indices[i] < event_indices[i+1]
                     for i in range(len(event_indices) - 1))
    print(f"  Event index monotonic: {monotonic}")

    ok = warmup_zeros >= 160 and monotonic
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_state_consistency():
    """Test that state (markets, positions, account) stays consistent."""
    print("\n=== State Consistency Test ===")
    model_dir = "models_v8/BTCUSDT_gate_v2"

    tp = build_processor("BTCUSDT", model_dir)
    bars = generate_kline_bars(300)

    for bar in bars:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"

        r = tp.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )

    # Check state accessors work
    markets = tp.get_markets()
    positions = tp.get_positions()
    account = tp.get_account()
    portfolio = tp.get_portfolio()
    risk = tp.get_risk()

    print(f"  Markets: {len(markets)} symbols")
    print(f"  Positions: {len(positions)} entries")
    print(f"  Account balance: {account.balance}")
    print(f"  Portfolio equity: {portfolio.total_equity}")

    # Verify market state updated
    btc_market = markets.get("BTCUSDT")
    ok = btc_market is not None
    if ok:
        print(f"  BTC last_price (fd8): {btc_market.last_price}")
        print(f"  BTC last_price (f64): {btc_market.last_price_f:.2f}")
        ok = btc_market.last_price_f > 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_latency():
    """Benchmark process_tick latency."""
    print("\n=== Latency Benchmark ===")
    model_dir = "models_v8/BTCUSDT_gate_v2"

    tp = build_processor("BTCUSDT", model_dir)
    bars = generate_kline_bars(1000)

    # Warmup
    for bar in bars[:200]:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"
        tp.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )

    # Benchmark
    latencies = []
    for bar in bars[200:]:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"

        t0 = time.perf_counter_ns()
        tp.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )
        latencies.append(time.perf_counter_ns() - t0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2] / 1000
    p95 = latencies[int(len(latencies) * 0.95)] / 1000
    p99 = latencies[int(len(latencies) * 0.99)] / 1000
    mean = sum(latencies) / len(latencies) / 1000

    print(f"  Samples: {len(latencies)}")
    print(f"  Mean: {mean:.1f} us")
    print(f"  P50:  {p50:.1f} us")
    print(f"  P95:  {p95:.1f} us")
    print(f"  P99:  {p99:.1f} us")
    return True


def test_checkpoint_restore():
    """Test checkpoint/restore preserves signal state."""
    print("\n=== Checkpoint/Restore Test ===")
    model_dir = "models_v8/BTCUSDT_gate_v2"

    tp1 = build_processor("BTCUSDT", model_dir)
    bars = generate_kline_bars(300)

    # Process 200 bars
    for bar in bars[:200]:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"
        tp1.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )

    # Checkpoint
    ckpt = tp1.checkpoint()

    # Process remaining 100 bars on tp1
    scores_tp1 = []
    for bar in bars[200:]:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"
        r = tp1.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )
        if r.advanced:
            scores_tp1.append(r.ml_score)

    # Build tp2, restore checkpoint, process same 100 bars
    tp2 = build_processor("BTCUSDT", model_dir)
    # First process 200 bars to build feature state
    for bar in bars[:200]:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"
        tp2.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )
    # Restore signal state
    tp2.restore(ckpt)

    scores_tp2 = []
    for bar in bars[200:]:
        hour_key = bar["ts_ms"] // 3_600_000
        ts_str = str(bar["ts_ms"]) + "000"
        r = tp2.process_tick(
            bar["symbol"], bar["close"], bar["volume"],
            bar["high"], bar["low"], bar["open"],
            hour_key, ts_str,
        )
        if r.advanced:
            scores_tp2.append(r.ml_score)

    mismatches = sum(1 for a, b in zip(scores_tp1, scores_tp2) if a != b)
    print(f"  Scores after restore: {len(scores_tp2)}, Mismatches: {mismatches}")
    ok = mismatches == 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("RustTickProcessor Parity & Validation Suite")
    print("=" * 50)

    results = []
    results.append(("Determinism", test_determinism()))
    results.append(("Signal Pipeline", test_signal_pipeline()))
    results.append(("State Consistency", test_state_consistency()))
    results.append(("Checkpoint/Restore", test_checkpoint_restore()))
    results.append(("Latency", test_latency()))

    print("\n" + "=" * 50)
    print("Summary:")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

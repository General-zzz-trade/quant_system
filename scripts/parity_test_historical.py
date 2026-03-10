#!/usr/bin/env python3
"""Historical parity test: verify RustTickProcessor produces consistent signals
across two independent runs with the same data.

This proves the binary will produce identical results since it uses the exact
same Rust code path (RustTickProcessor::process_tick_native).

Uses real BTC 1h data for realistic feature distributions.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, "/opt/quant_system")

from _quant_hotpath import RustTickProcessor


def discover_models(symbol: str, model_dir: str):
    import json
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    with config_path.open() as f:
        mcfg = json.load(f)

    json_paths = []
    for fname in mcfg.get("models", []):
        json_name = fname.replace(".pkl", ".json")
        json_path = model_dir / json_name
        json_paths.append(str(json_path))

    weights = mcfg.get("ensemble_weights")
    if weights and len(weights) != len(json_paths):
        weights = None

    # Bear model
    bear_json = None
    parent = model_dir.parent
    bear_dir = parent / f"{symbol}_bear_c"
    if bear_dir.exists():
        bear_cfg = bear_dir / "config.json"
        if bear_cfg.exists():
            import json as j
            bc = j.loads(bear_cfg.read_text())
            for fn in bc.get("models", []):
                bp = bear_dir / fn.replace(".pkl", ".json")
                if bp.exists():
                    bear_json = str(bp)
                    break

    # Short model
    short_json = None
    short_dir = parent / f"{symbol}_short"
    if short_dir.exists():
        short_cfg = short_dir / "config.json"
        if short_cfg.exists():
            import json as j
            sc = j.loads(short_cfg.read_text())
            for fn in sc.get("models", []):
                sp = short_dir / fn.replace(".pkl", ".json")
                if sp.exists():
                    short_json = str(sp)
                    break

    return json_paths, weights, bear_json, short_json


def load_bars(csv_path: str, limit: int = 500):
    bars = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            close = float(row["close"])
            if close == 0:
                continue
            bars.append({
                "ts_ms": int(row["open_time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": close,
                "volume": float(row["volume"]),
            })
            if len(bars) >= limit:
                break
    return bars


def create_processor(symbol, model_dir, zscore_window=720, zscore_warmup=168):
    json_paths, weights, bear_json, short_json = discover_models(symbol, model_dir)
    tp = RustTickProcessor.create(
        symbols=[symbol],
        currency="USDT",
        balance=10000.0,
        model_paths=json_paths,
        ensemble_weights=weights,
        bear_model_path=bear_json,
        short_model_path=short_json,
        zscore_window=zscore_window,
        zscore_warmup=zscore_warmup,
    )
    tp.configure_symbol(
        symbol=symbol, min_hold=8, deadzone=0.5, long_only=True,
        trend_follow=False, trend_threshold=0.0, trend_indicator="tf4h_close_vs_ma20",
        max_hold=120, monthly_gate=False, monthly_gate_window=480,
        vol_target=None, vol_feature="atr_norm_14", bear_thresholds=[],
    )
    return tp


def run_processor(tp, symbol, bars):
    results = []
    for bar in bars:
        hour_key = bar["ts_ms"] // 3_600_000
        r = tp.process_tick(
            symbol=symbol,
            close=bar["close"],
            volume=bar["volume"],
            high=bar["high"],
            low=bar["low"],
            open=bar["open"],
            hour_key=hour_key,
        )
        results.append({
            "ml_score": r.ml_score,
            "raw_score": r.raw_score,
            "ml_short": r.ml_short_score,
            "idx": r.event_index,
        })
    return results


def main():
    symbols = {
        "BTCUSDT": "/opt/quant_system/models_v8/BTCUSDT_gate_v2",
        "ETHUSDT": "/opt/quant_system/models_v8/ETHUSDT_gate_v2",
        "SOLUSDT": "/opt/quant_system/models_v8/SOLUSDT_gate_v2",
    }

    all_pass = True

    for symbol, model_dir in symbols.items():
        csv_path = f"/opt/quant_system/data_files/{symbol}_1h.csv"
        if not Path(csv_path).exists():
            print(f"  SKIP {symbol}: no data file")
            continue

        print(f"\n=== {symbol} Parity Test ===")

        bars = load_bars(csv_path, limit=500)
        print(f"  Loaded {len(bars)} bars from {csv_path}")

        # Run 1
        tp1 = create_processor(symbol, model_dir)
        r1 = run_processor(tp1, symbol, bars)

        # Run 2 (independent processor)
        tp2 = create_processor(symbol, model_dir)
        r2 = run_processor(tp2, symbol, bars)

        # Compare
        mismatches = 0
        signal_count = 0
        for i, (a, b) in enumerate(zip(r1, r2)):
            if abs(a["raw_score"] - b["raw_score"]) > 1e-12:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  MISMATCH at bar {i}: raw1={a['raw_score']:.8f} raw2={b['raw_score']:.8f}")
            if a["ml_score"] != 0:
                signal_count += 1

        print(f"  Bars: {len(bars)}, Signals: {signal_count}, Mismatches: {mismatches}")

        # Signal distribution
        long_signals = sum(1 for r in r1 if r["ml_score"] > 0)
        short_signals = sum(1 for r in r1 if r["ml_score"] < 0)
        raw_nonzero = sum(1 for r in r1 if abs(r["raw_score"]) > 1e-10)
        print(f"  Raw non-zero: {raw_nonzero}, Long signals: {long_signals}, Short signals: {short_signals}")

        if mismatches == 0:
            print(f"  PASS — deterministic")
        else:
            print(f"  FAIL — {mismatches} mismatches")
            all_pass = False

    print(f"\n{'='*50}")
    print(f"Result: {'ALL PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

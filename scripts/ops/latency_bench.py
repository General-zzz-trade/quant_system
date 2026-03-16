#!/usr/bin/env python3
"""Tick-to-trade latency benchmark — measures full pipeline latency distribution.

Builds the real production stack (feature hook + pipeline + decision bridge)
and injects synthetic market events, capturing per-stage timing.

Usage:
    QUANT_ALLOW_UNSIGNED_MODELS=1 python3 -m scripts.latency_bench \
        --config infra/config/examples/testnet_btc_gate_v2.yaml --bars 500
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Pin to isolated CPU1
try:
    os.sched_setaffinity(0, {1})
except OSError as e:
    logger.debug("Could not pin to CPU1: %s", e)


@dataclass(slots=True)
class TickTiming:
    bar_idx: int = 0
    t_start: int = 0
    t_features: int = 0
    t_pipeline: int = 0
    t_decision: int = 0
    t_end: int = 0


def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def _report(label: str, values_ns: List[int]) -> None:
    if not values_ns:
        print(f"  {label:25s}  (no data)")
        return
    us = [v / 1000.0 for v in values_ns]
    print(f"  {label:25s}  p50={_pct(us,50):7.1f}  p95={_pct(us,95):7.1f}  "
          f"p99={_pct(us,99):7.1f}  mean={statistics.mean(us):7.1f}  "
          f"min={min(us):7.1f}  max={max(us):7.1f}  (us)")


def run_bench(config_path: Path, num_bars: int, warmup_bars: int) -> None:
    from infra.config.loader import load_config_secure
    from runner.testnet_validation import _build_ml_stack
    from runner.live_paper_runner import LivePaperRunner, LivePaperConfig

    raw = load_config_secure(config_path)
    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    per_symbol_bridges = signal_kwargs.pop("per_symbol_bridges", None)
    unified_predictors = signal_kwargs.pop("unified_predictors", None)
    tick_processors = signal_kwargs.pop("tick_processors", None)

    config = LivePaperConfig(symbols=symbols, starting_balance=10000.0, testnet=True)

    # Build production stack (no WS transport — we'll inject events manually)
    runner = LivePaperRunner.build(
        config,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=tick_processors,
        **signal_kwargs,
    )

    coord = runner.coordinator
    coord.start()

    has_tp = tick_processors is not None
    mode_label = "TICK_PROCESSOR (full Rust)" if has_tp else "LEGACY (Python pipeline)"

    # Instrument: monkey-patch feature hook + decision bridge
    timings: List[TickTiming] = []
    current: List[Optional[TickTiming]] = [None]  # mutable ref
    _ns = time.perf_counter_ns

    if not has_tp:
        # Patch feature hook
        hook = coord._cfg.feature_hook
        if hook is not None:
            _orig_on_event = hook.on_event
            def _timed_on_event(event):
                result = _orig_on_event(event)
                if current[0] is not None:
                    current[0].t_features = _ns()
                return result
            hook.on_event = _timed_on_event

        # Patch pipeline output callback (frozen dataclass — bypass)
        _orig_output_cb = coord._cfg.on_pipeline_output
        def _timed_output(out):
            if current[0] is not None:
                current[0].t_pipeline = _ns()
            if _orig_output_cb is not None:
                _orig_output_cb(out)
        object.__setattr__(coord._cfg, "on_pipeline_output", _timed_output)

        # Patch the actual pipeline handler in the dispatcher's handler list
        from engine.dispatcher import Route
        pipeline_handlers = coord._dispatcher._handlers[Route.PIPELINE]
        if pipeline_handlers:
            _orig_handler_0 = pipeline_handlers[0]
            def _timed_handler_0(event):
                _orig_handler_0(event)
                if current[0] is not None and current[0].t_decision == 0:
                    current[0].t_decision = _ns()
            pipeline_handlers[0] = _timed_handler_0

    # Generate synthetic events
    from event.types import EventType

    @dataclass
    class _Header:
        event_type: Any = EventType.MARKET
        ts: float = 0.0
        event_id: Optional[str] = None
        symbol: Optional[str] = None

    @dataclass
    class _MarketEvent:
        EVENT_TYPE: str = "market"
        event_type: Any = EventType.MARKET
        header: Any = None
        symbol: str = "BTCUSDT"
        close: float = 50000.0
        open: float = 49990.0
        high: float = 50050.0
        low: float = 49950.0
        volume: float = 100.0
        ts: float = 0.0

    from datetime import datetime, timezone
    total = warmup_bars + num_bars
    base = 50000.0

    print(f"\n{'='*60}")
    print("  LATENCY BENCHMARK")
    print(f"{'='*60}")
    print(f"  Mode:       {mode_label}")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Warmup:     {warmup_bars} bars")
    print(f"  Measure:    {num_bars} bars")
    print(f"  Kernel:     {os.popen('uname -r').read().strip()}")
    print(f"  CPU pin:    {os.sched_getaffinity(0)}")
    try:
        print(f"  Isolated:   {open('/sys/devices/system/cpu/isolated').read().strip()}")
    except FileNotFoundError:
        print("  Isolated:   (not available)")
    print(f"{'='*60}\n")

    gc.disable()
    gc.collect()

    for i in range(total):
        noise = math.sin(i * 0.1) * 100 + math.cos(i * 0.03) * 50
        price = base + noise
        vol = 100.0 + abs(noise) * 0.5
        ts_epoch = 1700000000.0 + i * 60.0
        ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()

        for si, sym in enumerate(symbols):
            scale = {"BTCUSDT": 1.0, "ETHUSDT": 0.06, "SOLUSDT": 0.002}.get(sym, 0.01)
            sp = price * scale

            event = _MarketEvent(
                header=_Header(ts=ts, event_id=f"b-{sym}-{i}", symbol=sym),
                symbol=sym, close=sp, open=sp * 0.999, high=sp * 1.001,
                low=sp * 0.998, volume=vol, ts=ts,
            )

            measuring = i >= warmup_bars
            if measuring:
                t = TickTiming(bar_idx=i)
                current[0] = t
                t.t_start = _ns()

            coord.emit(event, actor="bench")

            if measuring:
                t.t_end = _ns()
                timings.append(t)
                current[0] = None

    gc.enable()
    coord.stop()

    # Report
    n_sym = len(symbols)
    print(f"Results ({len(timings)} ticks measured, mode={mode_label}):\n")

    if not has_tp:
        feat_ns = [t.t_features - t.t_start for t in timings if t.t_features > 0]
        pipe_ns = [t.t_pipeline - t.t_features for t in timings if t.t_pipeline > 0 and t.t_features > 0]
        dec_ns = [t.t_decision - t.t_pipeline for t in timings if t.t_decision > 0 and t.t_pipeline > 0]
        _report("Features (hook)", feat_ns)
        _report("Pipeline (state)", pipe_ns)
        _report("Decision", dec_ns)
        print("  " + "-" * 85)

    total_ns = [t.t_end - t.t_start for t in timings]
    _report("TOTAL (emit->return)", total_ns)

    # Histogram
    print("\n  Latency distribution (us):")
    total_us = sorted([v / 1000.0 for v in total_ns])
    for b in [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]:
        cnt = sum(1 for v in total_us if v <= b)
        pct = cnt / len(total_us) * 100
        bar = "#" * int(pct / 2)
        print(f"    <= {b:5d} us: {cnt:5d} ({pct:5.1f}%) {bar}")

    # Per-symbol
    if n_sym > 1:
        print("\n  Per-symbol:")
        for si, sym in enumerate(symbols):
            st = timings[si::n_sym]
            _report(f"  {sym}", [t.t_end - t.t_start for t in st])


def run_compare(config_path: Path, num_bars: int, warmup_bars: int) -> None:
    """Run both legacy and tick_processor paths and compare."""
    from infra.config.loader import load_config_secure
    from runner.testnet_validation import _build_ml_stack
    from runner.live_paper_runner import LivePaperRunner, LivePaperConfig
    from event.types import EventType
    from datetime import datetime, timezone

    raw = load_config_secure(config_path)
    trading = raw.get("trading", {})
    symbol = trading.get("symbol", "BTCUSDT")
    symbols = tuple(trading["symbols"]) if "symbols" in trading else (symbol,)

    fc, models, dms, signal_kwargs = _build_ml_stack(raw)
    per_symbol_bridges = signal_kwargs.pop("per_symbol_bridges", None)
    unified_predictors = signal_kwargs.pop("unified_predictors", None)
    tick_processors = signal_kwargs.pop("tick_processors", None)

    if tick_processors is None:
        print("ERROR: No tick_processors available. Need model JSON files.")
        sys.exit(1)

    config = LivePaperConfig(symbols=symbols, starting_balance=10000.0, testnet=True)
    _ns = time.perf_counter_ns

    @dataclass
    class _Header:
        event_type: Any = EventType.MARKET
        ts: float = 0.0
        event_id: Optional[str] = None
        symbol: Optional[str] = None

    @dataclass
    class _MarketEvent:
        EVENT_TYPE: str = "market"
        event_type: Any = EventType.MARKET
        header: Any = None
        symbol: str = "BTCUSDT"
        close: float = 50000.0
        open: float = 49990.0
        high: float = 50050.0
        low: float = 49950.0
        volume: float = 100.0
        ts: float = 0.0

    def _make_events(total: int):
        events = []
        base = 50000.0
        for i in range(total):
            noise = math.sin(i * 0.1) * 100 + math.cos(i * 0.03) * 50
            price = base + noise
            vol = 100.0 + abs(noise) * 0.5
            ts_epoch = 1700000000.0 + i * 60.0
            ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()
            batch = []
            for sym in symbols:
                scale = {"BTCUSDT": 1.0, "ETHUSDT": 0.06, "SOLUSDT": 0.002}.get(sym, 0.01)
                sp = price * scale
                batch.append(_MarketEvent(
                    header=_Header(ts=ts, event_id=f"b-{sym}-{i}", symbol=sym),
                    symbol=sym, close=sp, open=sp * 0.999, high=sp * 1.001,
                    low=sp * 0.998, volume=vol, ts=ts,
                ))
            events.append(batch)
        return events

    total = warmup_bars + num_bars
    events = _make_events(total)

    print(f"\n{'='*60}")
    print("  LATENCY COMPARISON BENCHMARK")
    print(f"{'='*60}")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Warmup:     {warmup_bars} bars")
    print(f"  Measure:    {num_bars} bars")
    print(f"  Kernel:     {os.popen('uname -r').read().strip()}")
    print(f"  CPU pin:    {os.sched_getaffinity(0)}")
    print(f"{'='*60}\n")

    results = {}

    # --- Run 1: Legacy path (unified predictor, no tick_processor) ---
    print("  [1/2] Running LEGACY path (unified predictor)...")
    sig_kw_copy = dict(signal_kwargs)
    runner_legacy = LivePaperRunner.build(
        config,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=None,
        **sig_kw_copy,
    )
    coord = runner_legacy.coordinator
    coord.start()

    gc.disable()
    gc.collect()
    legacy_ns = []
    for i, batch in enumerate(events):
        for ev in batch:
            measuring = i >= warmup_bars
            if measuring:
                t0 = _ns()
            coord.emit(ev, actor="bench")
            if measuring:
                legacy_ns.append(_ns() - t0)
    gc.enable()
    coord.stop()
    results["legacy"] = legacy_ns

    # --- Run 2: Tick processor path ---
    print("  [2/2] Running TICK_PROCESSOR path (full Rust)...")
    sig_kw_copy2 = dict(signal_kwargs)
    runner_tp = LivePaperRunner.build(
        config,
        feature_computer=fc,
        alpha_models=models or None,
        decision_modules=dms or None,
        inference_bridges=per_symbol_bridges,
        unified_predictors=unified_predictors,
        tick_processors=tick_processors,
        **sig_kw_copy2,
    )
    coord2 = runner_tp.coordinator
    coord2.start()

    gc.disable()
    gc.collect()
    tp_ns = []
    for i, batch in enumerate(events):
        for ev in batch:
            measuring = i >= warmup_bars
            if measuring:
                t0 = _ns()
            coord2.emit(ev, actor="bench")
            if measuring:
                tp_ns.append(_ns() - t0)
    gc.enable()
    coord2.stop()
    results["tick_processor"] = tp_ns

    # --- Report ---
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}\n")
    _report("Legacy (unified pred)", legacy_ns)
    _report("TickProcessor (Rust)", tp_ns)

    legacy_p50 = _pct([v / 1000.0 for v in legacy_ns], 50)
    tp_p50 = _pct([v / 1000.0 for v in tp_ns], 50)
    if legacy_p50 > 0:
        speedup = legacy_p50 / tp_p50
        savings = legacy_p50 - tp_p50
        print(f"\n  Speedup: {speedup:.2f}x  (p50 savings: {savings:.0f} us)")

    # Histogram comparison
    print("\n  Distribution comparison (us):")
    print(f"    {'Bucket':>10s}  {'Legacy':>12s}  {'TickProc':>12s}")
    legacy_us = sorted([v / 1000.0 for v in legacy_ns])
    tp_us = sorted([v / 1000.0 for v in tp_ns])
    for b in [50, 100, 200, 300, 500, 1000, 2000]:
        l_cnt = sum(1 for v in legacy_us if v <= b)
        t_cnt = sum(1 for v in tp_us if v <= b)
        l_pct = l_cnt / len(legacy_us) * 100
        t_pct = t_cnt / len(tp_us) * 100
        print(f"    <= {b:5d}:  {l_pct:5.1f}% ({l_cnt:4d})  {t_pct:5.1f}% ({t_cnt:4d})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--compare", action="store_true",
                        help="Run both legacy and tick_processor paths for comparison")
    main = parser.parse_args()
    if main.compare:
        run_compare(main.config, main.bars, main.warmup)
    else:
        run_bench(main.config, main.bars, main.warmup)

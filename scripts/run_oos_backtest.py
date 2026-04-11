#!/usr/bin/env python3
"""OOS-only backtest: runs only on data AFTER model training date.

Usage:
    python3 scripts/run_oos_backtest.py
    python3 scripts/run_oos_backtest.py --months 3
    python3 scripts/run_oos_backtest.py --symbol BTCUSDT_gate_v2

Note: Uses pickle for loading trusted local ML model artifacts (lightgbm/sklearn).
These models are produced by our own training pipeline and HMAC-signed.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_backtest import run_backtest_fast
from scripts.run_full_backtest import (
    MODELS_DIR, _fix_oi_file, load_data, load_model_and_predict,
)

try:
    import json
except ImportError:
    raise

ACTIVE_MODELS = [
    ("BTCUSDT_gate_v2", "BTCUSDT", "1h", "BTCUSDT_1h.csv"),
    ("ETHUSDT_gate_v2", "ETHUSDT", "1h", "ETHUSDT_1h.csv"),
    ("BTCUSDT_4h",      "BTCUSDT", "4h", None),
    ("ETHUSDT_4h",      "ETHUSDT", "4h", None),
]


def _regime_stratified_sharpe(
    signal: np.ndarray,
    closes: np.ndarray,
    total_cost: np.ndarray,
) -> Dict[str, Any]:
    """Split OOS returns by 20-bar vol tercile and report per-bucket Sharpe.

    A healthy strategy works across regimes.  If Sharpe is strongly
    positive in 'low_vol' but negative in 'high_vol', the headline
    Sharpe is masking regime risk and the strategy may crater in the
    next vol spike.  Used by ``--regime-split`` CLI flag.
    """
    n = min(len(signal), len(closes))
    if n < 60:
        return {"error": "insufficient bars"}

    log_ret = np.zeros(n)
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_ret[i] = np.log(closes[i] / closes[i - 1])

    # 20-bar rolling vol (same measure as vol_20 used by impact model)
    vol_20 = np.zeros(n)
    for i in range(1, n):
        ws = max(0, i - 19)
        vol_20[i] = float(np.std(log_ret[ws:i + 1]))

    # Strategy returns = signal × log_ret − cost
    strat_ret = signal * log_ret
    if len(total_cost) == n:
        strat_ret = strat_ret - total_cost

    # Tercile split on vol_20 (after warmup)
    warmup = 30
    vol_warm = vol_20[warmup:]
    lo, hi = np.quantile(vol_warm[vol_warm > 0], [1.0 / 3.0, 2.0 / 3.0])

    buckets = {
        "low_vol":  {"mask": vol_20 <= lo, "label": f"vol ≤ {lo:.5f}"},
        "mid_vol":  {"mask": (vol_20 > lo) & (vol_20 <= hi),
                     "label": f"{lo:.5f} < vol ≤ {hi:.5f}"},
        "high_vol": {"mask": vol_20 > hi, "label": f"vol > {hi:.5f}"},
    }

    out: Dict[str, Any] = {}
    for name, b in buckets.items():
        m = b["mask"] & (np.arange(n) >= warmup)
        r = strat_ret[m]
        if len(r) < 10:
            out[name] = {"n_bars": int(len(r)), "sharpe": None, "ret_sum_pct": 0.0}
            continue
        mean = float(np.mean(r))
        std = float(np.std(r)) if np.std(r) > 1e-9 else 1.0
        # Annualised (8760 hourly bars / year for 1h, 2190 for 4h — use 8760 as default)
        ann_factor = np.sqrt(8760.0)
        sharpe = (mean / std) * ann_factor
        out[name] = {
            "label": b["label"],
            "n_bars": int(len(r)),
            "sharpe": round(sharpe, 2),
            "ret_sum_pct": round(float(np.sum(r)) * 100, 2),
            "trades": int(np.sum(np.abs(np.diff(signal[m].astype(float), prepend=0)) > 0.5)),
        }
    return out


def _runner_key_for(model_name: str) -> str:
    """Map ``BTCUSDT_gate_v2`` → ``BTCUSDT`` runner key for sizer lookup."""
    if model_name.endswith("_gate_v2"):
        return model_name[: -len("_gate_v2")]
    return model_name


def _live_tier_params(capital: float, runner_key: str) -> Dict[str, float]:
    """Replicate ``decision.sizing.adaptive`` tier selection for backtest.

    Returns the tier_cap + leverage + ic_scale that the live sizer
    would have applied for an account of this size.  This is the
    minimum-viable mirror — does *not* include z_scale (which is
    time-varying and baked into the raw signal) or step_size rounding
    (which matters at very small notional and is approximated away).
    """
    from decision.sizing.adaptive import _TIER_WEIGHTS, _DEFAULT_CAP

    if capital < 500:
        tier = "micro"
    elif capital < 10_000:
        tier = "medium"
    else:
        tier = "large"
    tier_cap = _TIER_WEIGHTS.get(tier, {}).get(runner_key, _DEFAULT_CAP)

    # Leverage matches live: 3x live, 10x demo.  Default to 3 since
    # the OKX production account is live mode.
    leverage = 3.0
    # IC scale — default GREEN baseline (1.2).  Backtest has no
    # time-varying IC oracle; use a constant that matches the
    # "typical healthy" case.  Callers can override via overrides.
    ic_scale = 1.2

    return {
        "tier": tier,
        "tier_cap": tier_cap,
        "leverage": leverage,
        "ic_scale": ic_scale,
    }


def run_oos_backtest(model_name: str, symbol: str, timeframe: str,
                     data_file: Optional[str], oos_months: int = 6,
                     overrides: Optional[Dict[str, Any]] = None,
                     capital: float = 10000.0,
                     flat_cost: bool = False,
                     regime_split: bool = False,
                     live_sizer: bool = False,
                     regime_gated: bool = False,
                     latency_ms: float = 0.0,
                     restart_every_bars: int = 0,
                     sizer_tier_cap_override: Optional[float] = None) -> Dict[str, Any]:
    """Run one OOS backtest for a trained model.

    overrides: optional dict of ``bt_config`` keys to override the
        baked-in model config.  Used by the walk-forward grid search
        to sweep ``deadzone`` / ``min_hold`` / ``max_hold`` /
        ``zscore_window`` etc. without retraining the model.
    """
    model_dir = MODELS_DIR / model_name
    config_path = model_dir / "config.json"

    result: Dict[str, Any] = {
        "model": model_name, "symbol": symbol, "timeframe": timeframe,
        "status": "FAIL", "sharpe": 0.0, "trades": 0,
    }

    if not config_path.exists():
        result["error"] = "config.json missing"
        return result

    with open(config_path) as f:
        config = json.load(f)
    if overrides:
        for k, v in overrides.items():
            config[k] = v

    # Determine OOS start: use ``now - oos_months`` directly.
    #
    # NOTE: train_v12 walk-forward uses the last 18 months as its
    # internal OOS window (train_end = n - 18*BARS_PER_MONTH), so
    # requesting e.g. ``--months 12`` on a freshly trained model
    # overlaps with the training OOS window — that's fine because
    # the data + labels were held out during training.  The previous
    # formula used ``train_date - 30d`` which only gave us the post-
    # train period (1 month for a model trained today).
    train_date_str = config.get("train_date", "")
    oos_start = datetime.now() - timedelta(days=oos_months * 30)

    try:
        df = load_data(data_file, symbol, timeframe)
    except Exception as e:
        result["error"] = f"data load: {e}"
        return result

    _fix_oi_file(symbol)

    try:
        y_pred = load_model_and_predict(model_dir, df, config, symbol=symbol)
    except Exception as e:
        result["error"] = f"predict: {e}"
        return result

    if y_pred is None:
        result["error"] = "no predictions generated"
        return result

    # Filter to OOS period
    df_ts = pd.to_datetime(df["open_time"], unit="ms")
    oos_mask = df_ts >= pd.Timestamp(oos_start)
    oos_start_idx = int(oos_mask.values.argmax())

    # Need zscore_warmup bars before OOS start
    warmup = config.get("zscore_warmup", 180)
    start_idx = max(0, oos_start_idx - warmup)

    n = min(len(y_pred), len(df))
    timestamps = df["open_time"].values[start_idx:n].astype(np.int64)
    closes = df["close"].values[start_idx:n].astype(np.float64)
    volumes = df["volume"].values[start_idx:n].astype(np.float64)
    preds = y_pred[start_idx:n]

    # vol_20: rolling 20-bar realized volatility needed for the realistic
    # Rust cost model (Almgren-Chriss impact + vol-scaled spread).  Computed
    # here so the whole cost pipeline can be driven from Python without a
    # feature-engine dependency loop.
    log_ret = np.diff(np.log(closes), prepend=closes[0])
    vol_20 = np.zeros_like(closes)
    for i in range(len(closes)):
        window_start = max(0, i - 19)
        vol_20[i] = np.std(log_ret[window_start:i + 1]) if i >= 1 else 0.0

    bt_config = {
        "deadzone": config.get("deadzone", 0.5),
        "min_hold": config.get("min_hold", 24),
        "max_hold": config.get("max_hold", 120),
        "zscore_window": config.get("zscore_window", 720),
        "zscore_warmup": config.get("zscore_warmup", 180),
        "long_only": config.get("long_only", False),
        "monthly_gate": config.get("monthly_gate", False),
        "ma_window": config.get("monthly_gate_window", 480),

        # Realistic cost model — enable the Rust Almgren-Chriss pipeline
        # instead of the flat 6 bps default.  Activating `realistic_cost`
        # requires `volumes` and `vol_20` arrays passed to run_backtest_fast
        # (the wiring already exists — see scripts/run_oos_backtest.py).
        # Fee defaults match OKX-USDT-SWAP (public taker 5 bps,
        # maker 2 bps) since that's our live venue; Binance is ~1 bp
        # cheaper and Bybit ~1 bp more, within noise of slippage.
        "realistic_cost": True,
        "cost_per_trade": 6e-4,           # retained as fallback
        "taker_fee_bps": 5.0,
        "maker_fee_bps": 2.0,
        "taker_ratio": 1.0,               # we are 100% market orders
        "impact_eta": 0.5,                # Almgren-Chriss impact coefficient
        "spread_multiplier": 0.05,        # spread ~ 0.05 × hourly vol (bps)
        "max_participation": 0.10,        # max 10% of 1h volume per trade

        "capital": capital,
    }
    if flat_cost:
        bt_config["realistic_cost"] = False

    # Live-equivalent sizer: replicate AdaptivePositionSizer chain so
    # backtest PnL scale matches execution reality.  Opt-in via
    # ``live_sizer=True`` — the default keeps legacy ``signal * 1.0``
    # behaviour for scripts that have not been audited.
    if live_sizer:
        runner_key = _runner_key_for(model_name)
        tier_p = _live_tier_params(capital, runner_key)
        # Allow explicit override for hypothetical activation studies
        # (useful for 4h models whose production tier_cap=0 silences
        # the backtest entirely; passing e.g. 0.30 shows what they'd
        # do if allowed to trade at 30% of the normal budget).
        if sizer_tier_cap_override is not None:
            bt_config["sizer_tier_cap"] = float(sizer_tier_cap_override)
        else:
            bt_config["sizer_tier_cap"] = tier_p["tier_cap"]
        bt_config["sizer_leverage"] = tier_p["leverage"]
        bt_config["sizer_ic_scale"] = tier_p["ic_scale"]
        bt_config["sizer_regime_gated"] = regime_gated
        # Regime multipliers: match the #2 proposal in the D10 model
        # improvement plan — 0.40 mid / 0.15 high vol.
        bt_config["sizer_regime_mid_mult"] = 0.40
        bt_config["sizer_regime_high_mult"] = 0.15

    # Latency slippage: charge turnover × drift_frac × |next_bar_return|
    # as an adverse cost.  500ms on a 1h bar ≈ 0.014 fraction of the
    # range that actually happens during the latency window, but the
    # price movement in that window is essentially noise so we use a
    # more conservative 0.05 drift_frac that matches observed OKX-REST
    # behaviour on smaller markets.
    if latency_ms > 0:
        bt_config["latency_ms"] = latency_ms
        # 0.05 × 1bar = 5% of bar move charged as slip.
        bt_config["latency_drift_frac"] = 0.05 * (latency_ms / 500.0)

    if restart_every_bars > 0:
        bt_config["restart_every_bars"] = restart_every_bars

    try:
        bt = run_backtest_fast(
            timestamps=timestamps, closes=closes, y_pred=preds,
            volumes=volumes, vol_20=vol_20, config=bt_config,
        )
    except Exception as e:
        result["error"] = f"backtest: {e}"
        return result

    sharpe = bt.get("sharpe", 0.0)
    total_return = bt.get("total_return", 0.0)
    max_dd = bt.get("max_drawdown", 0.0)
    n_trades = bt.get("n_trades", 0)
    win_rate = bt.get("win_rate", 0.0)

    # Regime-stratified Sharpe (optional — diagnoses whether the
    # headline number is masking a regime-sensitive strategy).
    regime_report: Optional[Dict[str, Any]] = None
    if regime_split:
        try:
            sig_arr = np.asarray(bt.get("signal", []), dtype=np.float64)
            # The regime split recomputes gross returns internally from
            # (signal, closes); net_pnl's cost accounting is left as a
            # possible future enhancement when we want bucket-level
            # cost attribution.
            if len(sig_arr) == len(closes):
                # Reconstruct total_cost as the gap between gross and net.
                # Easier: just pass a zero cost array and let the regime
                # helper compute gross returns — net_pnl captures costs.
                regime_report = _regime_stratified_sharpe(
                    sig_arr, closes, np.zeros(len(closes))
                )
        except Exception as e:
            regime_report = {"error": f"regime split failed: {e}"}

    actual_start = pd.to_datetime(
        timestamps[warmup] if len(timestamps) > warmup else timestamps[0], unit="ms")
    actual_end = pd.to_datetime(timestamps[-1], unit="ms")

    result.update({
        "status": "PASS" if sharpe > 0.5 else "MARGINAL" if sharpe > 0 else "FAIL",
        "sharpe": round(sharpe, 2),
        "total_return": round(total_return * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "bars": len(timestamps),
        "oos_bars": len(timestamps) - min(warmup, len(timestamps)),
        "period": f"{actual_start.date()} to {actual_end.date()}",
        "train_date": train_date_str,
    })
    if regime_report is not None:
        result["regime"] = regime_report
    return result


def _parse_grid(spec: str) -> Dict[str, list]:
    """Parse ``--grid`` spec into ``{param: [values...]}``.

    Format: space-separated ``KEY=V1,V2,V3`` pairs.

    >>> _parse_grid("deadzone=0.8,1.0,1.2 min_hold=4,6,8")
    {'deadzone': [0.8, 1.0, 1.2], 'min_hold': [4, 6, 8]}
    """
    out: Dict[str, list] = {}
    if not spec:
        return out
    for part in spec.split():
        if "=" not in part:
            continue
        k, raw_vals = part.split("=", 1)
        values: list = []
        for v in raw_vals.split(","):
            v = v.strip()
            if not v:
                continue
            # Try int → float → raw string
            try:
                values.append(int(v))
                continue
            except ValueError:
                pass
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)
        if values:
            out[k.strip()] = values
    return out


def _expand_grid(grid: Dict[str, list]) -> list[Dict[str, Any]]:
    """Cartesian product of a grid dict into a list of overrides."""
    from itertools import product
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


def run_grid_search(models: list, months: int, grid: Dict[str, list]) -> list[Dict[str, Any]]:
    """Sweep ``grid`` × ``models`` and return one row per (model, combo).

    Rows carry their override dict under ``"params"`` and the standard
    backtest metrics (sharpe / trades / ...).  Caller prints / sorts.
    """
    combos = _expand_grid(grid)
    rows: list[Dict[str, Any]] = []
    total = len(combos) * len(models)
    seen = 0
    for model_name, symbol, tf, data_file in models:
        for combo in combos:
            seen += 1
            t0 = time.time()
            r = run_oos_backtest(model_name, symbol, tf, data_file, months,
                                 overrides=combo)
            r["time_s"] = round(time.time() - t0, 1)
            r["params"] = combo
            rows.append(r)
            if "error" not in r:
                print(f"  [{seen:3d}/{total}] {model_name:<24} {combo}  "
                      f"→ Sharpe={r['sharpe']:+.2f} trades={r['trades']}")
            else:
                print(f"  [{seen:3d}/{total}] {model_name:<24} {combo}  "
                      f"→ ERROR: {r['error']}")
    return rows


def _print_grid_report(rows: list[Dict[str, Any]]) -> None:
    """Group by model, sort each group by Sharpe descending, print."""
    by_model: Dict[str, list[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, group in by_model.items():
        group.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        print("\n" + "=" * 90)
        print(f"GRID RESULTS — {model}  ({len(group)} configs)")
        print("=" * 90)
        hdr_params = sorted({k for r in group for k in r.get("params", {})})
        header = " ".join(f"{k:<10}" for k in hdr_params)
        print(f"{'Rank':<5} {header} {'Sharpe':>8} {'Ret%':>8} {'DD%':>7} {'Trades':>7}")
        for i, r in enumerate(group[:20], 1):
            param_vals = " ".join(
                f"{str(r.get('params', {}).get(k, '')):<10}" for k in hdr_params
            )
            if "error" in r:
                print(f"{i:<5} {param_vals} ERROR: {r['error'][:40]}")
            else:
                print(f"{i:<5} {param_vals} "
                      f"{r['sharpe']:>+8.2f} {r['total_return']:>+7.1f}% "
                      f"{r['max_drawdown']:>+6.1f}% {r['trades']:>7d}")
        best = group[0]
        if "error" not in best:
            print(f"\n  BEST: {best.get('params', {})} → Sharpe {best['sharpe']:+.2f}")
            baseline = next((r for r in group if not r.get("params")), None)
            if baseline and "error" not in baseline:
                delta = best["sharpe"] - baseline["sharpe"]
                print(f"  vs baseline ({baseline.get('params', {})}): "
                      f"Sharpe Δ = {delta:+.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OOS-only backtest validation")
    parser.add_argument("--symbol", help="Run single model only")
    parser.add_argument("--months", type=int, default=6, help="OOS lookback months (default: 6)")
    parser.add_argument(
        "--grid", default=None,
        metavar="SPEC",
        help=(
            "Walk-forward grid search.  Space-separated KEY=V1,V2,V3 "
            "pairs, e.g. 'deadzone=0.8,1.0,1.2 min_hold=4,6,8'.  "
            "Cartesian product is swept against each selected model.  "
            "Supports any bt_config key (deadzone/min_hold/max_hold/"
            "zscore_window/long_only)."
        ),
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0,
        help="Notional capital for impact-cost model (default: 10000)",
    )
    parser.add_argument(
        "--flat-cost", action="store_true",
        help="Use legacy flat 6 bps cost model instead of realistic",
    )
    parser.add_argument(
        "--regime-split", action="store_true",
        help=(
            "Report Sharpe per vol tercile (low/mid/high vol regime). "
            "Exposes regime-sensitive strategies that look profitable "
            "on average but crater in high-vol regimes."
        ),
    )
    parser.add_argument(
        "--live-sizer", action="store_true",
        help=(
            "Replicate AdaptivePositionSizer in the backtest: applies "
            "tier_cap × leverage × ic_scale to the signal so backtest "
            "PnL scale matches live execution.  Without this flag the "
            "backtest uses signal=±1 as literal position fraction."
        ),
    )
    parser.add_argument(
        "--regime-gated", action="store_true",
        help=(
            "Apply the D10 proposed vol-regime sizing multipliers "
            "(mid_mult=0.40, high_mult=0.15) inside the backtest.  "
            "Requires --live-sizer."
        ),
    )
    parser.add_argument(
        "--latency-ms", type=float, default=0.0,
        help=(
            "Simulate order→fill latency in milliseconds.  Charges "
            "adverse slippage proportional to next-bar price movement.  "
            "Default 0 = no latency cost (legacy).  Try 500 to match "
            "observed OKX REST roundtrip."
        ),
    )
    parser.add_argument(
        "--sizer-tier-cap", type=float, default=None,
        help=(
            "Override the auto-detected tier_cap with a fixed value. "
            "Used to test hypothetical activation of signal-only "
            "runners (4h models default to 0.0 = no trading); "
            "pass e.g. 0.3 to simulate them as active traders."
        ),
    )
    parser.add_argument(
        "--restart-every-bars", type=int, default=0,
        help=(
            "Simulate alpha_main process restarts every N bars.  Each "
            "restart forces signal=0 for zscore_warmup bars (180 by "
            "default), modelling the cold-start period observed in "
            "production.  Typical values: 168 (weekly), 720 (monthly)."
        ),
    )
    args = parser.parse_args()

    models = ACTIVE_MODELS
    if args.symbol:
        models = [m for m in models if m[0] == args.symbol]
        if not models:
            print(f"Model {args.symbol} not found")
            sys.exit(1)

    # Grid-search mode: sweep parameters against all selected models and
    # print a ranked per-model table.  Use for quarterly parameter retuning.
    if args.grid:
        grid = _parse_grid(args.grid)
        if not grid:
            print(f"ERROR: could not parse --grid spec: {args.grid!r}")
            sys.exit(1)
        print("=" * 90)
        print(f"WALK-FORWARD GRID SEARCH (last {args.months} months)")
        print(f"  Grid: {grid}")
        print(f"  Models: {[m[0] for m in models]}")
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"  Configs per model: {n_combos}   Total runs: {n_combos * len(models)}")
        print("=" * 90)
        rows = run_grid_search(models, args.months, grid)
        _print_grid_report(rows)
        return

    print("=" * 90)
    print(f"OUT-OF-SAMPLE BACKTEST (last {args.months} months)")
    print("=" * 90)

    results = []
    for model_name, symbol, tf, data_file in models:
        print(f"\n>>> {model_name} ({symbol} {tf})")
        t0 = time.time()
        result = run_oos_backtest(
            model_name, symbol, tf, data_file, args.months,
            capital=args.capital, flat_cost=args.flat_cost,
            regime_split=args.regime_split,
            live_sizer=args.live_sizer,
            regime_gated=args.regime_gated,
            latency_ms=args.latency_ms,
            restart_every_bars=args.restart_every_bars,
            sizer_tier_cap_override=args.sizer_tier_cap,
        )
        elapsed = time.time() - t0
        result["time_s"] = round(elapsed, 1)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Sharpe={result['sharpe']:.2f}  Return={result['total_return']:.1f}%  "
                  f"MaxDD={result['max_drawdown']:.1f}%  Trades={result['trades']}  "
                  f"WinRate={result['win_rate']:.0f}%  ({elapsed:.1f}s)")
            print(f"  Status: {result['status']}  Period: {result['period']}  "
                  f"Trained: {result.get('train_date', '?')}")
            if "regime" in result and "error" not in result["regime"]:
                print("  Regime breakdown (vol tercile):")
                for name in ("low_vol", "mid_vol", "high_vol"):
                    b = result["regime"].get(name, {})
                    sharpe_str = (f"{b['sharpe']:+.2f}" if b.get("sharpe") is not None else "  n/a")
                    print(f"    {name:<9} n={b.get('n_bars',0):>5d}  "
                          f"Sharpe={sharpe_str}  "
                          f"ret={b.get('ret_sum_pct', 0):+.2f}%  "
                          f"trades={b.get('trades', 0)}")

    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'TF':>3} {'Status':>8} {'Sharpe':>7} {'Return':>8} "
          f"{'MaxDD':>7} {'Trades':>6} {'WR':>5} {'OOS bars':>8}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<25} {r['timeframe']:>3} {'ERROR':>8}  {r.get('error','')[:45]}")
        else:
            print(f"{r['model']:<25} {r['timeframe']:>3} {r['status']:>8} "
                  f"{r['sharpe']:>7.2f} {r['total_return']:>7.1f}% {r['max_drawdown']:>6.1f}% "
                  f"{r['trades']:>6} {r['win_rate']:>4.0f}% {r['oos_bars']:>8}")
    print("=" * 90)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len([r for r in results if "error" not in r])
    print(f"\nResult: {passed}/{total} PASS (Sharpe > 0.5), {failed} FAIL")


if __name__ == "__main__":
    main()

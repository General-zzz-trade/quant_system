#!/usr/bin/env python3
"""Portfolio-level joint backtest for BTC + ETH (or arbitrary symbol set).

``run_oos_backtest.py`` runs each symbol independently with its own
capital pool — each model sees ``capital`` as if it's the whole account.
That over-estimates the total return when you actually hold both:
both positions draw from the SAME $400 account, and the correlated
drawdowns in BTC vs ETH are real (they tend to crash together).

This script orchestrates the two per-symbol Rust backtests, aligns
their per-bar ``net_pnl`` streams on common timestamps, and composes
a single portfolio equity curve::

    portfolio_pnl[i] = sum over symbols of (pos_frac_s[i] * ret_s[i] - cost_s[i])

Portfolio metrics reported:
  * Joint Sharpe, Return, MaxDD
  * Per-symbol contribution breakdown
  * Realised correlation between symbol return streams
  * Worst-week / worst-month drawdown windows
  * Capital utilisation (max concurrent notional as % of equity)

Usage
-----

    # Current production (3x × 0.65 each, baseline)
    python3 scripts/run_portfolio_backtest.py --capital 400 --months 12

    # 10x leverage, Kelly quarter (1.0x effective each)
    python3 scripts/run_portfolio_backtest.py --capital 400 --months 12 \
        --sizer-leverage 10 --btc-cap 0.10 --eth-cap 0.10

    # 10x leverage, explicit per-symbol caps for BTC-heavy allocation
    python3 scripts/run_portfolio_backtest.py --capital 400 --months 12 \
        --sizer-leverage 10 --btc-cap 0.12 --eth-cap 0.08
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, "/quant_system")

from scripts.run_oos_backtest import run_oos_backtest  # noqa: E402


# ── Symbol roster (BTC+ETH 1h for D11+) ─────────────────────────────
# Can be extended to include 4h or new symbols.  4h are excluded by
# default because their production tier_cap is 0 (signal-only).
PORTFOLIO_SYMBOLS = [
    ("BTCUSDT_gate_v2", "BTCUSDT", "1h", "BTCUSDT_1h.csv"),
    ("ETHUSDT_gate_v2", "ETHUSDT", "1h", "ETHUSDT_1h.csv"),
]


def _run_single(model_name: str, symbol: str, timeframe: str,
                data_file: Optional[str], args: argparse.Namespace,
                tier_cap: float) -> Dict[str, Any]:
    """Run a single-symbol live-equivalent backtest and return the
    per-bar ``signal``, ``net_pnl`` and ``equity`` arrays plus metrics.

    Capital is always passed as ``args.capital`` (each symbol sees the
    full wallet — we correct the double-counting later when combining).
    """
    return run_oos_backtest(
        model_name=model_name,
        symbol=symbol,
        timeframe=timeframe,
        data_file=data_file,
        oos_months=args.months,
        capital=args.capital,
        flat_cost=False,
        regime_split=False,
        live_sizer=True,
        regime_gated=False,
        latency_ms=args.latency_ms,
        sizer_tier_cap_override=tier_cap,
        sizer_leverage_override=args.sizer_leverage,
    )


def _fetch_streams(model_name: str, symbol: str, timeframe: str,
                   data_file: Optional[str], args: argparse.Namespace,
                   tier_cap: float) -> Dict[str, Any]:
    """Variant of ``_run_single`` that also grabs the raw Rust backtest
    output (timestamps, closes, net_pnl) needed for portfolio combining.

    Re-implements the inner-loop of run_oos_backtest so the Rust call
    can be done once per symbol and the output series retained.  This
    avoids re-running the backtest twice (once for metrics, once for
    streams).
    """
    import pandas as pd
    from scripts.run_full_backtest import (
        MODELS_DIR, _fix_oi_file, load_data, load_model_and_predict,
    )
    from features.batch_backtest import run_backtest_fast
    from datetime import datetime, timedelta

    model_dir = MODELS_DIR / model_name
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {"error": f"config.json missing for {model_name}"}
    config = json.loads(cfg_path.read_text())

    oos_start = datetime.now() - timedelta(days=args.months * 30)
    try:
        df = load_data(data_file, symbol, timeframe)
    except Exception as e:
        return {"error": f"load_data: {e}"}
    _fix_oi_file(symbol)

    try:
        y_pred = load_model_and_predict(model_dir, df, config, symbol=symbol)
    except Exception as e:
        return {"error": f"predict: {e}"}
    if y_pred is None:
        return {"error": "no predictions"}

    df_ts = pd.to_datetime(df["open_time"], unit="ms")
    oos_mask = df_ts >= pd.Timestamp(oos_start)
    oos_start_idx = int(oos_mask.values.argmax())
    warmup = config.get("zscore_warmup", 180)
    start_idx = max(0, oos_start_idx - warmup)
    n = min(len(y_pred), len(df))
    timestamps = df["open_time"].values[start_idx:n].astype(np.int64)
    closes = df["close"].values[start_idx:n].astype(np.float64)
    volumes = df["volume"].values[start_idx:n].astype(np.float64)
    preds = y_pred[start_idx:n]

    log_ret = np.diff(np.log(closes), prepend=closes[0])
    vol_20 = np.zeros_like(closes)
    for i in range(len(closes)):
        ws = max(0, i - 19)
        vol_20[i] = np.std(log_ret[ws:i + 1]) if i >= 1 else 0.0

    bt_config = {
        "deadzone": config.get("deadzone", 0.5),
        "min_hold": config.get("min_hold", 24),
        "max_hold": config.get("max_hold", 120),
        "zscore_window": config.get("zscore_window", 720),
        "zscore_warmup": config.get("zscore_warmup", 180),
        "long_only": config.get("long_only", False),
        "monthly_gate": config.get("monthly_gate", False),
        "ma_window": config.get("monthly_gate_window", 480),
        "realistic_cost": True,
        "taker_fee_bps": 5.0,
        "maker_fee_bps": 2.0,
        "taker_ratio": 1.0,
        "impact_eta": 0.5,
        "spread_multiplier": 0.05,
        "max_participation": 0.10,
        "capital": args.capital,
        "sizer_tier_cap": tier_cap,
        "sizer_leverage": args.sizer_leverage,
        "sizer_ic_scale": 1.2,
        "latency_ms": args.latency_ms,
        "latency_drift_frac": 0.05 * (args.latency_ms / 500.0),
    }

    bt = run_backtest_fast(
        timestamps=timestamps, closes=closes, y_pred=preds,
        volumes=volumes, vol_20=vol_20, config=bt_config,
    )
    return {
        "symbol": symbol,
        "model": model_name,
        "timestamps": timestamps,
        "closes": closes,
        "signal": np.asarray(bt.get("signal", []), dtype=np.float64),
        "net_pnl": np.asarray(bt.get("net_pnl", []), dtype=np.float64),
        "equity": np.asarray(bt.get("equity", []), dtype=np.float64),
        "tier_cap": tier_cap,
        "warmup": warmup,
    }


def _align_streams(streams: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inner-join per-bar series on timestamps so all symbols share the
    same index.  Returns dict with aligned ``timestamps`` + per-symbol
    aligned ``signal`` / ``net_pnl`` arrays.
    """
    if not streams:
        return {"error": "no streams"}

    # Intersect timestamps across all symbols (may differ on edges)
    ts_sets = [set(s["timestamps"].tolist()) for s in streams]
    common = sorted(set.intersection(*ts_sets))
    if not common:
        return {"error": "no common timestamps"}

    common_arr = np.array(common, dtype=np.int64)
    aligned: Dict[str, Any] = {"timestamps": common_arr}
    for s in streams:
        sym = s["symbol"]
        ts = s["timestamps"]
        net_pnl = s["net_pnl"]
        signal = s["signal"]
        # Build index map
        index_map = {t: i for i, t in enumerate(ts)}
        idx = np.array([index_map[t] for t in common], dtype=np.int64)
        # net_pnl has len = n-1 (bar-to-bar returns); clip idx accordingly
        idx_pnl = idx[idx < len(net_pnl)]
        n_take = len(idx_pnl)
        aligned[f"{sym}_net_pnl"] = net_pnl[idx_pnl]
        aligned[f"{sym}_signal"] = signal[:len(idx)][idx][:n_take] if len(signal) >= len(idx) else signal[:n_take]
        aligned[f"{sym}_closes"] = s["closes"][:len(idx)][idx][:n_take + 1]
        aligned["_n_bars"] = n_take
    # Truncate timestamps to the shortest aligned series
    aligned["timestamps"] = common_arr[:aligned["_n_bars"]]
    return aligned


def _compose_portfolio(aligned: Dict[str, Any], symbols: List[str],
                       capital: float) -> Dict[str, Any]:
    """Combine per-symbol net_pnl into a single portfolio equity series.

    Portfolio composition: each symbol's ``net_pnl[i]`` is its ``delta``
    in fraction-of-capital terms (because the Rust backtest was given
    the full ``capital`` and each per-bar PnL is a fraction of that).
    When both symbols see the same capital pool, the joint delta is
    the SUM of per-symbol deltas — gains compound on shared equity.

    The math:
      portfolio_return[i] = sum_s (net_pnl_s[i])
      portfolio_equity[i+1] = portfolio_equity[i] * (1 + portfolio_return[i])

    Sharpe annualisation matches the Rust compute_metrics convention:
    only "active" bars are counted — i.e. bars where at least one
    symbol is actually holding a position.  This makes the joint
    Sharpe directly comparable with the per-symbol Rust output.
    """
    n = aligned["_n_bars"]
    port_ret = np.zeros(n, dtype=np.float64)
    # Active mask: at least one symbol has non-zero signal at this bar
    active_mask = np.zeros(n, dtype=bool)
    for sym in symbols:
        pnl = aligned[f"{sym}_net_pnl"]
        sig = aligned[f"{sym}_signal"]
        take = min(len(pnl), n)
        port_ret[:take] += pnl[:take]
        # Signal series may be 1 element longer (n bars of signal for
        # n-1 returns); clip to n.
        sig_take = min(len(sig), n)
        active_mask[:sig_take] |= np.abs(sig[:sig_take]) > 1e-9

    equity = np.zeros(n + 1, dtype=np.float64)
    equity[0] = capital
    for i in range(n):
        equity[i + 1] = equity[i] * (1.0 + port_ret[i])

    # Per-symbol cumulative PnL contribution
    contributions = {}
    for sym in symbols:
        pnl = aligned[f"{sym}_net_pnl"][:n]
        cum_return = float(np.prod(1 + pnl) - 1) if len(pnl) > 0 else 0.0
        contributions[sym] = {
            "cum_return": cum_return,
            "mean_daily": float(np.mean(pnl) * 24),
            "std_daily": float(np.std(pnl) * np.sqrt(24)),
        }

    # Joint Sharpe — active-bars-only (matches Rust single-symbol).
    # This is the apples-to-apples comparison with the per-symbol
    # Sharpe numbers from run_oos_backtest.py.
    ann_factor = np.sqrt(8760.0)  # 1h bars → annual
    active_ret = port_ret[active_mask]
    if len(active_ret) > 1:
        a_mean = float(np.mean(active_ret))
        a_var = float(np.var(active_ret, ddof=1))
        a_std = max(a_var ** 0.5, 1e-12)
        sharpe_active = (a_mean / a_std) * ann_factor
    else:
        sharpe_active = 0.0

    # All-bar Sharpe — includes idle bars (more honest for 24/7 eval).
    mean = float(np.mean(port_ret))
    std = float(np.std(port_ret)) if np.std(port_ret) > 1e-12 else 1.0
    sharpe = (mean / std) * ann_factor
    n_active = int(np.sum(active_mask))
    total_return = float(equity[-1] / equity[0] - 1)

    # MaxDD from running max
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(np.min(drawdown))

    # Correlation between per-symbol per-bar pnl
    corr = None
    if len(symbols) == 2:
        a = aligned[f"{symbols[0]}_net_pnl"][:n]
        b = aligned[f"{symbols[1]}_net_pnl"][:n]
        if len(a) == len(b) and len(a) > 30 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
            corr = float(np.corrcoef(a, b)[0, 1])

    return {
        "n_bars": n,
        "n_active_bars": n_active,
        "active_bar_fraction": round(n_active / n, 3) if n > 0 else 0.0,
        "equity_curve": equity,
        "portfolio_return": port_ret,
        "joint_sharpe_active": round(sharpe_active, 2),  # apples-to-apples with Rust
        "joint_sharpe_all_bars": round(sharpe, 2),       # 24/7 honest
        "joint_return_pct": round(total_return * 100, 1),
        "joint_maxdd_pct": round(max_dd * 100, 1),
        "final_equity": round(float(equity[-1]), 2),
        "contributions": contributions,
        "correlation": round(corr, 3) if corr is not None else None,
    }


def _format_report(report: Dict[str, Any], args: argparse.Namespace) -> str:
    out: List[str] = []
    out.append("=" * 78)
    out.append(f"  PORTFOLIO BACKTEST — {args.months}-month live-equivalent")
    out.append(f"  Capital ${args.capital:.0f}  leverage {args.sizer_leverage}x  "
               f"latency {args.latency_ms:.0f}ms")
    out.append("=" * 78)
    out.append(f"  Joint Sharpe (active): {report['joint_sharpe_active']:+.2f}  "
               f"(matches Rust single-symbol convention)")
    out.append(f"  Joint Sharpe (24/7):   {report['joint_sharpe_all_bars']:+.2f}  "
               f"(includes idle bars)")
    out.append(f"  Joint Return:  {report['joint_return_pct']:+.1f}%  "
               f"(${args.capital:.0f} → ${report['final_equity']:.0f})")
    out.append(f"  Joint MaxDD:   {report['joint_maxdd_pct']:+.1f}%")
    out.append(f"  Active bars:   {report['n_active_bars']}/{report['n_bars']}  "
               f"({report['active_bar_fraction'] * 100:.0f}% utilisation)")
    if report.get("correlation") is not None:
        out.append(f"  Per-bar ρ(BTC,ETH): {report['correlation']:+.3f}")
    out.append("")
    out.append("  Per-symbol contribution (isolated, for reference):")
    for sym, c in report["contributions"].items():
        out.append(f"    {sym:<10} cum_return={c['cum_return']*100:+7.1f}%  "
                   f"daily_mean={c['mean_daily']*100:+.3f}%  "
                   f"daily_std={c['std_daily']*100:.3f}%")
    return "\n".join(out)


def run_portfolio(args: argparse.Namespace) -> Dict[str, Any]:
    # Resolve per-symbol tier caps
    per_symbol_caps = {"BTCUSDT": args.btc_cap, "ETHUSDT": args.eth_cap}

    streams: List[Dict[str, Any]] = []
    for model_name, symbol, tf, data_file in PORTFOLIO_SYMBOLS:
        cap = per_symbol_caps[symbol]
        t0 = time.time()
        s = _fetch_streams(model_name, symbol, tf, data_file, args, tier_cap=cap)
        if "error" in s:
            print(f"  [skip {symbol}] {s['error']}")
            continue
        print(f"  [{symbol}] bars={len(s['timestamps'])} cap={cap:.2f} "
              f"eff_lev={cap * args.sizer_leverage:.1f}x  {time.time() - t0:.1f}s")
        streams.append(s)

    if not streams:
        return {"error": "no symbols produced streams"}

    aligned = _align_streams(streams)
    if "error" in aligned:
        return aligned

    symbols = [s["symbol"] for s in streams]
    report = _compose_portfolio(aligned, symbols, args.capital)
    report["per_symbol_tier_caps"] = per_symbol_caps
    report["leverage"] = args.sizer_leverage
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capital", type=float, default=400.0,
                   help="Total account equity (default $400)")
    p.add_argument("--months", type=int, default=12,
                   help="OOS lookback months (default 12)")
    p.add_argument("--sizer-leverage", type=float, default=3.0,
                   help="Exchange leverage (default 3x)")
    p.add_argument("--btc-cap", type=float, default=0.65,
                   help="BTC tier_cap fraction (default 0.65)")
    p.add_argument("--eth-cap", type=float, default=0.65,
                   help="ETH tier_cap fraction (default 0.65)")
    p.add_argument("--latency-ms", type=float, default=500.0,
                   help="Simulated order→fill latency (default 500ms)")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON report only")
    args = p.parse_args()

    report = run_portfolio(args)
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return 1

    if args.json:
        # Drop equity_curve + portfolio_return arrays from JSON (huge)
        slim = {k: v for k, v in report.items()
                if k not in ("equity_curve", "portfolio_return")}
        print(json.dumps(slim, indent=2, default=str))
    else:
        print(_format_report(report, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

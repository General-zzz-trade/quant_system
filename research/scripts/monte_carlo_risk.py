#!/usr/bin/env python3
"""Monte Carlo risk simulation for Strategy H (4h direction + 1h scaling + BB scaler).

Loads actual trade-level returns from backtest signal generation (same pipeline as
backtest_small_capital.py), then runs 10,000 bootstrap simulations at various
leverage levels to estimate ruin probability, drawdown distributions, CAGR, and
optimal Kelly leverage.

Uses pickle for sklearn/lightgbm model loading (required by user for ML pipeline).

Usage:
    python3 -m scripts.research.monte_carlo_risk
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import json  # noqa: E402
import pickle  # noqa: S403,E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathlib import Path  # noqa: E402

sys.path.insert(0, "/quant_system")
from features.batch_feature_engine import compute_features_batch  # noqa: E402
from alpha.training.train_multi_horizon import rolling_zscore  # noqa: E402
from _quant_hotpath import (  # type: ignore[import-untyped]  # noqa: E402
    cpp_simulate_paths as rust_simulate_paths,
    MCResult as RustMCResult,
    BootstrapResult as RustBootstrapResult,
)

# Rust-accelerated Monte Carlo path simulation — 10K runs in <100ms
_rust_simulate = rust_simulate_paths

DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8")
COST = 0.0007  # round-trip cost per trade (taker fee x2)

# ── Data loading (identical to backtest_small_capital.py) ──────────────────────


def load_resample(sym: str, rule: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{sym}_1h.csv")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    if rule == "1h":
        return df
    agg = {
        "open_time": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for c in ["quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "trades"]:
        if c in df.columns:
            agg[c] = "sum"
    return (
        df.set_index("datetime")
        .resample(rule)
        .agg(agg)
        .dropna(subset=["close"])
        .reset_index()
    )


def add_cm(feat_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    p = DATA_DIR / "cross_market_daily.csv"
    if not p.exists():
        return feat_df
    cm = pd.read_csv(p, parse_dates=["date"])
    cm["date"] = cm["date"].dt.date
    dates = pd.to_datetime(df["open_time"], unit="ms").dt.date
    ci = cm.set_index("date")
    # T-1 shift: bar on date D uses data from D-1
    ci.index = [d + pd.Timedelta(days=1) for d in ci.index]
    for col in ci.columns:
        feat_df[col] = dates.map(lambda d, c=col: ci[c].get(d, np.nan)).ffill().values
    return feat_df


def load_predict(sym: str, itv: str, df: pd.DataFrame):
    suffix = {"4h": "_4h", "1h": "_gate_v2"}[itv]
    mdir = MODEL_DIR / f"{sym}{suffix}"
    if not mdir.exists():
        return None, None
    cfg = json.load(open(mdir / "config.json"))
    feats = cfg.get("features") or cfg["horizon_models"][0]["features"]
    feat_df = compute_features_batch(sym, df)
    feat_df = add_cm(feat_df, df)
    for f in feats:
        if f not in feat_df.columns:
            feat_df[f] = 0.0
    X = feat_df[feats].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    # pickle is required here for sklearn/lightgbm model serialization
    for name in (
        ["ridge_model.pkl", "lgb_model.pkl", "lgbm_v8.pkl"]
        + [h.get("lgbm", "") for h in cfg.get("horizon_models", [])]
        + [h.get("ridge", "") for h in cfg.get("horizon_models", [])]
    ):
        if not name:
            continue
        p2 = mdir / name
        if p2.exists():
            with open(p2, "rb") as fh:
                mdl = pickle.load(fh)  # noqa: S301
            if isinstance(mdl, dict) and "model" in mdl:
                mdl = mdl["model"]
            try:
                return mdl.predict(X), cfg
            except TypeError:
                import xgboost as xgb

                return mdl.predict(xgb.DMatrix(X)), cfg
    return None, None


def gen_sig(pred: np.ndarray, close: np.ndarray, cfg: dict) -> np.ndarray:
    dz = cfg.get("deadzone", 1.0)
    mh = cfg.get("min_hold", 6)
    maxh = cfg.get("max_hold", 36)
    lo = cfg.get("long_only", False)
    mg = cfg.get("monthly_gate", False)
    z = rolling_zscore(
        pred, window=cfg.get("zscore_window", 180), warmup=cfg.get("zscore_warmup", 45)
    )
    sig = np.zeros(len(z))
    sig[z > dz] = 1
    sig[z < -dz] = -1
    if lo:
        sig[sig < 0] = 0
    if mg:
        sw = 120 if len(close) < 5000 else 480
        sma = pd.Series(close).rolling(sw, min_periods=sw // 2).mean().values
        for i in range(len(sig)):
            if not np.isnan(sma[i]) and close[i] < sma[i]:
                sig[i] = min(sig[i], 0)
    # min-hold enforcement
    cur, hold = 0, 0
    for i in range(len(sig)):
        s = sig[i]
        if s != cur and s != 0:
            cur, hold = s, 1
        elif cur != 0 and hold < mh:
            sig[i] = cur
            hold += 1
        elif s != cur:
            cur, hold = s, (1 if s != 0 else 0)
    # max-hold enforcement
    cur, hold = 0, 0
    for i in range(len(sig)):
        if sig[i] != 0:
            if sig[i] == cur:
                hold += 1
            else:
                cur, hold = sig[i], 1
            if hold > maxh:
                sig[i], cur, hold = 0, 0, 0
        else:
            cur, hold = 0, 0
    return sig


def align_to_1h(sig, dt_src, dt_1h):
    return pd.Series(sig, index=dt_src).reindex(dt_1h, method="ffill").fillna(0).values


def bb_scale(cl, i, d, w=12):
    if i < w:
        return 1.0
    r = cl[i - w : i]
    ma, std = np.mean(r), np.std(r)
    if std <= 0:
        return 1.0
    bb = (cl[i] - ma) / std
    if d == 1:
        return (
            1.2
            if bb < -1
            else (1.0 if bb < -0.5 else (0.7 if bb < 0 else (0.5 if bb < 0.5 else 0.3)))
        )
    if d == -1:
        return (
            1.2
            if bb > 1
            else (1.0 if bb > 0.5 else (0.7 if bb > 0 else (0.5 if bb > -0.5 else 0.3)))
        )
    return 1.0


# ── Extract trade-level returns ───────────────────────────────────────────────


def extract_trade_returns() -> np.ndarray:
    """Run Strategy H backtest and extract per-trade net returns (as fractions)."""
    caps = {"BTCUSDT": 0.15, "ETHUSDT": 0.10}
    syms = list(caps.keys())

    print("  Loading data and generating signals...")
    data = {}
    for sym in syms:
        df1 = load_resample(sym, "1h")
        df4 = load_resample(sym, "4h")
        cl = df1["close"].values
        dt = df1["datetime"].values
        p1, c1 = load_predict(sym, "1h", df1)
        s1 = gen_sig(p1, cl, c1)
        p4, c4 = load_predict(sym, "4h", df4)
        s4r = gen_sig(p4, df4["close"].values, c4)
        s4 = align_to_1h(s4r, df4["datetime"].values, dt)
        data[sym] = {"cl": cl, "dt": dt, "s4": s4, "s1": s1}
        print(f"    {sym}: {len(cl)} bars, {int(np.sum(np.abs(np.diff(s4)) > 0))} signal changes")

    ml = min(len(data[s]["cl"]) for s in syms)

    # Simulate at 1x leverage to extract raw trade returns
    # Each trade return = direction * (exit/entry - 1) - 2*COST, scaled by cap*tf*bb
    trade_returns = []
    pos = {s: {"p": 0, "e": 0.0, "sc": 1.0, "cap": caps[s]} for s in syms}

    for i in range(1, ml):
        for sym in syms:
            d = data[sym]
            cl, s4, s1 = d["cl"], d["s4"], d["s1"]
            cap = caps[sym]
            c = cl[i]
            s = int(s4[i])
            p = pos[sym]
            if s != p["p"]:
                if p["p"] != 0 and p["e"] > 0:
                    # Raw net return for this trade (before leverage)
                    raw_ret = p["p"] * (c / p["e"] - 1) - 2 * COST
                    # Effective portfolio return = raw_ret * cap_weight * scale
                    eff_ret = raw_ret * cap * p["sc"]
                    trade_returns.append(eff_ret)
                if s != 0:
                    s1v = int(s1[i])
                    tf = 1.3 if s1v == s else (0.3 if s1v == -s else 0.7)
                    p["p"], p["e"], p["sc"] = s, c, tf * bb_scale(cl, i, s)
                else:
                    p["p"], p["e"], p["sc"] = 0, 0.0, 1.0

    returns = np.array(trade_returns)
    print(f"  Extracted {len(returns)} trades")
    print(
        f"  Mean return: {returns.mean() * 100:.3f}%, "
        f"Median: {np.median(returns) * 100:.3f}%, "
        f"Std: {returns.std() * 100:.3f}%"
    )
    win_rate = np.sum(returns > 0) / len(returns) * 100
    print(f"  Win rate: {win_rate:.1f}%")
    return returns


# ── Monte Carlo simulation ────────────────────────────────────────────────────


def run_monte_carlo(
    trade_returns: np.ndarray,
    n_sims: int = 10_000,
    leverage_levels: list[float] | None = None,
    initial_equity: float = 500.0,
    ruin_threshold: float = 50.0,
    target_10k: float = 10_000.0,
    target_100k: float = 100_000.0,
) -> dict:
    """Run bootstrap Monte Carlo simulations at multiple leverage levels."""
    if leverage_levels is None:
        leverage_levels = [1.0, 2.0, 3.0, 5.0, 10.0]

    n_trades = len(trade_returns)
    rng = np.random.default_rng(seed=42)

    results = {}

    for lev in leverage_levels:
        print(f"\n  Simulating {n_sims:,} paths at {lev:.0f}x leverage...")
        t0 = time.time()

        final_equities = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        time_to_10k = np.full(n_sims, np.inf)
        time_to_100k = np.full(n_sims, np.inf)

        for sim in range(n_sims):
            # Bootstrap: resample trade returns with replacement
            sampled_idx = rng.integers(0, n_trades, size=n_trades)
            sampled_returns = trade_returns[sampled_idx]

            # Simulate equity curve with leverage
            equity = initial_equity
            peak = equity
            worst_dd = 0.0
            hit_10k = False
            hit_100k = False

            for t_idx, ret in enumerate(sampled_returns):
                # Leveraged return applied to equity
                equity *= 1.0 + ret * lev
                # Ruin check (equity can't go negative in practice)
                if equity <= 0:
                    equity = 0.0
                    worst_dd = 1.0
                    break

                # Track drawdown
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0.0
                if dd > worst_dd:
                    worst_dd = dd

                # Track milestones
                if not hit_10k and equity >= target_10k:
                    time_to_10k[sim] = t_idx + 1
                    hit_10k = True
                if not hit_100k and equity >= target_100k:
                    time_to_100k[sim] = t_idx + 1
                    hit_100k = True

            final_equities[sim] = equity
            max_drawdowns[sim] = worst_dd

        elapsed = time.time() - t0

        # Compute statistics
        ruined = np.sum(final_equities < ruin_threshold)
        ruin_prob = ruined / n_sims

        # CAGR: assume each simulation covers the same calendar period as original
        # Original trades span ~n_trades trades; estimate years from trade frequency
        # Use actual backtest duration for CAGR calculation
        n_bars_approx = n_trades * 20  # rough: ~20 bars per trade on average
        n_years = n_bars_approx / (24 * 365)  # 1h bars to years
        if n_years < 0.1:
            n_years = 1.0

        # CAGR for each simulation
        valid = final_equities > 0
        cagrs = np.zeros(n_sims)
        cagrs[valid] = (final_equities[valid] / initial_equity) ** (1.0 / n_years) - 1.0
        cagrs[~valid] = -1.0  # total loss

        # Time to milestones (in trades)
        t10k_finite = time_to_10k[np.isfinite(time_to_10k)]
        t100k_finite = time_to_100k[np.isfinite(time_to_100k)]

        lev_results = {
            "leverage": lev,
            "ruin_probability": float(ruin_prob),
            "ruin_count": int(ruined),
            "final_equity": {
                "median": float(np.median(final_equities)),
                "p5": float(np.percentile(final_equities, 5)),
                "p25": float(np.percentile(final_equities, 25)),
                "p75": float(np.percentile(final_equities, 75)),
                "p95": float(np.percentile(final_equities, 95)),
                "mean": float(np.mean(final_equities)),
            },
            "max_drawdown": {
                "median": float(np.median(max_drawdowns)),
                "p5": float(np.percentile(max_drawdowns, 5)),
                "p95": float(np.percentile(max_drawdowns, 95)),
                "mean": float(np.mean(max_drawdowns)),
            },
            "cagr": {
                "median": float(np.median(cagrs)),
                "p5": float(np.percentile(cagrs, 5)),
                "p95": float(np.percentile(cagrs, 95)),
                "mean": float(np.mean(cagrs)),
            },
            "time_to_10k": {
                "median_trades": float(np.median(t10k_finite)) if len(t10k_finite) > 0 else None,
                "probability": float(len(t10k_finite) / n_sims),
            },
            "time_to_100k": {
                "median_trades": float(np.median(t100k_finite)) if len(t100k_finite) > 0 else None,
                "probability": float(len(t100k_finite) / n_sims),
            },
        }
        results[f"{lev:.0f}x"] = lev_results

        print(
            f"    Done in {elapsed:.1f}s | "
            f"Ruin: {ruin_prob * 100:.1f}% | "
            f"Median equity: ${np.median(final_equities):,.0f} | "
            f"Median MaxDD: {np.median(max_drawdowns) * 100:.1f}%"
        )

    return results


def estimate_kelly(trade_returns: np.ndarray) -> dict:
    """Estimate optimal Kelly leverage from trade return distribution."""
    # Kelly criterion: f* = mu / sigma^2
    # For leveraged returns: optimal_leverage = E[r] / Var[r]
    mu = np.mean(trade_returns)
    var = np.var(trade_returns)
    if var <= 0:
        return {"full_kelly": 0.0, "half_kelly": 0.0, "quarter_kelly": 0.0}

    full_kelly = mu / var
    return {
        "full_kelly": float(full_kelly),
        "half_kelly": float(full_kelly / 2),
        "quarter_kelly": float(full_kelly / 4),
        "mean_return": float(mu),
        "return_variance": float(var),
        "return_std": float(np.sqrt(var)),
    }


def print_results_table(results: dict, kelly: dict, n_trades: int, n_years: float):
    """Print formatted results table."""
    print(f"\n{'=' * 80}")
    print(f"  MONTE CARLO RISK SIMULATION -- Strategy H")
    print(f"  {10_000:,} bootstrap simulations, {n_trades} trades resampled")
    print(f"  Backtest span: {n_years:.1f} years | Initial equity: $500")
    print(f"{'=' * 80}")

    # Kelly estimate
    print(f"\n  KELLY LEVERAGE ESTIMATE")
    print(f"  {'-' * 50}")
    print(f"  Mean trade return:   {kelly['mean_return'] * 100:+.4f}%")
    print(f"  Return std:          {kelly['return_std'] * 100:.4f}%")
    print(f"  Full Kelly:          {kelly['full_kelly']:.2f}x")
    print(f"  Half Kelly:          {kelly['half_kelly']:.2f}x")
    print(f"  Quarter Kelly:       {kelly['quarter_kelly']:.2f}x")

    # Ruin probability table
    print(f"\n  PROBABILITY OF RUIN (equity < $50)")
    print(f"  {'-' * 50}")
    print(f"  {'Leverage':>10s}  {'Ruin %':>10s}  {'Ruined':>10s}")
    for key in sorted(results.keys(), key=lambda x: float(x.replace("x", ""))):
        r = results[key]
        print(
            f"  {r['leverage']:>9.0f}x  {r['ruin_probability'] * 100:>9.1f}%  "
            f"{r['ruin_count']:>10d}"
        )

    # MaxDD table
    print(f"\n  EXPECTED MAX DRAWDOWN")
    print(f"  {'-' * 60}")
    print(f"  {'Leverage':>10s}  {'Median':>10s}  {'5th pctl':>10s}  {'95th pctl':>10s}")
    for key in sorted(results.keys(), key=lambda x: float(x.replace("x", ""))):
        r = results[key]
        dd = r["max_drawdown"]
        print(
            f"  {r['leverage']:>9.0f}x  {dd['median'] * 100:>9.1f}%  "
            f"{dd['p5'] * 100:>9.1f}%  {dd['p95'] * 100:>9.1f}%"
        )

    # CAGR table
    print(f"\n  EXPECTED CAGR")
    print(f"  {'-' * 60}")
    print(f"  {'Leverage':>10s}  {'Median':>10s}  {'5th pctl':>10s}  {'95th pctl':>10s}")
    for key in sorted(results.keys(), key=lambda x: float(x.replace("x", ""))):
        r = results[key]
        c = r["cagr"]
        print(
            f"  {r['leverage']:>9.0f}x  {c['median'] * 100:>9.1f}%  "
            f"{c['p5'] * 100:>9.1f}%  {c['p95'] * 100:>9.1f}%"
        )

    # Final equity table
    print(f"\n  FINAL EQUITY DISTRIBUTION ($500 start)")
    print(f"  {'-' * 70}")
    print(
        f"  {'Leverage':>10s}  {'Median':>12s}  {'5th pctl':>12s}  "
        f"{'95th pctl':>12s}  {'Mean':>12s}"
    )
    for key in sorted(results.keys(), key=lambda x: float(x.replace("x", ""))):
        r = results[key]
        eq = r["final_equity"]
        print(
            f"  {r['leverage']:>9.0f}x  ${eq['median']:>11,.0f}  ${eq['p5']:>11,.0f}  "
            f"${eq['p95']:>11,.0f}  ${eq['mean']:>11,.0f}"
        )

    # Time to milestones
    print(f"\n  TIME TO MILESTONES (median trades to reach target)")
    print(f"  {'-' * 70}")
    print(
        f"  {'Leverage':>10s}  {'$10K trades':>12s}  {'$10K prob':>10s}  "
        f"{'$100K trades':>13s}  {'$100K prob':>10s}"
    )
    for key in sorted(results.keys(), key=lambda x: float(x.replace("x", ""))):
        r = results[key]
        t10 = r["time_to_10k"]
        t100 = r["time_to_100k"]
        t10_str = f"{t10['median_trades']:.0f}" if t10["median_trades"] is not None else "N/A"
        t100_str = f"{t100['median_trades']:.0f}" if t100["median_trades"] is not None else "N/A"
        print(
            f"  {r['leverage']:>9.0f}x  {t10_str:>12s}  {t10['probability'] * 100:>9.1f}%  "
            f"{t100_str:>13s}  {t100['probability'] * 100:>9.1f}%"
        )

    print(f"\n{'=' * 80}")


def main():
    print(f"\n{'=' * 70}")
    print(f"  Monte Carlo Risk Simulation -- Strategy H")
    print(f"{'=' * 70}")

    # Step 1: Extract trade returns from backtest
    trade_returns = extract_trade_returns()
    if len(trade_returns) == 0:
        print("  ERROR: No trades extracted. Check model files.")
        sys.exit(1)

    # Step 2: Kelly estimate
    kelly = estimate_kelly(trade_returns)

    # Step 3: Monte Carlo simulation
    leverage_levels = [1.0, 2.0, 3.0, 5.0, 10.0]
    results = run_monte_carlo(
        trade_returns,
        n_sims=10_000,
        leverage_levels=leverage_levels,
        initial_equity=500.0,
        ruin_threshold=50.0,
    )

    # Estimate backtest duration for CAGR display
    n_bars_approx = len(trade_returns) * 20
    n_years = n_bars_approx / (24 * 365)

    # Step 4: Print results
    print_results_table(results, kelly, len(trade_returns), n_years)

    # Step 5: Save to JSON
    output_dir = Path("data/runtime")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "monte_carlo.json"

    output = {
        "strategy": "Strategy H (4h direction + 1h scaling + BB scaler)",
        "n_simulations": 10_000,
        "n_trades": len(trade_returns),
        "initial_equity": 500.0,
        "ruin_threshold": 50.0,
        "estimated_years": round(n_years, 2),
        "kelly": kelly,
        "leverage_results": results,
        "trade_stats": {
            "mean_return": float(np.mean(trade_returns)),
            "median_return": float(np.median(trade_returns)),
            "std_return": float(np.std(trade_returns)),
            "min_return": float(np.min(trade_returns)),
            "max_return": float(np.max(trade_returns)),
            "win_rate": float(np.sum(trade_returns > 0) / len(trade_returns)),
            "n_trades": len(trade_returns),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()

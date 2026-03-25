#!/usr/bin/env python3
# ruff: noqa: E402,E501
"""A/B Test: ETH long_only=true vs long_only=false.

Compares ETHUSDT backtest performance with and without the short-selling
constraint using the last 3 months of data. Outputs Sharpe, trades, win_rate,
max_drawdown for both variants.

Decision rule:
  If false-Sharpe > true-Sharpe * 0.9  AND  false-MDD < true-MDD * 2.0
  => RECOMMEND switching to long_only=false
  Otherwise => KEEP long_only=true

Usage:
    python3 -m scripts.ab_test_eth_short
"""
from __future__ import annotations

import json
import pickle  # noqa: S403 — trusted local model files produced by our training pipeline
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "ETHUSDT"
DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8/ETHUSDT_gate_v2")
COST = 0.0007  # taker fee per side
LEVERAGE = 10.0
CAP = 0.10  # capital fraction per trade
INIT_EQUITY = 500.0
LAST_N_MONTHS = 3  # backtest on last 3 months of data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rolling_zscore(arr: np.ndarray, window: int = 720, warmup: int = 180) -> np.ndarray:
    """Compute rolling z-score (matching production signal pipeline)."""
    s = pd.Series(arr)
    mu = s.rolling(window, min_periods=warmup).mean()
    std = s.rolling(window, min_periods=warmup).std()
    z = ((s - mu) / std.replace(0, np.nan)).values
    return np.nan_to_num(z, nan=0.0)


def load_data() -> pd.DataFrame:
    """Load ETH 1h data."""
    path = DATA_DIR / f"{SYMBOL}_1h.csv"
    if not path.exists():
        print(f"ERROR: data file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def load_model_and_predict(df: pd.DataFrame) -> tuple[np.ndarray | None, dict | None]:
    """Load LGBM model, compute features, generate predictions.

    Uses pickle for loading LightGBM model files — these are trusted local
    artifacts produced by our own training pipeline.
    """
    config_path = MODEL_DIR / "config.json"
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return None, None

    with open(config_path) as f:
        cfg = json.load(f)

    # Use first horizon model's features
    hm = cfg["horizon_models"][0]
    features = hm["features"]

    feat_df = compute_features_batch(SYMBOL, df)

    # Add cross-market features if available
    cm_path = DATA_DIR / "cross_market_daily.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, parse_dates=["date"])
        cm["date"] = cm["date"].dt.date
        dates = pd.to_datetime(df["open_time"], unit="ms").dt.date
        ci = cm.set_index("date")
        # T-1 shift
        ci.index = [d + pd.Timedelta(days=1) for d in ci.index]
        for col in ci.columns:
            feat_df[col] = dates.map(lambda d, c=col: ci[c].get(d, np.nan)).ffill().values

    for f in features:
        if f not in feat_df.columns:
            feat_df[f] = 0.0

    X = feat_df[features].values.astype(np.float64)

    # Load LGBM model
    model_path = MODEL_DIR / hm["lgbm"]
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        return None, None

    with open(model_path, "rb") as fh:
        mdl = pickle.load(fh)  # noqa: S301 — trusted local artifact from our training pipeline
    if isinstance(mdl, dict) and "model" in mdl:
        mdl = mdl["model"]

    preds = mdl.predict(X)
    return preds, cfg


def generate_signal(
    preds: np.ndarray,
    close: np.ndarray,
    cfg: dict,
    long_only: bool,
) -> np.ndarray:
    """Generate trading signal from predictions.

    Replicates the production signal pipeline with configurable long_only.
    """
    dz = cfg.get("deadzone", 0.9)
    mh = cfg.get("min_hold", 18)
    maxh = cfg.get("max_hold", 60)
    zw = cfg.get("zscore_window", 720)
    zwarm = cfg.get("zscore_warmup", 180)

    z = _rolling_zscore(preds, window=zw, warmup=zwarm)
    sig = np.zeros(len(z))
    sig[z > dz] = 1
    sig[z < -dz] = -1

    if long_only:
        sig[sig < 0] = 0

    # Min-hold enforcement
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

    # Max-hold enforcement
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


def backtest(close: np.ndarray, sig: np.ndarray) -> dict:
    """Run simple backtest and return metrics."""
    eq = INIT_EQUITY
    pk = INIT_EQUITY
    mdd = 0.0
    pos = 0
    ep = 0.0
    trades = []
    dpnl_list = []
    dpnl = 0.0

    for i in range(1, len(close)):
        c = close[i]
        s = int(sig[i])
        if s != pos:
            if pos != 0 and ep > 0:
                net = pos * (c / ep - 1) - 2 * COST
                pnl = eq * CAP * LEVERAGE * net
                eq += pnl
                dpnl += pnl
                trades.append({"net": net, "dir": pos})
            if s != 0:
                pos, ep = s, c
            else:
                pos, ep = 0, 0.0
        pk = max(pk, eq)
        dd = (eq - pk) / pk * 100 if pk > 0 else 0
        mdd = min(mdd, dd)
        if i % 24 == 0:
            dpnl_list.append(dpnl / max(eq - dpnl, 1) if eq > 0 else 0)
            dpnl = 0.0

    dr = np.array(dpnl_list)
    sharpe = np.mean(dr) / np.std(dr) * np.sqrt(365) if len(dr) > 1 and np.std(dr) > 0 else 0.0
    rets = [t["net"] for t in trades]
    wr = np.mean(np.array(rets) > 0) * 100 if rets else 0.0
    n_long = sum(1 for t in trades if t["dir"] > 0)
    n_short = sum(1 for t in trades if t["dir"] < 0)
    long_wr = np.mean([t["net"] > 0 for t in trades if t["dir"] > 0]) * 100 if n_long else 0.0
    short_wr = np.mean([t["net"] > 0 for t in trades if t["dir"] < 0]) * 100 if n_short else 0.0
    total_ret = (eq / INIT_EQUITY - 1) * 100

    return {
        "sharpe": round(float(sharpe), 3),
        "trades": len(trades),
        "long_trades": n_long,
        "short_trades": n_short,
        "win_rate": round(float(wr), 2),
        "long_win_rate": round(float(long_wr), 2),
        "short_win_rate": round(float(short_wr), 2),
        "max_drawdown_pct": round(float(mdd), 2),
        "total_return_pct": round(float(total_ret), 2),
        "final_equity": round(float(eq), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("  A/B Test: ETH long_only=true vs long_only=false")
    print(f"  Last {LAST_N_MONTHS} months | Leverage {LEVERAGE}x | Cost {COST * 100:.2f}%/side")
    print("=" * 70)

    # Load data
    df = load_data()
    print(f"\n  Total data: {len(df)} bars ({len(df) / 24:.0f} days)")

    # Trim to last N months
    bars_needed = LAST_N_MONTHS * 30 * 24  # approximate
    # We need warmup bars before the test period for z-score convergence
    warmup_bars = 900  # > zscore_window(720) + zscore_warmup(180)
    total_needed = bars_needed + warmup_bars

    if len(df) < total_needed:
        print(f"  WARNING: only {len(df)} bars available, need ~{total_needed}")
        print("  Using all available data")
    else:
        df = df.iloc[-total_needed:].reset_index(drop=True)
        print(f"  Using last {len(df)} bars ({len(df) / 24:.0f} days incl. warmup)")

    # Load model and predict
    preds, cfg = load_model_and_predict(df)
    if preds is None or cfg is None:
        print("\n  ERROR: could not load model or generate predictions")
        return 1

    close = df["close"].values.astype(np.float64)

    # Trim to test period only (after warmup)
    test_start = warmup_bars if len(df) >= total_needed else 0
    close_test = close[test_start:]
    preds_test = preds[test_start:]  # noqa: F841

    print(f"  Test period: {len(close_test)} bars ({len(close_test) / 24:.0f} days)")
    print(f"  Config: deadzone={cfg.get('deadzone')}, min_hold={cfg.get('min_hold')}, "
          f"max_hold={cfg.get('max_hold')}")

    # --- Variant A: long_only=true (current production) ---
    sig_a = generate_signal(preds, close, cfg, long_only=True)
    sig_a_test = sig_a[test_start:]
    result_a = backtest(close_test, sig_a_test)

    # --- Variant B: long_only=false (allow shorts) ---
    sig_b = generate_signal(preds, close, cfg, long_only=False)
    sig_b_test = sig_b[test_start:]
    result_b = backtest(close_test, sig_b_test)

    # --- Output ---
    print(f"\n{'Metric':<22} {'long_only=true':>16} {'long_only=false':>16} {'Delta':>10}")
    print("-" * 70)

    metrics = [
        ("Sharpe", "sharpe"),
        ("Trades", "trades"),
        ("  Long trades", "long_trades"),
        ("  Short trades", "short_trades"),
        ("Win rate %", "win_rate"),
        ("  Long WR %", "long_win_rate"),
        ("  Short WR %", "short_win_rate"),
        ("Max drawdown %", "max_drawdown_pct"),
        ("Total return %", "total_return_pct"),
        ("Final equity $", "final_equity"),
    ]

    for label, key in metrics:
        va = result_a[key]
        vb = result_b[key]
        delta = vb - va
        sign = "+" if delta > 0 else ""
        print(f"  {label:<20} {va:>16} {vb:>16} {sign}{delta:>9.2f}")

    # --- Decision logic ---
    sharpe_a = result_a["sharpe"]
    sharpe_b = result_b["sharpe"]
    mdd_a = abs(result_a["max_drawdown_pct"])
    mdd_b = abs(result_b["max_drawdown_pct"])

    print(f"\n{'=' * 70}")
    print("  Decision Criteria:")
    sharpe_threshold = sharpe_a * 0.9
    mdd_threshold = mdd_a * 2.0 if mdd_a > 0 else 999.0

    crit_sharpe = sharpe_b > sharpe_threshold
    crit_mdd = mdd_b < mdd_threshold

    print(f"    Sharpe(false) > Sharpe(true) * 0.9:  "
          f"{sharpe_b:.3f} > {sharpe_threshold:.3f}  -> {'PASS' if crit_sharpe else 'FAIL'}")
    print(f"    MDD(false) < MDD(true) * 2.0:        "
          f"{mdd_b:.1f}% < {mdd_threshold:.1f}%  -> {'PASS' if crit_mdd else 'FAIL'}")

    if crit_sharpe and crit_mdd:
        print("\n  >>> RECOMMEND: Switch ETH to long_only=false")
        print(f"      Sharpe improves from {sharpe_a:.3f} to {sharpe_b:.3f}")
        print(f"      Short trades add {result_b['short_trades']} opportunities "
              f"with {result_b['short_win_rate']:.1f}% win rate")
    else:
        print("\n  >>> KEEP: ETH long_only=true (current production)")
        if not crit_sharpe:
            print("      Short trades degrade Sharpe below threshold")
        if not crit_mdd:
            print("      Short trades increase drawdown beyond tolerance")

    print()

    # Save results
    output = {
        "symbol": SYMBOL,
        "test_bars": len(close_test),
        "test_days": len(close_test) / 24,
        "variant_a_long_only_true": result_a,
        "variant_b_long_only_false": result_b,
        "decision": {
            "sharpe_criterion": crit_sharpe,
            "mdd_criterion": crit_mdd,
            "recommend_switch": bool(crit_sharpe and crit_mdd),
        },
    }

    out_path = Path("data/runtime/ab_test_eth_short.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

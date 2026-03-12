#!/usr/bin/env python3
"""Train 4-Hour BTCUSDT Production Model — V8-level rigor.

Key design:
  1. Resample 1m → 4h bars (~4800 bars over 800 days)
  2. Compute 105 features via Rust batch engine (no lag needed at 4h)
  3. LGBM + XGB ensemble with Optuna HPO
  4. Walk-forward validation (expanding window, 3-month folds)
  5. Bootstrap Sharpe validation + 4 production checks
  6. Export for Rust binary (LGBM JSON + pickle)

4h advantages over 1h:
  - IC sum 1.50 > 1h's 1.42 (from prior analysis)
  - Larger per-trade alpha (4h moves ~40-80bp vs 1h ~10-20bp)
  - Lower turnover → less cost drag
  - Basis/funding/FGI valid without lag

Usage:
    python3 -m scripts.train_4h_production --symbol BTCUSDT
    python3 -m scripts.train_4h_production --symbol BTCUSDT --no-hpo
"""
from __future__ import annotations
import sys, time, json, pickle, logging, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.dynamic_selector import greedy_ic_select, _rankdata, _spearman_ic
from scripts.train_v7_alpha import (
    V7_DEFAULT_PARAMS, V7_SEARCH_SPACE, INTERACTION_FEATURES, BLACKLIST,
)
from infra.model_signing import sign_file

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
BARS_PER_DAY = 6          # 4h bars per day
BARS_PER_MONTH = BARS_PER_DAY * 30  # 180 bars/month
OOS_MONTHS = 18
OOS_BARS = BARS_PER_MONTH * OOS_MONTHS  # 3240 bars = 18 months
WARMUP = 30               # 30 bars = 5 days warmup for feature stability
HORIZON = 12              # 12 bars = 48h lookahead (best from IC analysis)
TARGET_MODE = "clipped"
HPO_TRIALS = 10
N_FLEXIBLE = 4

COST_PER_TRADE = 6e-4     # 6 bps (4 fee + 2 slippage)

# Fixed features — strong at 4h from IC analysis
FIXED_FEATURES = [
    "vol_ma_ratio_5_20",    # IC=0.069 (top at 4h)
    "atr_norm_14",          # IC=0.060
    "basis",                # IC=0.058 (negative, contrarian)
    "parkinson_vol",        # Volatility measure
    "rsi_14",               # Mean reversion signal
    "ret_24",               # Momentum
    "fgi_normalized",       # Sentiment
    "fgi_extreme",          # Sentiment extremes
    "basis_zscore_24",      # Basis z-score
    "cvd_20",               # Cumulative volume delta
]

CANDIDATE_POOL = [
    "funding_zscore_24", "basis_momentum", "vol_5",
    "mean_reversion_20", "funding_sign_persist", "hour_sin",
    "bb_width_20", "body_ratio", "range_vs_rv",
    "vwap_dev_20", "ma_cross_10_30", "lower_shadow",
    "implied_vol_zscore_24", "iv_rv_spread",
    "exchange_supply_zscore_30", "mempool_size_zscore_24",
]


# ── Resampling ────────────────────────────────────────────────
def resample_1m_to_4h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to 4-hour bars."""
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)
    group_ms = 4 * 60 * 60_000  # 4 hours in milliseconds
    groups = ts // group_ms
    work = pd.DataFrame({
        "group": groups, "open_time": ts,
        "open": df_1m["open"].values.astype(np.float64),
        "high": df_1m["high"].values.astype(np.float64),
        "low": df_1m["low"].values.astype(np.float64),
        "close": df_1m["close"].values.astype(np.float64),
        "volume": df_1m["volume"].values.astype(np.float64),
    })
    for col in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"):
        work[col] = df_1m[col].values.astype(np.float64) if col in df_1m.columns else 0.0
    return work.groupby("group", sort=True).agg(
        open_time=("open_time", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
        taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
    ).reset_index(drop=True)


# ── Feature Computation ──────────────────────────────────────
def compute_4h_features(symbol: str, df_4h: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute 105 features on 4h bars. No lag needed — external data is hourly."""
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df_4h, include_v11=_has_v11)

    # Add interaction features
    for int_name, feat_a, feat_b in INTERACTION_FEATURES:
        if feat_a in feat_df.columns and feat_b in feat_df.columns:
            feat_df[int_name] = feat_df[feat_a].astype(float) * feat_df[feat_b].astype(float)

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    return feat_df, feature_names


# ── Target ────────────────────────────────────────────────────
def compute_target(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Forward return, clipped at 1st/99th percentile."""
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n-horizon] = closes[horizon:] / closes[:n-horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


# ── Bootstrap ─────────────────────────────────────────────────
def bootstrap_sharpe(pnl: np.ndarray, n_boot: int = 5000,
                     block_size: int = 6) -> Tuple[float, float, float]:
    """Block bootstrap P(Sharpe > 0) with 4h-appropriate block size (1 day)."""
    n = len(pnl)
    if n < block_size * 2:
        s = float(np.mean(pnl) / (np.std(pnl, ddof=1) + 1e-12)) * np.sqrt(365 * BARS_PER_DAY)
        return (1.0 if s > 0 else 0.0, s, s)

    rng = np.random.default_rng(42)
    n_blocks = (n + block_size - 1) // block_size
    sharpes = np.empty(n_boot)

    for b in range(n_boot):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot_pnl = np.concatenate([pnl[s:s + block_size] for s in block_starts])[:n]
        std = np.std(boot_pnl, ddof=1)
        if std > 1e-12:
            sharpes[b] = float(np.mean(boot_pnl)) / std * np.sqrt(365 * BARS_PER_DAY)
        else:
            sharpes[b] = 0.0

    return float(np.mean(sharpes > 0)), float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))


# ── Signal Generation ─────────────────────────────────────────
def pred_to_signal(pred: np.ndarray, deadzone: float, min_hold: int,
                   max_hold: int, long_only: bool = True) -> np.ndarray:
    """Z-score predictions → position signal with hold constraints."""
    std = np.nanstd(pred)
    if std < 1e-12:
        return np.zeros(len(pred))
    z = pred / std

    pos = np.zeros(len(pred))
    current = 0.0
    entry_bar = 0

    for i in range(len(pred)):
        if current != 0:
            held = i - entry_bar
            # Exit conditions
            exit_signal = (held >= max_hold or
                           (held >= min_hold and (current * z[i] < -0.3 or abs(z[i]) < 0.2)))
            if exit_signal:
                current = 0.0
        if current == 0:
            if z[i] > deadzone:
                current = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                current = -1.0
                entry_bar = i
        pos[i] = current
    return pos


# ── Backtest ──────────────────────────────────────────────────
def backtest_signal(signal: np.ndarray, closes: np.ndarray,
                    cost_per_trade: float = COST_PER_TRADE) -> Dict[str, Any]:
    """Backtest a position signal with realistic costs."""
    ret_1bar = np.diff(closes) / closes[:-1]
    sig = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(sig, prepend=0))
    gross_pnl = sig * ret_1bar
    cost = turnover * cost_per_trade
    net_pnl = gross_pnl - cost

    active = sig != 0
    n_active = int(active.sum())

    # Count trades (transitions from 0)
    entries = (sig != 0) & (np.concatenate([[0], sig[:-1]]) == 0)
    n_trades = int(entries.sum())

    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(365 * BARS_PER_DAY)

    total_return = float(np.sum(net_pnl))
    total_gross = float(np.sum(gross_pnl))
    total_cost = float(np.sum(cost))

    # Per-trade stats
    avg_gross_bps = total_gross / max(n_trades, 1) * 10000
    avg_net_bps = total_return / max(n_trades, 1) * 10000

    # Win rate
    trade_pnls = []
    in_trade = False
    trade_start = 0
    for i in range(len(sig)):
        if sig[i] != 0 and not in_trade:
            in_trade = True
            trade_start = i
        elif sig[i] == 0 and in_trade:
            trade_pnls.append(float(np.sum(net_pnl[trade_start:i])))
            in_trade = False
    if in_trade:
        trade_pnls.append(float(np.sum(net_pnl[trade_start:])))

    win_rate = float(np.mean([p > 0 for p in trade_pnls]) * 100) if trade_pnls else 0

    return {
        "trades": n_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "total_return": total_return,
        "total_gross": total_gross,
        "total_cost": total_cost,
        "avg_gross_bps": avg_gross_bps,
        "avg_net_bps": avg_net_bps,
        "n_active": n_active,
        "n_total": len(ret_1bar),
        "days": len(closes) / BARS_PER_DAY,
    }


# ── OOS Evaluation (V8 compatible) ───────────────────────────
def evaluate_oos(y_pred: np.ndarray, closes: np.ndarray, y_target: np.ndarray,
                 deadzone: float, min_hold: int, max_hold: int,
                 long_only: bool) -> Dict[str, Any]:
    """Full OOS evaluation: IC, Sharpe, bootstrap, monthly breakdown."""
    # IC
    valid = ~np.isnan(y_target)
    ic = 0.0
    if valid.sum() > 10:
        yp = y_pred[valid]
        yt = y_target[valid]
        if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
            ic = float(_spearman_ic(_rankdata(yp), _rankdata(yt)))

    # Signal & backtest
    signal = pred_to_signal(y_pred, deadzone, min_hold, max_hold, long_only)
    bt = backtest_signal(signal, closes)

    # Bootstrap
    ret_1bar = np.diff(closes) / closes[:-1]
    sig = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(sig, prepend=0))
    net_pnl = sig * ret_1bar - turnover * COST_PER_TRADE
    active = sig != 0
    n_active = int(active.sum())

    if n_active > 10:
        p_pos, ci_low, ci_high = bootstrap_sharpe(net_pnl[active])
    else:
        p_pos, ci_low, ci_high = 0.0, 0.0, 0.0

    # Monthly breakdown
    n_months = len(net_pnl) // BARS_PER_MONTH
    monthly_returns = []
    for m in range(n_months):
        s = m * BARS_PER_MONTH
        e = (m + 1) * BARS_PER_MONTH
        monthly_returns.append(float(np.sum(net_pnl[s:e])))
    pos_months = sum(1 for r in monthly_returns if r > 0)

    # H2 IC (second half of OOS)
    half = len(y_pred) // 2
    h2_ic = 0.0
    valid_h2 = ~np.isnan(y_target[half:])
    if valid_h2.sum() > 10:
        yp2 = y_pred[half:][valid_h2]
        yt2 = y_target[half:][valid_h2]
        if np.std(yp2) > 1e-12 and np.std(yt2) > 1e-12:
            h2_ic = float(_spearman_ic(_rankdata(yp2), _rankdata(yt2)))

    return {
        "sharpe": bt["sharpe"],
        "total_return": bt["total_return"],
        "ic": ic,
        "h2_ic": h2_ic,
        "bootstrap_p_positive": p_pos,
        "bootstrap_ci_95": [ci_low, ci_high],
        "n_months": n_months,
        "positive_months": pos_months,
        "monthly_returns": monthly_returns,
        "n_active_bars": n_active,
        "n_total_bars": len(net_pnl),
        "trades": bt["trades"],
        "win_rate": bt["win_rate"],
        "avg_gross_bps": bt["avg_gross_bps"],
        "avg_net_bps": bt["avg_net_bps"],
    }


# ── Walk-Forward Validation ──────────────────────────────────
def walk_forward(
    feat_df: pd.DataFrame,
    feature_names: List[str],
    closes: np.ndarray,
    horizon: int,
    params: Dict,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    long_only: bool,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """Expanding-window walk-forward with 3-month folds."""
    import lightgbm as lgb

    n = len(closes)
    fold_bars = BARS_PER_MONTH * 3  # 3-month test folds
    min_train = BARS_PER_MONTH * 6  # 6-month minimum training

    # Build folds
    folds = []
    start = min_train
    while start + fold_bars <= n:
        end = min(start + fold_bars, n)
        folds.append((0, start, start, end))
        start = end
    if not folds:
        return {"folds": 0, "avg_ic": 0, "avg_sharpe": 0}

    X = feat_df[feature_names].values.astype(np.float64)
    y_full = compute_target(closes, horizon)

    fold_results = []
    all_pred = np.full(n, np.nan)

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        # Train
        y_tr = y_full[tr_start:tr_end]
        valid_tr = ~np.isnan(y_tr) & (np.arange(len(y_tr)) >= WARMUP)
        X_tr = X[tr_start:tr_end][valid_tr]
        y_tr = y_tr[valid_tr]

        if len(y_tr) < 100:
            continue

        lgb_params = {**params, "n_estimators": n_estimators, "objective": "regression", "verbosity": -1}
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        bst = lgb.train(lgb_params, dtrain, num_boost_round=n_estimators,
                        callbacks=[lgb.log_evaluation(0)])

        # Predict on test fold
        X_te = X[te_start:te_end]
        pred = bst.predict(X_te)
        all_pred[te_start:te_end] = pred

        # Evaluate fold
        y_te = y_full[te_start:te_end]
        valid_te = ~np.isnan(y_te)
        ic = fast_ic(pred, y_te)

        c_te = closes[te_start:te_end]
        signal = pred_to_signal(pred, deadzone, min_hold, max_hold, long_only)
        bt = backtest_signal(signal, c_te)

        fold_results.append({
            "fold": fi + 1,
            "train_bars": int(valid_tr.sum()),
            "test_bars": te_end - te_start,
            "ic": ic,
            "sharpe": bt["sharpe"],
            "trades": bt["trades"],
            "avg_net_bps": bt["avg_net_bps"],
            "net_pnl_pct": bt["total_return"] * 100,
        })

    if not fold_results:
        return {"folds": 0, "avg_ic": 0, "avg_sharpe": 0}

    avg_ic = np.mean([f["ic"] for f in fold_results])
    avg_sharpe = np.mean([f["sharpe"] for f in fold_results])
    avg_net_bps = np.mean([f["avg_net_bps"] for f in fold_results])
    total_net_pct = sum(f["net_pnl_pct"] for f in fold_results)
    profitable_folds = sum(1 for f in fold_results if f["net_pnl_pct"] > 0)

    return {
        "folds": len(fold_results),
        "fold_details": fold_results,
        "avg_ic": float(avg_ic),
        "avg_sharpe": float(avg_sharpe),
        "avg_net_bps": float(avg_net_bps),
        "total_net_pct": float(total_net_pct),
        "profitable_folds": profitable_folds,
    }


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train 4h production model (V8-level)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--both-sides", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    hpo_trials = args.hpo_trials
    use_hpo = not args.no_hpo
    long_only = not args.both_sides
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"models_v8/{symbol}_4h_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  4H PRODUCTION TRAINING: {symbol}")
    print(f"{'='*65}")
    print(f"  Pipeline: V8-level (ensemble LGBM+XGB, HPO, bootstrap)")
    print(f"  Horizon: {HORIZON} bars = {HORIZON * 4}h")
    print(f"  Long-only: {long_only}")
    print(f"  Features: {len(FIXED_FEATURES)} fixed + {N_FLEXIBLE} flexible")
    print(f"  HPO: {'ON' if use_hpo else 'OFF'}"
          f"{f' ({hpo_trials} trials)' if use_hpo else ''}")
    print(f"  OOS holdout: {OOS_BARS} bars ({OOS_MONTHS} months)")

    # ── Load & resample data ──────────────────────────────
    data_path = Path(f"data_files/{symbol}_1m.csv")
    if not data_path.exists():
        # Try 1h data
        data_path = Path(f"data_files/{symbol}_1h.csv")
        if not data_path.exists():
            print(f"  Data not found: data_files/{symbol}_1m.csv or _1h.csv")
            return
        print(f"  Loading 1h data from {data_path}...")
        df_base = pd.read_csv(data_path)
        # Resample 1h → 4h
        ts_col = "open_time" if "open_time" in df_base.columns else "timestamp"
        ts = df_base[ts_col].values.astype(np.int64)
        group_ms = 4 * 60 * 60_000
        groups = ts // group_ms
        work = pd.DataFrame({
            "group": groups, "open_time": ts,
            "open": df_base["open"].values.astype(np.float64),
            "high": df_base["high"].values.astype(np.float64),
            "low": df_base["low"].values.astype(np.float64),
            "close": df_base["close"].values.astype(np.float64),
            "volume": df_base["volume"].values.astype(np.float64),
        })
        for col in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"):
            work[col] = df_base[col].values.astype(np.float64) if col in df_base.columns else 0.0
        df_4h = work.groupby("group", sort=True).agg(
            open_time=("open_time", "first"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
            taker_buy_volume=("taker_buy_volume", "sum"),
            taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
        ).reset_index(drop=True)
    else:
        print(f"  Loading 1m data from {data_path}...")
        df_1m = pd.read_csv(data_path)
        print(f"  Loaded {len(df_1m):,} 1m bars")
        print(f"  Resampling to 4h...")
        df_4h = resample_1m_to_4h(df_1m)

    n_bars = len(df_4h)
    days = n_bars / BARS_PER_DAY
    print(f"  4h bars: {n_bars:,} ({days:.0f} days)")

    if n_bars <= OOS_BARS + WARMUP:
        print(f"  Not enough data: {n_bars} <= {OOS_BARS + WARMUP}")
        return

    # ── Compute features ──────────────────────────────────
    print(f"  Computing features...")
    t0 = time.time()
    feat_df, feature_names = compute_4h_features(symbol, df_4h)
    closes = df_4h["close"].values.astype(np.float64)
    feat_df["close"] = closes
    print(f"  Features computed in {time.time()-t0:.1f}s"
          f" ({len(feature_names)} features, {n_bars} bars)")

    # ── IC scan across horizons ───────────────────────────
    print(f"\n  IC scan (top features per horizon):")
    for h in [3, 6, 12, 24]:
        y_h = compute_target(closes, h)
        ics = []
        for fn in feature_names:
            if fn in feat_df.columns:
                ic = fast_ic(feat_df[fn].values, y_h)
                ics.append((fn, abs(ic), ic))
        ics.sort(key=lambda x: -x[1])
        ic_sum = sum(x[1] for x in ics[:10])
        top3 = ", ".join(f"{x[0]}={x[2]:+.3f}" for x in ics[:3])
        print(f"    h={h:2d} ({h*4:3d}h): IC_sum(top10)={ic_sum:.3f}  top3: {top3}")

    # ── Split train/OOS ───────────────────────────────────
    split = n_bars - OOS_BARS
    print(f"\n  Train: {split:,} bars ({split/BARS_PER_DAY:.0f}d), OOS: {OOS_BARS:,} bars ({OOS_MONTHS}m)")

    # ── Walk-forward validation ───────────────────────────
    print(f"\n  {'='*65}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  {'='*65}")

    # Test multiple deadzone/param combos
    best_wf = None
    best_score = -999
    best_dz = 0
    best_params = None

    DEADZONES = [0.5, 1.0, 2.0]
    PARAM_CONFIGS = [
        {"max_depth": 5, "num_leaves": 16, "learning_rate": 0.01, "min_child_samples": 80,
         "reg_alpha": 0.1, "reg_lambda": 2.0, "subsample": 0.7, "colsample_bytree": 0.7},  # V7 default
        {"max_depth": 4, "num_leaves": 12, "learning_rate": 0.01, "min_child_samples": 100,
         "reg_alpha": 0.3, "reg_lambda": 3.0, "subsample": 0.6, "colsample_bytree": 0.7},  # Conservative
        {"max_depth": 3, "num_leaves": 8, "learning_rate": 0.015, "min_child_samples": 120,
         "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.5, "colsample_bytree": 0.6},  # Very conservative
    ]

    min_hold = max(HORIZON // 4, 1)  # 3 bars = 12h
    max_hold = HORIZON * 3           # 36 bars = 144h = 6 days

    for pi, params in enumerate(PARAM_CONFIGS):
        for dz in DEADZONES:
            wf = walk_forward(feat_df, feature_names, closes, HORIZON,
                              params, dz, min_hold, max_hold, long_only)
            if wf["folds"] == 0:
                continue
            # Score = Sharpe-weighted IC (reward both predictability and profitability)
            score = wf["avg_sharpe"] * 0.5 + wf["avg_ic"] * 50
            marker = " ***" if score > best_score else ""
            print(f"    Config {pi+1}, dz={dz}: IC={wf['avg_ic']:.4f}, "
                  f"Sharpe={wf['avg_sharpe']:.2f}, net={wf['avg_net_bps']:.1f}bp, "
                  f"total={wf['total_net_pct']:.1f}%, "
                  f"folds={wf['profitable_folds']}/{wf['folds']}{marker}")

            if score > best_score:
                best_score = score
                best_wf = wf
                best_dz = dz
                best_params = params.copy()

    if best_wf is None:
        print("  No valid walk-forward results!")
        return

    DEADZONE = best_dz
    MIN_HOLD = min_hold
    MAX_HOLD = max_hold

    print(f"\n  Best config: dz={DEADZONE}, params={best_params}")
    print(f"  Walk-Forward: IC={best_wf['avg_ic']:.4f}, Sharpe={best_wf['avg_sharpe']:.2f}")
    if best_wf.get("fold_details"):
        for fd in best_wf["fold_details"]:
            marker = "+" if fd["net_pnl_pct"] > 0 else " "
            print(f"    Fold {fd['fold']}: IC={fd['ic']:.4f}, Sharpe={fd['sharpe']:.2f}, "
                  f"trades={fd['trades']}, net={marker}{fd['net_pnl_pct']:.1f}%")

    # ── Feature selection on training data ────────────────
    print(f"\n  Feature selection...")
    y_train_full = compute_target(closes[:split], HORIZON)
    X_train_full = feat_df[feature_names].values[:split].astype(np.float64)

    valid_tr = ~np.isnan(y_train_full) & (np.arange(split) >= WARMUP)
    X_train = X_train_full[valid_tr]
    y_train = y_train_full[valid_tr]
    print(f"  Train samples: {len(X_train):,}")

    # Select features: fixed + flexible from pool
    selected = []
    for f in FIXED_FEATURES:
        if f in feature_names:
            selected.append(f)
        else:
            print(f"    Warning: fixed feature '{f}' not found")

    pool_in_data = [f for f in CANDIDATE_POOL if f in feature_names]
    if pool_in_data and N_FLEXIBLE > 0:
        pool_idx = [feature_names.index(f) for f in pool_in_data]
        X_pool = X_train[:, pool_idx]
        flex = greedy_ic_select(X_pool, y_train, pool_in_data, top_k=N_FLEXIBLE)
        selected.extend(flex)
    print(f"  Selected {len(selected)} features: {selected}")

    sel_idx = [feature_names.index(n) for n in selected if n in feature_names]
    selected = [feature_names[i] for i in sel_idx]
    X_train_sel = X_train[:, sel_idx]

    X_oos = feat_df[feature_names].values[split:].astype(np.float64)
    X_oos_sel = X_oos[:, sel_idx]
    oos_closes = closes[split:]

    # ── HPO ───────────────────────────────────────────────
    params = dict(best_params)
    if use_hpo:
        print(f"\n  Running Optuna HPO ({hpo_trials} trials)...")
        try:
            from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
            import lightgbm as lgb

            n_tr = len(X_train_sel)
            val_size = min(n_tr // 4, BARS_PER_MONTH * 6)  # up to 6 months val
            X_hpo_train = X_train_sel[:-val_size]
            y_hpo_train = y_train[:-val_size]
            X_hpo_val = X_train_sel[-val_size:]
            y_hpo_val = y_train[-val_size:]

            def objective(trial_params):
                p = {**best_params, **trial_params, "objective": "regression", "verbosity": -1}
                dtrain = lgb.Dataset(X_hpo_train, label=y_hpo_train)
                dval = lgb.Dataset(X_hpo_val, label=y_hpo_val, reference=dtrain)
                bst = lgb.train(
                    p, dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(0)],
                )
                y_hat = bst.predict(X_hpo_val)
                vm = ~np.isnan(y_hpo_val)
                if vm.sum() < 10:
                    return 0.0
                return float(_spearman_ic(_rankdata(y_hat[vm]), _rankdata(y_hpo_val[vm])))

            opt = HyperOptimizer(
                search_space=V7_SEARCH_SPACE,
                objective_fn=objective,
                config=HyperOptConfig(n_trials=hpo_trials, direction="maximize"),
            )
            result = opt.optimize()
            params = {**best_params, **result.best_params}
            print(f"  HPO best IC: {result.best_value:.4f}")
            print(f"  HPO params: {result.best_params}")
        except Exception as e:
            print(f"  HPO failed: {e}, using walk-forward best params")

    # ── Train LGBM ────────────────────────────────────────
    print(f"\n  Training LGBM...")
    import lightgbm as lgb
    lgb_params = {**params, "objective": "regression", "verbosity": -1}
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    n_est = params.get("n_estimators", 500)
    lgbm_bst = lgb.train(lgb_params, dtrain, num_boost_round=n_est,
                         callbacks=[lgb.log_evaluation(0)])
    lgbm_pred = lgbm_bst.predict(X_oos_sel)
    print(f"  LGBM: {lgbm_bst.num_trees()} trees")

    # ── Train XGB ─────────────────────────────────────────
    print(f"  Training XGB...")
    import xgboost as xgb
    xgb_params = {
        "max_depth": params.get("max_depth", 5),
        "learning_rate": params.get("learning_rate", 0.01),
        "objective": "reg:squarederror",
        "verbosity": 0,
        "subsample": params.get("subsample", 0.7),
        "colsample_bytree": params.get("colsample_bytree", 0.7),
    }
    dtrain_xgb = xgb.DMatrix(X_train_sel, label=y_train)
    doos_xgb = xgb.DMatrix(X_oos_sel)
    xgb_bst = xgb.train(xgb_params, dtrain_xgb, num_boost_round=n_est)
    xgb_pred = xgb_bst.predict(doos_xgb)

    # ── Ensemble ──────────────────────────────────────────
    y_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred
    print(f"  Ensemble: 0.5 × LGBM + 0.5 × XGB")

    # ── OOS Evaluation ────────────────────────────────────
    y_oos = compute_target(oos_closes, HORIZON)
    metrics = evaluate_oos(y_pred, oos_closes, y_oos,
                           DEADZONE, MIN_HOLD, MAX_HOLD, long_only)

    print(f"\n  {'='*65}")
    print(f"  OOS RESULTS ({OOS_MONTHS} months)")
    print(f"  {'='*65}")
    print(f"  Sharpe:              {metrics['sharpe']:+.2f}")
    print(f"  Total return:        {metrics['total_return']*100:+.2f}%")
    print(f"  IC:                  {metrics['ic']:.4f}")
    print(f"  H2 IC:               {metrics['h2_ic']:.4f}")
    print(f"  Trades:              {metrics['trades']}")
    print(f"  Win rate:            {metrics['win_rate']:.1f}%")
    print(f"  Avg gross:           {metrics['avg_gross_bps']:+.1f} bp/trade")
    print(f"  Avg net:             {metrics['avg_net_bps']:+.1f} bp/trade")
    print(f"  Bootstrap P(S>0):    {metrics['bootstrap_p_positive']:.1%}")
    print(f"  Bootstrap 95% CI:    [{metrics['bootstrap_ci_95'][0]:.2f}, "
          f"{metrics['bootstrap_ci_95'][1]:.2f}]")
    print(f"  Positive months:     {metrics['positive_months']}/{metrics['n_months']}")
    print(f"  Active bars:         {metrics['n_active_bars']}/{metrics['n_total_bars']}")

    if metrics["monthly_returns"]:
        print(f"\n  Monthly returns:")
        for i, r in enumerate(metrics["monthly_returns"]):
            marker = "+" if r > 0 else " "
            print(f"    Month {i+1:2d}: {marker}{r*100:+.2f}%")

    # ── Pass/Fail checks ─────────────────────────────────
    min_pos_months = max(1, metrics["n_months"] * 10 // 18)
    checks = {
        "OOS Sharpe > 0.5": metrics["sharpe"] > 0.5,
        "H2 IC > 0": metrics["h2_ic"] > 0,
        "Bootstrap P(S>0) > 80%": metrics["bootstrap_p_positive"] > 0.80,
        f"Positive months >= {min_pos_months}/{metrics['n_months']}":
            metrics["positive_months"] >= min_pos_months,
    }
    all_pass = all(checks.values())

    print(f"\n  {'='*65}")
    print(f"  PRODUCTION READINESS")
    print(f"  {'='*65}")
    for desc, passed in checks.items():
        print(f"    {'PASS' if passed else 'FAIL'}: {desc}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ── Save models ───────────────────────────────────────
    print(f"\n  Saving to {out_dir}/...")

    # LGBM pickle
    lgbm_path = out_dir / "lgbm_v8.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump({"model": lgbm_bst, "features": selected}, f)
    sign_file(lgbm_path)

    # LGBM JSON for Rust binary
    lgbm_json_path = out_dir / "lgbm_4h.json"
    lgbm_bst.save_model(str(lgbm_json_path))

    # LGBM text
    lgbm_txt_path = out_dir / "lgbm_4h.txt"
    lgbm_bst.save_model(str(lgbm_txt_path))

    # XGB pickle
    xgb_path = out_dir / "xgb_v8.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_bst, "features": selected}, f)
    sign_file(xgb_path)

    # Features list
    with open(out_dir / "features.json", "w") as f:
        json.dump(selected, f, indent=2)

    # Config
    config = {
        "version": "v8_4h",
        "symbol": symbol,
        "ensemble": True,
        "ensemble_weights": [0.5, 0.5],
        "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
        "features": selected,
        "fixed_features": list(FIXED_FEATURES),
        "candidate_pool": list(CANDIDATE_POOL),
        "n_flexible": N_FLEXIBLE,
        "long_only": long_only,
        "deadzone": DEADZONE,
        "min_hold": MIN_HOLD,
        "max_hold": MAX_HOLD,
        "horizon": HORIZON,
        "horizon_hours": HORIZON * 4,
        "timeframe": "4h",
        "target_mode": TARGET_MODE,
        "engine": "rust_feature_engine",
        "params": params,
        "xgb_params": xgb_params,
        "hpo_trials": hpo_trials,
        "walk_forward": {
            "avg_ic": best_wf["avg_ic"],
            "avg_sharpe": best_wf["avg_sharpe"],
            "avg_net_bps": best_wf["avg_net_bps"],
            "total_net_pct": best_wf["total_net_pct"],
            "profitable_folds": best_wf["profitable_folds"],
            "total_folds": best_wf["folds"],
        },
        "metrics": metrics,
        "checks": {k: str(v) for k, v in checks.items()},
        "passed": all_pass,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Saved: lgbm_v8.pkl, xgb_v8.pkl, lgbm_4h.json, config.json")

    # ── Compare with 1h V8 ────────────────────────────────
    v8_1h_path = Path(f"models_v8/{symbol}_gate_v2/config.json")
    if not v8_1h_path.exists():
        v8_1h_path = Path(f"models_v8/{symbol}/config.json")
    if v8_1h_path.exists():
        with open(v8_1h_path) as f:
            v8_1h = json.load(f)
        v8m = v8_1h.get("metrics", {})
        print(f"\n  {'='*65}")
        print(f"  COMPARISON: 4h vs 1h V8")
        print(f"  {'='*65}")
        print(f"  {'Metric':<25s} {'4h':>10s} {'1h':>10s}")
        print(f"  {'-'*45}")
        print(f"  {'Sharpe':<25s} {metrics['sharpe']:>+10.2f} {v8m.get('sharpe', 0):>+10.2f}")
        print(f"  {'IC':<25s} {metrics['ic']:>10.4f} {v8m.get('ic', 0):>10.4f}")
        print(f"  {'H2 IC':<25s} {metrics['h2_ic']:>10.4f} {v8m.get('h2_ic', 0):>10.4f}")
        print(f"  {'Total return':<25s} {metrics['total_return']*100:>+10.2f}% {v8m.get('total_return', 0)*100:>+10.2f}%")
        print(f"  {'Positive months':<25s} {metrics['positive_months']:>5d}/{metrics['n_months']:<4d}"
              f" {v8m.get('positive_months', 0):>5d}/{v8m.get('n_months', 0):<4d}")

    # ── Registry ──────────────────────────────────────────
    try:
        from research.model_registry.registry import ModelRegistry
        registry = ModelRegistry()
        mv = registry.register(
            name=f"alpha_v8_4h_{symbol}",
            params=params,
            features=selected,
            metrics={
                "oos_sharpe": metrics["sharpe"],
                "oos_return": metrics["total_return"],
                "oos_ic": metrics["ic"],
                "h2_ic": metrics["h2_ic"],
                "bootstrap_p": metrics["bootstrap_p_positive"],
            },
            tags=["v8", "4h", "ensemble", "production" if all_pass else "candidate"],
        )
        if all_pass:
            registry.promote(mv.model_id)
            print(f"  Registered & promoted: {mv.name} v{mv.version}")
        else:
            print(f"  Registered (not promoted): {mv.name} v{mv.version}")
    except Exception as e:
        print(f"  Registry: {e}")

    if all_pass:
        print(f"\n  Production model ready at {out_dir}/")
        print(f"  Next: paper trading validation")
    else:
        print(f"\n  Model did not pass all checks. Review metrics.")


if __name__ == "__main__":
    main()

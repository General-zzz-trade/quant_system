#!/usr/bin/env python3
"""Bear Market BTC Alpha Research.

End-to-end pipeline: IC analysis → 7 strategy train/backtest → comparison report.

Strategies:
  A) Short Momentum — LGBM regression, long+short, no gate
  B) Funding Rate Carry — rule-based, 3 variants
  C) Bear Regime Detector — LGBM binary classifier, short-only in bear
  D) C+B2 Ensemble — C-priority merge (C when active, else B2)
  E) Vol Breakout Short — rule-based, bear + ATR > rolling p80 + RSI > 30
  F) Regime-Switch Portfolio — Bull=V8 gate_v2 long-only, Bear=C short
  G) Probability Sizing — continuous sizing from C's OOS probabilities

Usage:
    python3 -m scripts.bear_alpha_research --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from features.dynamic_selector import _rankdata, _spearman_ic
from alpha.signal_transform import pred_to_signal as _pred_to_signal, enforce_min_hold as _enforce_min_hold

logger = logging.getLogger(__name__)

# ── Constants (reuse from V7/V8) ─────────────────────────────

V7_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "num_leaves": 16,
    "min_child_samples": 80,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "regression",
    "verbosity": -1,
}

BLACKLIST = {
    "ret_1", "ret_3",
    "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
    "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    "avg_trade_size",
    "oi_chg_x_ret1",
}

INTERACTION_FEATURES = [
    ("rsi14_x_vol_regime", "rsi_14", "vol_regime"),
    ("funding_x_taker_imb", "funding_rate", "taker_imbalance"),
    ("btc_ret1_x_beta30", "btc_ret_1", "rolling_beta_30"),
    ("trade_int_x_body", "trade_intensity", "body_ratio"),
    ("cvd_x_oi_chg", "cvd_20", "oi_change_pct"),
    ("vol_of_vol_x_range", "vol_of_vol", "range_vs_rv"),
    ("basis_x_funding", "basis", "funding_rate"),
    ("basis_x_vol_regime", "basis", "vol_regime"),
    ("fgi_x_rsi14", "fgi_normalized", "rsi_14"),
]

COST_PER_TRADE = 6e-4  # 6 bps (4 fee + 2 slippage)
EARLY_STOPPING_ROUNDS = 50
MIN_TRAIN = 2000
MA_WINDOW = 480


# ── Regime helpers ────────────────────────────────────────────

def _compute_bear_bull_mask(closes: np.ndarray, ma_window: int = MA_WINDOW) -> np.ndarray:
    """Return bear mask: True where close < SMA(ma_window). Vectorized via cumsum."""
    n = len(closes)
    mask = np.zeros(n, dtype=bool)
    if n < ma_window:
        return mask
    cs = np.cumsum(closes)
    ma = np.empty(n)
    ma[:ma_window] = np.nan
    ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
    mask = (~np.isnan(ma)) & (closes < ma)
    return mask


def _compute_target_clipped(closes: np.ndarray, horizon: int = 24) -> np.ndarray:
    """24h clipped return target (same as V7)."""
    n = len(closes)
    raw = np.full(n, np.nan)
    raw[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1.0
    valid = raw[~np.isnan(raw)]
    if len(valid) < 10:
        return raw
    p1, p99 = np.percentile(valid, [1, 99])
    return np.where(np.isnan(raw), np.nan, np.clip(raw, p1, p99))


def _compute_bear_target(closes: np.ndarray, horizon: int = 24, threshold: float = -0.02) -> np.ndarray:
    """Binary target: 1 if 24h return < threshold (drop > 2%), else 0."""
    n = len(closes)
    raw = np.full(n, np.nan)
    raw[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1.0
    return np.where(np.isnan(raw), np.nan, (raw < threshold).astype(np.float64))


# _pred_to_signal and _enforce_min_hold imported from alpha.signal_transform (canonical)


# ── Shared backtest ───────────────────────────────────────────

def _backtest_strategy(signal: np.ndarray, closes: np.ndarray,
                       label: str = "") -> Dict[str, Any]:
    """Compute PnL metrics from signal array and close prices."""
    ret_1bar = np.diff(closes) / closes[:-1]
    sig = signal[:len(ret_1bar)]

    turnover = np.abs(np.diff(sig, prepend=0))
    gross_pnl = sig * ret_1bar
    cost = turnover * COST_PER_TRADE
    net_pnl = gross_pnl - cost

    equity = np.ones(len(net_pnl) + 1)
    for i in range(len(net_pnl)):
        equity[i + 1] = equity[i] * (1 + net_pnl[i])

    active = sig != 0
    n_active = int(active.sum())

    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    total_return = equity[-1] / equity[0] - 1.0
    win_rate = float(np.mean(net_pnl[active] > 0)) if n_active > 0 else 0.0

    # Bear-only metrics
    bear_mask = _compute_bear_bull_mask(closes)
    sig.copy()
    bear_ret_mask = bear_mask[:len(sig)]
    bear_active = (sig != 0) & bear_ret_mask
    n_bear_active = int(bear_active.sum())
    bear_sharpe = 0.0
    bear_return = 0.0
    if n_bear_active > 1:
        bear_pnl = net_pnl[bear_active]
        std_b = float(np.std(bear_pnl, ddof=1))
        if std_b > 0:
            bear_sharpe = float(np.mean(bear_pnl)) / std_b * np.sqrt(8760)
        bear_return = float(np.sum(bear_pnl))

    n_trades = int(np.sum(np.diff(sig) != 0))

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "total_return": round(total_return * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": n_trades,
        "active_pct": round(np.mean(sig != 0) * 100, 1),
        "long_pct": round(np.mean(sig > 0) * 100, 1),
        "short_pct": round(np.mean(sig < 0) * 100, 1),
        "bear_sharpe": round(bear_sharpe, 2),
        "bear_return": round(bear_return * 100, 2),
        "n_bear_active": n_bear_active,
    }


# ── Feature loading (reuse from train_v7_alpha) ──────────────

def _load_features(symbol: str) -> pd.DataFrame:
    """Load CSV + compute 73+ features via train_v7_alpha pipeline."""
    from scripts.train_v7_alpha import _load_and_compute_features

    csv_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} bars from {csv_path}")

    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    df = df.rename(columns={ts_col: "timestamp"})

    feat_df = _load_and_compute_features(symbol, df)
    if feat_df is None:
        raise RuntimeError(f"Feature computation failed for {symbol}")
    print(f"  Computed {len(feat_df.columns)} columns, {len(feat_df)} rows")
    return feat_df


def _get_available_features(feat_df: pd.DataFrame) -> List[str]:
    """Get available feature names excluding close/target/blacklisted."""
    exclude = {"close", "timestamp", "open_time"} | BLACKLIST
    available = [c for c in feat_df.columns if c not in exclude and not c.startswith("ret_")]
    all_nan = set(feat_df.columns[feat_df.isna().all()])
    available = [c for c in available if c not in all_nan]
    return available


def expanding_window_folds(n: int, n_folds: int = 5, min_train: int = MIN_TRAIN):
    """Expanding window CV folds (same as V7)."""
    test_total = n - min_train
    if test_total <= 0 or n_folds <= 0:
        return []
    fold_size = test_total // n_folds
    if fold_size < 50:
        return []
    folds = []
    for i in range(n_folds):
        test_start = min_train + i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n
        folds.append((0, test_start, test_start, test_end))
    return folds


# ══════════════════════════════════════════════════════════════
# Phase 1: Bear IC Analysis
# ══════════════════════════════════════════════════════════════

def run_bear_ic_analysis(
    feat_df: pd.DataFrame,
    available: List[str],
    out_dir: Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """Compute bear-side IC for all features, rank by |bear_ic| * sqrt(bear_icir)."""
    closes = feat_df["close"].values.astype(np.float64)
    target = _compute_target_clipped(closes, horizon=24)
    bear_mask = _compute_bear_bull_mask(closes)
    bull_mask = (~bear_mask) & (~np.isnan(closes[:len(bear_mask)]))
    # Only bars where we have valid features (after warmup)
    bear_mask[:65] = False
    bull_mask[:65] = False

    neg_mask = (~np.isnan(target)) & (target < 0)
    neg_mask[:65] = False

    valid = ~np.isnan(target)
    valid[:65] = False

    n_bear = int(bear_mask.sum())
    n_bull = int(bull_mask.sum())
    n_total = int(valid.sum())
    print(f"\n  Regime stats: bear={n_bear} bars ({n_bear/max(n_total,1)*100:.1f}%), "
          f"bull={n_bull} ({n_bull/max(n_total,1)*100:.1f}%), total={n_total}")

    results = []
    for feat_name in available:
        if feat_name not in feat_df.columns:
            continue
        x = feat_df[feat_name].values.astype(np.float64)

        # Overall IC
        mask_all = valid & ~np.isnan(x)
        ic_all = _spearman_ic(x[mask_all], target[mask_all]) if mask_all.sum() > 50 else 0.0

        # Bear IC
        mask_bear = bear_mask & ~np.isnan(x) & ~np.isnan(target)
        ic_bear = _spearman_ic(x[mask_bear], target[mask_bear]) if mask_bear.sum() > 50 else 0.0

        # Bull IC
        mask_bull = bull_mask & ~np.isnan(x) & ~np.isnan(target)
        ic_bull = _spearman_ic(x[mask_bull], target[mask_bull]) if mask_bull.sum() > 50 else 0.0

        # Negative-return IC
        mask_neg = neg_mask & ~np.isnan(x)
        ic_neg = _spearman_ic(x[mask_neg], target[mask_neg]) if mask_neg.sum() > 50 else 0.0

        # Bear ICIR: split bear bars into 5 windows of 200 bars
        bear_indices = np.where(mask_bear)[0]
        window_size = 200
        n_windows = min(5, len(bear_indices) // window_size)
        window_ics = []
        if n_windows >= 2:
            stride = max(1, (len(bear_indices) - window_size) // (n_windows - 1))
            for w in range(n_windows):
                start = w * stride
                end = min(start + window_size, len(bear_indices))
                widx = bear_indices[start:end]
                if len(widx) > 30:
                    wic = _spearman_ic(x[widx], target[widx])
                    window_ics.append(wic)

        if len(window_ics) >= 2:
            bear_icir = abs(np.mean(window_ics)) / max(np.std(window_ics, ddof=1), 1e-12)
            # IC sign consistency: fraction of windows with same sign
            signs = np.sign(window_ics)
            dominant = max(np.sum(signs > 0), np.sum(signs < 0))
            ic_consistency = dominant / len(window_ics)
        else:
            bear_icir = 0.0
            ic_consistency = 0.0

        # Composite score: |bear_ic| * sqrt(bear_icir)
        score = abs(ic_bear) * np.sqrt(max(bear_icir, 0))

        results.append({
            "feature": feat_name,
            "ic_overall": round(ic_all, 4),
            "ic_bear": round(ic_bear, 4),
            "ic_bull": round(ic_bull, 4),
            "ic_neg_return": round(ic_neg, 4),
            "bear_icir": round(bear_icir, 2),
            "ic_consistency": round(ic_consistency, 2),
            "score": round(score, 4),
            "n_bear_bars": int(mask_bear.sum()),
        })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # Print top 25
    print(f"\n  {'─'*90}")
    print(f"  {'Feature':<30} {'Overall':>8} {'Bear IC':>8} {'Bull IC':>8} "
          f"{'ICIR':>6} {'Consist':>7} {'Score':>7}")
    print(f"  {'─'*90}")
    for r in results[:25]:
        print(f"  {r['feature']:<30} {r['ic_overall']:>8.4f} {r['ic_bear']:>8.4f} "
              f"{r['ic_bull']:>8.4f} {r['bear_icir']:>6.2f} {r['ic_consistency']:>7.2f} "
              f"{r['score']:>7.4f}")
    print(f"  {'─'*90}")

    # Select top features (score > 0 and top 20)
    top_features = [r["feature"] for r in results[:20] if r["score"] > 0]
    print(f"\n  Selected {len(top_features)} bear features: {top_features}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase1_ic_analysis.json", "w") as f:
        json.dump({
            "n_bear": n_bear, "n_bull": n_bull, "n_total": n_total,
            "features": results,
            "top_features": top_features,
        }, f, indent=2)

    ic_df = pd.DataFrame(results)
    ic_df.to_csv(out_dir / "phase1_ic_ranked.csv", index=False)
    print(f"  Saved → {out_dir}/phase1_ic_analysis.json, phase1_ic_ranked.csv")

    return top_features, {"n_bear": n_bear, "n_bull": n_bull, "results": results}


# ══════════════════════════════════════════════════════════════
# Phase 2A: Short Momentum Model
# ══════════════════════════════════════════════════════════════

def run_strategy_short_momentum(
    feat_df: pd.DataFrame,
    bear_features: List[str],
    out_dir: Path,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Strategy A: LGBM regression, long+short, using bear-selected features."""
    print(f"\n{'='*70}")
    print("  Strategy A: Short Momentum Model")
    print(f"  Features: {len(bear_features)}")
    print(f"{'='*70}")

    closes = feat_df["close"].values.astype(np.float64)
    target = _compute_target_clipped(closes, horizon=24)

    # Build feature matrix
    for f in bear_features:
        if f not in feat_df.columns:
            feat_df[f] = np.nan
    X_all = feat_df[bear_features].values.astype(np.float64)

    # Valid mask
    valid_mask = ~np.isnan(target)
    valid_mask[:65] = False
    for ci in range(X_all.shape[1]):
        valid_mask &= ~np.isnan(X_all[:, ci])

    valid_idx = np.where(valid_mask)[0]
    X_clean = X_all[valid_idx]
    y_clean = target[valid_idx]
    print(f"  Valid samples: {len(X_clean)} / {len(X_all)}")

    if len(X_clean) < MIN_TRAIN:
        print(f"  ERROR: Not enough samples ({len(X_clean)} < {MIN_TRAIN})")
        return {"label": "A: Short Momentum", "sharpe": 0, "error": "insufficient data"}

    folds = expanding_window_folds(len(X_clean), n_folds=n_folds)
    if not folds:
        print("  ERROR: Could not create folds")
        return {"label": "A: Short Momentum", "sharpe": 0, "error": "no folds"}

    # Collect OOS predictions across all folds
    oos_pred = np.full(len(X_clean), np.nan)
    fold_metrics = []

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_clean[tr_start:tr_end]
        y_train = y_clean[tr_start:tr_end]
        X_test = X_clean[te_start:te_end]
        y_test = y_clean[te_start:te_end]

        weights = np.linspace(0.5, 1.0, len(X_train))

        model = LGBMAlphaModel(name=f"bear_a_fold{fi}", feature_names=tuple(bear_features))
        model.fit(
            X_train, y_train,
            params=V7_DEFAULT_PARAMS.copy(),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=26,
            sample_weight=weights,
        )

        y_pred = model._model.predict(X_test)
        oos_pred[te_start:te_end] = y_pred

        # Fold IC
        rx = _rankdata(y_pred)
        ry = _rankdata(y_test)
        ic = float(np.corrcoef(rx, ry)[0, 1]) if len(y_pred) > 1 else 0.0
        fold_metrics.append({"fold": fi, "ic": round(ic, 4), "n_test": len(y_test)})
        print(f"  Fold {fi}: IC={ic:.4f}, train={len(X_train)}, test={len(X_test)}")

    # Generate signal from OOS predictions (no gate, no long-only)
    valid_oos = ~np.isnan(oos_pred)
    signal_full = np.zeros(len(closes))
    oos_indices = valid_idx[valid_oos]
    pred_valid = oos_pred[valid_oos]
    sig = _pred_to_signal(pred_valid, deadzone=0.5, min_hold=24)
    for i, idx in enumerate(oos_indices):
        if idx < len(signal_full):
            signal_full[idx] = sig[i]

    result = _backtest_strategy(signal_full, closes, label="A: Short Momentum")
    result["fold_metrics"] = fold_metrics
    result["features"] = bear_features

    print(f"\n  Result: Sharpe={result['sharpe']}, Return={result['total_return']}%, "
          f"MaxDD={result['max_dd']}%")
    print(f"  Bear: Sharpe={result['bear_sharpe']}, Return={result['bear_return']}%")
    print(f"  Active: {result['active_pct']}% (long={result['long_pct']}%, short={result['short_pct']}%)")

    with open(out_dir / "strategy_a_short_momentum.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════════
# Phase 2B: Funding Rate Carry
# ══════════════════════════════════════════════════════════════

def run_strategy_funding_carry(
    feat_df: pd.DataFrame,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Strategy B: Rule-based funding carry, 3 variants."""
    print(f"\n{'='*70}")
    print("  Strategy B: Funding Rate Carry (3 variants)")
    print(f"{'='*70}")

    closes = feat_df["close"].values.astype(np.float64)
    n = len(closes)

    # Extract features
    fz24 = feat_df["funding_zscore_24"].values.astype(np.float64) if "funding_zscore_24" in feat_df.columns else np.zeros(n)
    basis = feat_df["basis"].values.astype(np.float64) if "basis" in feat_df.columns else np.zeros(n)
    oi_acc = feat_df["oi_acceleration"].values.astype(np.float64) if "oi_acceleration" in feat_df.columns else np.zeros(n)

    # Replace NaN with 0 for rule-based
    fz24 = np.nan_to_num(fz24, 0.0)
    basis = np.nan_to_num(basis, 0.0)
    oi_acc = np.nan_to_num(oi_acc, 0.0)

    results = []

    # B1: Pure funding
    raw_b1 = np.where(fz24 > 1.5, -1.0, np.where(fz24 < -1.5, 1.0, 0.0))
    sig_b1 = _enforce_min_hold(raw_b1, min_hold=24)
    r1 = _backtest_strategy(sig_b1, closes, label="B1: Funding Pure")
    results.append(r1)
    print(f"  B1 (pure funding): Sharpe={r1['sharpe']}, Return={r1['total_return']}%, "
          f"Bear Sharpe={r1['bear_sharpe']}")

    # B2: Funding + OI filter (short only when OI accelerating = leverage building)
    raw_b2 = np.where((fz24 > 1.5) & (oi_acc > 0), -1.0,
                       np.where(fz24 < -1.5, 1.0, 0.0))
    sig_b2 = _enforce_min_hold(raw_b2, min_hold=24)
    r2 = _backtest_strategy(sig_b2, closes, label="B2: Funding+OI")
    results.append(r2)
    print(f"  B2 (funding+OI): Sharpe={r2['sharpe']}, Return={r2['total_return']}%, "
          f"Bear Sharpe={r2['bear_sharpe']}")

    # B3: Funding + basis filter (short when funding high AND basis positive = contango)
    raw_b3 = np.where((fz24 > 1.5) & (basis > 0), -1.0,
                       np.where((fz24 < -1.5) & (basis < 0), 1.0, 0.0))
    sig_b3 = _enforce_min_hold(raw_b3, min_hold=24)
    r3 = _backtest_strategy(sig_b3, closes, label="B3: Funding+Basis")
    results.append(r3)
    print(f"  B3 (funding+basis): Sharpe={r3['sharpe']}, Return={r3['total_return']}%, "
          f"Bear Sharpe={r3['bear_sharpe']}")

    with open(out_dir / "strategy_b_funding_carry.json", "w") as f:
        json.dump(results, f, indent=2)

    # Attach B2 signal for reuse by Strategy D
    r2["_signal"] = sig_b2
    return results


# ══════════════════════════════════════════════════════════════
# Phase 2C: Bear Regime Detector
# ══════════════════════════════════════════════════════════════

def run_strategy_bear_detector(
    feat_df: pd.DataFrame,
    bear_features: List[str],
    out_dir: Path,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Strategy C: Binary classifier — predict >2% drop, short-only in bear regime."""
    print(f"\n{'='*70}")
    print("  Strategy C: Bear Regime Detector (binary classifier)")
    print(f"{'='*70}")

    closes = feat_df["close"].values.astype(np.float64)
    bear_mask = _compute_bear_bull_mask(closes)
    target_bin = _compute_bear_target(closes, horizon=24, threshold=-0.02)

    # Use subset: funding/basis/OI/vol/sentiment ~15 features
    bear_detector_pool = [
        "funding_zscore_24", "funding_momentum", "funding_extreme",
        "funding_sign_persist", "funding_cumulative_8",
        "basis", "basis_zscore_24", "basis_momentum",
        "vol_20", "vol_regime", "parkinson_vol", "atr_norm_14",
        "fgi_normalized", "fgi_extreme",
        "rsi_14", "bb_pctb_20",
        "oi_acceleration", "leverage_proxy",
    ]
    features_c = [f for f in bear_detector_pool if f in feat_df.columns]
    print(f"  Detector features: {len(features_c)}: {features_c}")

    X_all = feat_df[features_c].values.astype(np.float64)
    # Fill NaN with 0 for sparse features (funding, OI, basis may have gaps)
    X_all = np.nan_to_num(X_all, 0.0)

    # Only train on bear bars
    valid_mask = bear_mask & ~np.isnan(target_bin)
    valid_mask[:65] = False

    valid_idx = np.where(valid_mask)[0]
    X_clean = X_all[valid_idx]
    y_clean = target_bin[valid_idx]
    print(f"  Bear training samples: {len(X_clean)} (pos={int(y_clean.sum())}, "
          f"neg={int(len(y_clean) - y_clean.sum())})")

    if len(X_clean) < 500:
        print(f"  ERROR: Not enough bear samples ({len(X_clean)})")
        return {"label": "C: Bear Detector", "sharpe": 0, "error": "insufficient bear data"}

    folds = expanding_window_folds(len(X_clean), n_folds=n_folds, min_train=max(500, MIN_TRAIN // 4))
    if not folds:
        print("  ERROR: Could not create folds")
        return {"label": "C: Bear Detector", "sharpe": 0, "error": "no folds"}

    # Collect OOS predictions
    oos_prob = np.full(len(X_clean), np.nan)
    fold_metrics = []

    cls_params = V7_DEFAULT_PARAMS.copy()
    cls_params["objective"] = "binary"
    cls_params["metric"] = "binary_logloss"
    cls_params["is_unbalance"] = True

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_clean[tr_start:tr_end]
        y_train = y_clean[tr_start:tr_end]
        X_test = X_clean[te_start:te_end]
        y_test = y_clean[te_start:te_end]

        model = LGBMAlphaModel(name=f"bear_c_fold{fi}", feature_names=tuple(features_c))
        model.fit_classifier(
            X_train, y_train,
            params=cls_params,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=26,
        )

        # predict_proba
        prob = model._model.predict_proba(X_test)[:, 1]
        oos_prob[te_start:te_end] = prob

        acc = float(np.mean((prob > 0.5).astype(int) == y_test.astype(int)))
        auc = _simple_auc(y_test, prob)
        fold_metrics.append({"fold": fi, "acc": round(acc, 4), "auc": round(auc, 4)})
        print(f"  Fold {fi}: acc={acc:.4f}, AUC={auc:.4f}, "
              f"train={len(X_train)} (pos={int(y_train.sum())}), test={len(X_test)}")

    # Generate signal: prob > threshold → short, only in bear regime
    signal_full = np.zeros(len(closes))
    valid_oos = ~np.isnan(oos_prob)
    oos_indices = valid_idx[valid_oos]
    prob_valid = oos_prob[valid_oos]

    # Use adaptive threshold: if mean prob is low, lower threshold
    prob_in_bear = prob_valid  # all these are already bear bars
    p50, p75 = np.percentile(prob_in_bear, [50, 75]) if len(prob_in_bear) > 0 else (0.5, 0.6)
    threshold = max(0.3, min(0.6, p75))
    print(f"  Prob stats: mean={np.mean(prob_in_bear):.3f}, p50={p50:.3f}, p75={p75:.3f}, threshold={threshold:.3f}")

    for i, idx in enumerate(oos_indices):
        if idx < len(signal_full) and bear_mask[idx]:
            signal_full[idx] = -1.0 if prob_valid[i] > threshold else 0.0

    signal_full = _enforce_min_hold(signal_full, min_hold=24)

    result = _backtest_strategy(signal_full, closes, label="C: Bear Detector")
    result["fold_metrics"] = fold_metrics
    result["features"] = features_c

    print(f"\n  Result: Sharpe={result['sharpe']}, Return={result['total_return']}%, "
          f"MaxDD={result['max_dd']}%")
    print(f"  Bear: Sharpe={result['bear_sharpe']}, Return={result['bear_return']}%")
    print(f"  Active: {result['active_pct']}% (short={result['short_pct']}%)")

    with open(out_dir / "strategy_c_bear_detector.json", "w") as f:
        json.dump(result, f, indent=2)

    # Attach internals for reuse by Strategies D, F, G
    result["_signal"] = signal_full
    result["_oos_probs"] = prob_valid
    result["_oos_indices"] = oos_indices
    result["_threshold"] = threshold
    result["_bear_mask"] = bear_mask
    return result


def _simple_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Simple AUC computation without sklearn."""
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_scores = y_score[pos]
    neg_scores = y_score[neg]
    # Mann-Whitney U
    concordant = 0
    for ps in pos_scores:
        concordant += int(np.sum(neg_scores < ps)) + 0.5 * int(np.sum(neg_scores == ps))
    return concordant / (n_pos * n_neg)


# ══════════════════════════════════════════════════════════════
# Phase 2D: C+B2 Ensemble
# ══════════════════════════════════════════════════════════════

def run_strategy_ensemble_cb2(
    signal_c: np.ndarray,
    signal_b2: np.ndarray,
    closes: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """Strategy D: C-priority ensemble. Use C signal when active, else fall back to B2."""
    print(f"\n{'='*70}")
    print("  Strategy D: C+B2 Ensemble")
    print(f"{'='*70}")

    n = min(len(signal_c), len(signal_b2), len(closes))
    signal = np.where(signal_c[:n] != 0, signal_c[:n], signal_b2[:n])
    signal = _enforce_min_hold(signal, min_hold=24)

    result = _backtest_strategy(signal, closes[:n], label="D: C+B2 Ensemble")

    print(f"\n  Result: Sharpe={result['sharpe']}, Return={result['total_return']}%, "
          f"MaxDD={result['max_dd']}%")
    print(f"  Bear: Sharpe={result['bear_sharpe']}, Return={result['bear_return']}%")
    print(f"  Active: {result['active_pct']}% (long={result['long_pct']}%, short={result['short_pct']}%)")

    with open(out_dir / "strategy_d_ensemble_cb2.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════════
# Phase 2E: Vol Breakout Short
# ══════════════════════════════════════════════════════════════

def run_strategy_vol_breakout(
    feat_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    """Strategy E: Rule-based vol breakout short. Bear + ATR > rolling p80 + RSI > 30."""
    print(f"\n{'='*70}")
    print("  Strategy E: Vol Breakout Short")
    print(f"{'='*70}")

    closes = feat_df["close"].values.astype(np.float64)
    bear_mask = _compute_bear_bull_mask(closes)

    atr = feat_df["atr_norm_14"].values.astype(np.float64) if "atr_norm_14" in feat_df.columns else np.zeros(len(closes))
    rsi = feat_df["rsi_14"].values.astype(np.float64) if "rsi_14" in feat_df.columns else np.full(len(closes), 50.0)
    atr = np.nan_to_num(atr, nan=0.0)
    rsi = np.nan_to_num(rsi, nan=50.0)

    # Rolling 80th percentile of atr_norm_14 over 20 bars
    atr_p80 = pd.Series(atr).rolling(20, min_periods=20).quantile(0.8).values

    # Short when: bear regime + atr > rolling p80 + rsi > 0.30 (not oversold; 0-1 scale)
    raw = np.where(
        bear_mask & (~np.isnan(atr_p80)) & (atr > atr_p80) & (rsi > 0.30),
        -1.0, 0.0,
    )
    signal = _enforce_min_hold(raw, min_hold=24)

    n_raw = int((raw != 0).sum())
    n_held = int((signal != 0).sum())
    print(f"  Raw signals: {n_raw}, after min_hold: {n_held}")

    result = _backtest_strategy(signal, closes, label="E: Vol Breakout")

    print(f"\n  Result: Sharpe={result['sharpe']}, Return={result['total_return']}%, "
          f"MaxDD={result['max_dd']}%")
    print(f"  Bear: Sharpe={result['bear_sharpe']}, Return={result['bear_return']}%")
    print(f"  Active: {result['active_pct']}% (short={result['short_pct']}%)")

    with open(out_dir / "strategy_e_vol_breakout.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════════
# Phase 2F: Regime-Switch Portfolio
# ══════════════════════════════════════════════════════════════

def run_strategy_regime_switch(
    feat_df: pd.DataFrame,
    symbol: str,
    signal_c: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """Strategy F: Bull=V8 gate_v2 long-only, Bear=Strategy C short."""
    print(f"\n{'='*70}")
    print("  Strategy F: Regime-Switch Portfolio")
    print(f"{'='*70}")

    closes = feat_df["close"].values.astype(np.float64)
    bear_mask = _compute_bear_bull_mask(closes)

    # Load V8 gate_v2 ensemble
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    if not model_dir.exists():
        print(f"  ERROR: Model dir not found: {model_dir}")
        return {"label": "F: Regime-Switch", "sharpe": 0, "error": f"missing {model_dir}"}

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    features_v8 = cfg["features"]
    ensemble_weights = cfg.get("ensemble_weights", [])

    from infra.model_signing import load_verified_pickle
    raw_models = []
    for fname in cfg.get("models", []):
        data = load_verified_pickle(model_dir / fname)
        raw_models.append(data["model"])

    if len(ensemble_weights) < len(raw_models):
        ensemble_weights = [1.0 / len(raw_models)] * len(raw_models)

    print(f"  V8 models: {len(raw_models)}, features: {features_v8}")

    # Build V8 feature matrix
    X_v8 = np.column_stack([
        np.nan_to_num(feat_df[f].values.astype(np.float64), nan=0.0)
        if f in feat_df.columns else np.zeros(len(feat_df))
        for f in features_v8
    ])

    # Ensemble prediction
    import xgboost as xgb

    y_pred = np.zeros(len(closes))
    w_sum = sum(ensemble_weights)
    for rm, w in zip(raw_models, ensemble_weights):
        if isinstance(rm, xgb.core.Booster):
            p = rm.predict(xgb.DMatrix(X_v8))
        else:
            p = rm.predict(X_v8)
        y_pred += p * w
    y_pred /= w_sum

    # V8 signal: long-only
    target_mode = cfg.get("target_mode", "clipped")
    sig_v8 = _pred_to_signal(y_pred, target_mode=target_mode, deadzone=0.5, min_hold=24)
    sig_v8 = np.clip(sig_v8, 0.0, 1.0)  # long-only

    # Combine: bear → C signal, bull → V8 long-only
    n = min(len(bear_mask), len(signal_c), len(sig_v8))
    signal = np.where(bear_mask[:n], signal_c[:n], sig_v8[:n])

    bull_active = int(((~bear_mask[:n]) & (sig_v8[:n] != 0)).sum())
    bear_active = int((bear_mask[:n] & (signal_c[:n] != 0)).sum())
    print(f"  Bull active (V8 long): {bull_active}, Bear active (C short): {bear_active}")

    result = _backtest_strategy(signal, closes[:n], label="F: Regime-Switch")

    print(f"\n  Result: Sharpe={result['sharpe']}, Return={result['total_return']}%, "
          f"MaxDD={result['max_dd']}%")
    print(f"  Bear: Sharpe={result['bear_sharpe']}, Return={result['bear_return']}%")
    print(f"  Active: {result['active_pct']}% (long={result['long_pct']}%, short={result['short_pct']}%)")

    with open(out_dir / "strategy_f_regime_switch.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════════
# Phase 2G: Probability Sizing
# ══════════════════════════════════════════════════════════════

def run_strategy_prob_sizing(
    closes: np.ndarray,
    oos_probs: np.ndarray,
    oos_indices: np.ndarray,
    bear_mask: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> Dict[str, Any]:
    """Strategy G: Continuous sizing from C's OOS probabilities.

    Position size = -(prob - threshold) / (1 - threshold), yielding [-1, 0].
    """
    print(f"\n{'='*70}")
    print("  Strategy G: Probability Sizing")
    print(f"{'='*70}")

    signal = np.zeros(len(closes))
    denom = max(1.0 - threshold, 1e-6)
    for i, idx in enumerate(oos_indices):
        if idx < len(signal) and bear_mask[idx]:
            prob = oos_probs[i]
            if prob > threshold:
                signal[idx] = -(prob - threshold) / denom

    signal = _enforce_min_hold(signal, min_hold=24)

    n_active = int((signal != 0).sum())
    avg_size = float(np.mean(np.abs(signal[signal != 0]))) if n_active > 0 else 0.0
    print(f"  Threshold: {threshold:.3f}, active bars: {n_active}, avg size: {avg_size:.3f}")

    result = _backtest_strategy(signal, closes, label="G: Prob Sizing")

    print(f"\n  Result: Sharpe={result['sharpe']}, Return={result['total_return']}%, "
          f"MaxDD={result['max_dd']}%")
    print(f"  Bear: Sharpe={result['bear_sharpe']}, Return={result['bear_return']}%")
    print(f"  Active: {result['active_pct']}% (short={result['short_pct']}%)")

    with open(out_dir / "strategy_g_prob_sizing.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════════
# Phase 4: Strategy F — 21-Fold Walk-Forward Validation
# ══════════════════════════════════════════════════════════════

# Walk-forward constants (same as walkforward_validate.py)
WF_MIN_TRAIN = 8760   # 12 months
WF_TEST = 2190         # 3 months
WF_STEP = 2190         # 3 months


def run_strategy_f_walkforward(
    feat_df: pd.DataFrame,
    symbol: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Phase 4: 21-fold expanding-window walk-forward for Strategy F (Regime-Switch).

    Bull = V8 gate_v2 long-only, Bear = C detector short.
    V8 predictions computed once globally; C retrained per fold on bear bars.
    """
    import xgboost as xgb

    print(f"\n{'='*70}")
    print("  Phase 4: Strategy F — 21-Fold Walk-Forward Validation")
    print(f"{'='*70}")

    closes = feat_df["close"].values.astype(np.float64)
    n_bars = len(closes)
    bear_mask = _compute_bear_bull_mask(closes)
    target_bin = _compute_bear_target(closes, horizon=24, threshold=-0.02)

    # ── V8 global signal (compute once) ──────────────────────
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    if not model_dir.exists():
        print(f"  ERROR: Model dir not found: {model_dir}")
        return {"error": f"missing {model_dir}"}

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    features_v8 = cfg["features"]
    ensemble_weights = cfg.get("ensemble_weights", [])
    from infra.model_signing import load_verified_pickle
    raw_models = []
    for fname in cfg.get("models", []):
        data = load_verified_pickle(model_dir / fname)
        raw_models.append(data["model"])

    if len(ensemble_weights) < len(raw_models):
        ensemble_weights = [1.0 / len(raw_models)] * len(raw_models)

    X_v8 = np.column_stack([
        np.nan_to_num(feat_df[f].values.astype(np.float64), nan=0.0)
        if f in feat_df.columns else np.zeros(n_bars)
        for f in features_v8
    ])

    y_pred_v8 = np.zeros(n_bars)
    w_sum = sum(ensemble_weights)
    for rm, w in zip(raw_models, ensemble_weights):
        if isinstance(rm, xgb.core.Booster):
            p = rm.predict(xgb.DMatrix(X_v8))
        else:
            p = rm.predict(X_v8)
        y_pred_v8 += p * w
    y_pred_v8 /= w_sum

    target_mode = cfg.get("target_mode", "clipped")
    sig_v8 = _pred_to_signal(y_pred_v8, target_mode=target_mode, deadzone=0.5, min_hold=24)
    sig_v8 = np.clip(sig_v8, 0.0, 1.0)  # long-only

    print(f"  V8: {len(raw_models)} models, {len(features_v8)} features, "
          f"long bars={int((sig_v8 > 0).sum())}")

    # ── C bear detector features ─────────────────────────────
    bear_detector_pool = [
        "funding_zscore_24", "funding_momentum", "funding_extreme",
        "funding_sign_persist", "funding_cumulative_8",
        "basis", "basis_zscore_24", "basis_momentum",
        "vol_20", "vol_regime", "parkinson_vol", "atr_norm_14",
        "fgi_normalized", "fgi_extreme",
        "rsi_14", "bb_pctb_20",
        "oi_acceleration", "leverage_proxy",
    ]
    features_c = [f for f in bear_detector_pool if f in feat_df.columns]
    X_c = np.nan_to_num(feat_df[features_c].values.astype(np.float64), 0.0)

    # Valid bear mask for C training (bear + valid target + warmup)
    valid_c = bear_mask.copy()
    valid_c[:65] = False
    valid_c &= ~np.isnan(target_bin)

    cls_params = V7_DEFAULT_PARAMS.copy()
    cls_params["objective"] = "binary"
    cls_params["metric"] = "binary_logloss"
    cls_params["is_unbalance"] = True

    print(f"  C: {len(features_c)} features, bear bars={int(valid_c.sum())}")

    # ── Generate folds ───────────────────────────────────────
    folds = []
    test_start = WF_MIN_TRAIN
    while test_start + WF_TEST <= n_bars:
        folds.append((len(folds), test_start, test_start + WF_TEST))
        test_start += WF_STEP

    print(f"  {n_bars} bars → {len(folds)} folds "
          f"(train≥{WF_MIN_TRAIN}, test={WF_TEST}, step={WF_STEP})\n")

    # ── Per-fold ─────────────────────────────────────────────
    fold_results = []

    for fi, te_start, te_end in folds:
        train_bear_idx = np.where(valid_c[:te_start])[0]

        if len(train_bear_idx) < 500:
            print(f"  Fold {fi:>2}: SKIP ({len(train_bear_idx)} bear bars < 500)")
            continue

        # Train C classifier on bear bars in [0, te_start)
        model = LGBMAlphaModel(name=f"wf_f_fold{fi}", feature_names=tuple(features_c))
        model.fit_classifier(
            X_c[train_bear_idx], target_bin[train_bear_idx],
            params=cls_params,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=26,
        )

        # Build combined signal for test window
        test_bear = bear_mask[te_start:te_end]
        test_signal = sig_v8[te_start:te_end].copy()
        test_signal[test_bear] = 0.0  # clear V8 in bear bars

        # Predict on bear bars in test window
        bear_rel = np.where(test_bear)[0]
        if len(bear_rel) > 0:
            bear_abs = bear_rel + te_start
            prob = model._model.predict_proba(X_c[bear_abs])[:, 1]
            p75 = float(np.percentile(prob, 75))
            threshold = max(0.3, min(0.6, p75))
            for i, rel in enumerate(bear_rel):
                test_signal[rel] = -1.0 if prob[i] > threshold else 0.0

        test_signal = _enforce_min_hold(test_signal, min_hold=24)

        # Backtest test window
        metrics = _backtest_strategy(test_signal, closes[te_start:te_end],
                                     label=f"Fold {fi}")

        fold_results.append({
            "fold": fi,
            "sharpe": metrics["sharpe"],
            "total_return": metrics["total_return"],
            "max_dd": metrics["max_dd"],
            "bear_sharpe": metrics["bear_sharpe"],
            "bear_return": metrics["bear_return"],
            "active_pct": metrics["active_pct"],
            "n_bear_train": len(train_bear_idx),
            "n_bear_test": len(bear_rel),
        })

        print(f"  Fold {fi:>2}: Sharpe={metrics['sharpe']:>6.2f}  "
              f"Return={metrics['total_return']:>7.1f}%  "
              f"MaxDD={metrics['max_dd']:>6.1f}%  "
              f"BearSh={metrics['bear_sharpe']:>6.2f}  "
              f"Active={metrics['active_pct']:>5.1f}%  "
              f"BearTr={len(train_bear_idx):>5}  BearTe={len(bear_rel):>4}")

    # ── Aggregate ────────────────────────────────────────────
    n_valid = len(fold_results)
    if n_valid == 0:
        print("\n  No valid folds!")
        return {"error": "no valid folds"}

    pos_sharpe = sum(1 for r in fold_results if r["sharpe"] > 0)
    pass_threshold = int(np.ceil(n_valid * 2 / 3))
    passed = pos_sharpe >= pass_threshold

    avg_sharpe = float(np.mean([r["sharpe"] for r in fold_results]))
    total_return = sum(r["total_return"] for r in fold_results)

    print(f"\n  {'─'*80}")
    print(f"  {'Fold':>4} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
          f"{'BearSh':>7} {'BearRet':>8} {'Active':>7}")
    print(f"  {'─'*80}")
    for r in fold_results:
        print(f"  {r['fold']:>4} {r['sharpe']:>7.2f} {r['total_return']:>7.1f}% "
              f"{r['max_dd']:>6.1f}% {r['bear_sharpe']:>7.2f} "
              f"{r['bear_return']:>7.1f}% {r['active_pct']:>6.1f}%")
    print(f"  {'─'*80}")

    verdict = "PASS" if passed else "FAIL"
    print(f"\n  VERDICT: {pos_sharpe}/{n_valid} positive Sharpe "
          f"(need >= {pass_threshold}) → {verdict}")
    print(f"  Avg Sharpe: {avg_sharpe:.2f}, Total Return: {total_return:.1f}%")

    result = {
        "verdict": verdict,
        "n_folds": n_valid,
        "positive_sharpe": pos_sharpe,
        "pass_threshold": pass_threshold,
        "passed": passed,
        "avg_sharpe": round(avg_sharpe, 2),
        "total_return": round(total_return, 1),
        "fold_results": fold_results,
    }

    with open(out_dir / "phase4_strategy_f_walkforward.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_dir}/phase4_strategy_f_walkforward.json")

    return result


# ══════════════════════════════════════════════════════════════
# Phase 3: Comparison Report
# ══════════════════════════════════════════════════════════════

def _judge(result: Dict[str, Any]) -> str:
    """PASS / PROMISING / FAIL."""
    bear_sharpe = result.get("bear_sharpe", 0)
    full_sharpe = result.get("sharpe", 0)
    max_dd = result.get("max_dd", -100)
    if bear_sharpe > 1.0 and full_sharpe > 0 and max_dd > -20:
        return "PASS"
    if bear_sharpe > 0.5 or full_sharpe > 0:
        return "PROMISING"
    return "FAIL"


def _print_comparison(all_results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Print formatted comparison table and save report."""
    print(f"\n{'='*90}")
    print("  BEAR ALPHA RESEARCH — COMPARISON REPORT")
    print(f"{'='*90}")

    header = (f"  {'Strategy':<25} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
              f"{'BearSh':>7} {'BearRet':>8} {'Active':>7} {'Verdict':>10}")
    print(header)
    print(f"  {'─'*85}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['label']:<25} {'ERROR':>7}  {r['error']}")
            continue
        verdict = _judge(r)
        print(f"  {r['label']:<25} {r['sharpe']:>7.2f} {r['total_return']:>7.1f}% "
              f"{r['max_dd']:>6.1f}% {r['bear_sharpe']:>7.2f} {r['bear_return']:>7.1f}% "
              f"{r['active_pct']:>6.1f}% {verdict:>10}")

    print(f"  {'─'*85}")

    # Recommendation
    valid_results = [r for r in all_results if "error" not in r]
    if not valid_results:
        print("\n  All strategies failed.")
        return
    best = max(valid_results, key=lambda r: r.get("bear_sharpe", -999))
    verdict = _judge(best)

    print(f"\n  Best bear strategy: {best['label']}")
    print(f"  Verdict: {verdict}")

    if verdict == "PASS":
        print("  → Upgrade to 21-fold walk-forward validation")
    elif verdict == "PROMISING":
        print("  → Add HPO and re-test")
    else:
        print("  → Bear market alpha not viable at this time")

    # Save full report (strip internal numpy arrays)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    report = {
        "strategies": clean,
        "best": best["label"],
        "verdict": verdict,
    }
    with open(out_dir / "phase3_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out_dir}/phase3_comparison.json")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Bear Market Alpha Research")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--phase", type=int, default=0, help="Run only phase N (1/2/3/4), 0=all")
    args = parser.parse_args()

    out_dir = Path("results/bear_alpha_research")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'#'*70}")
    print(f"  Bear Market Alpha Research — {args.symbol}")
    print(f"{'#'*70}")

    # Load features
    print("\n  Loading features...")
    feat_df = _load_features(args.symbol)
    available = _get_available_features(feat_df)
    print(f"  Available features: {len(available)}")

    # Phase 1
    if args.phase in (0, 1):
        print(f"\n{'#'*70}")
        print("  PHASE 1: Bear IC Analysis")
        print(f"{'#'*70}")
        bear_features, ic_data = run_bear_ic_analysis(feat_df, available, out_dir)
    else:
        # Load from saved
        with open(out_dir / "phase1_ic_analysis.json") as f:
            saved = json.load(f)
        bear_features = saved["top_features"]
        print(f"  Loaded {len(bear_features)} bear features from phase 1")

    # Phase 2
    all_results = []
    if args.phase in (0, 2):
        print(f"\n{'#'*70}")
        print("  PHASE 2: Strategy Comparison")
        print(f"{'#'*70}")

        # Strategy A
        result_a = run_strategy_short_momentum(feat_df, bear_features, out_dir, n_folds=args.n_folds)
        all_results.append(result_a)

        # Strategy B
        results_b = run_strategy_funding_carry(feat_df, out_dir)
        all_results.extend(results_b)

        # Strategy C
        result_c = run_strategy_bear_detector(feat_df, bear_features, out_dir, n_folds=args.n_folds)
        all_results.append(result_c)

        # Strategy D: C+B2 Ensemble
        closes = feat_df["close"].values.astype(np.float64)
        signal_c = result_c.get("_signal", np.zeros(len(closes)))
        signal_b2 = results_b[1].get("_signal", np.zeros(len(closes)))
        result_d = run_strategy_ensemble_cb2(signal_c, signal_b2, closes, out_dir)
        all_results.append(result_d)

        # Strategy E: Vol Breakout Short
        result_e = run_strategy_vol_breakout(feat_df, out_dir)
        all_results.append(result_e)

        # Strategy F: Regime-Switch Portfolio
        result_f = run_strategy_regime_switch(feat_df, args.symbol, signal_c, out_dir)
        all_results.append(result_f)

        # Strategy G: Probability Sizing
        oos_probs = result_c.get("_oos_probs", np.array([]))
        oos_indices = result_c.get("_oos_indices", np.array([]))
        bear_mask_c = result_c.get("_bear_mask", np.zeros(len(closes), dtype=bool))
        threshold_c = result_c.get("_threshold", 0.5)
        result_g = run_strategy_prob_sizing(closes, oos_probs, oos_indices, bear_mask_c, threshold_c, out_dir)
        all_results.append(result_g)
    else:
        # Load from saved
        for fname in ["strategy_a_short_momentum.json", "strategy_b_funding_carry.json",
                       "strategy_c_bear_detector.json"]:
            p = out_dir / fname
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)

    # Phase 3
    if args.phase in (0, 3):
        print(f"\n{'#'*70}")
        print("  PHASE 3: Comparison Report")
        print(f"{'#'*70}")
        _print_comparison(all_results, out_dir)

    # Phase 4: Strategy F Walk-Forward Validation
    if args.phase in (0, 4):
        print(f"\n{'#'*70}")
        print("  PHASE 4: Strategy F — 21-Fold Walk-Forward")
        print(f"{'#'*70}")
        run_strategy_f_walkforward(feat_df, args.symbol, out_dir)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()

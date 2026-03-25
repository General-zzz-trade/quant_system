#!/usr/bin/env python3
"""Bear-Market Short-Only Alpha Model — Walk-Forward Research.

Concept: when BTC/ETH close < SMA(480) (monthly gate triggers), instead of
going flat, activate a SEPARATE short-only model that profits from downtrends.

Key design:
  - Only active when close < SMA(480) — the bear regime
  - Signal in {-1, 0} — no longs allowed
  - Uses bear-favourable features (funding, OI unwind, overbought bounces, etc.)
  - Higher deadzone (1.5) — only short on strong signals
  - Shorter min_hold (12 bars) — bear moves are fast
  - NO monthly gate (this model IS the bear regime)

Walk-forward framework: expanding-window folds, per-fold feature selection,
LGBM train, OOS evaluation. Mirrors walkforward_validate.py.

Usage:
    python3 -m scripts.research.bear_model_research --symbol BTCUSDT
    python3 -m scripts.research.bear_model_research --symbol ETHUSDT --horizon 12
    python3 -m scripts.research.bear_model_research --symbol BTCUSDT --top-k 10
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from alpha.training.train_v7_alpha import (
    _compute_target,
    _load_and_compute_features,
    BLACKLIST,
)
from scripts.shared.signal_postprocess import (
    _compute_bear_mask,
    _enforce_min_hold,
    rolling_zscore,
)
from features.dynamic_selector import greedy_ic_select, _rankdata, _spearman_ic

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

MIN_TRAIN_BARS = 8760      # 12 months minimum training
TEST_BARS = 2190           # 3 months per test fold
STEP_BARS = 2190           # 3-month step (expanding window)
WARMUP = 65                # feature warmup bars
TOP_K = 7                  # fewer features — bear regime has less data
HORIZON = 24               # 24-bar (1-day) forward return
TARGET_MODE = "clipped"
DEADZONE = 1.5             # higher deadzone — only short strong signals
MIN_HOLD = 12              # shorter hold — bear moves are fast
ZSCORE_WINDOW = 720
ZSCORE_WARMUP = 168
MA_WINDOW = 480            # SMA window for bear regime detection

COST_PER_TRADE = 0.001     # 10 bps round-trip (taker + slippage)

# Bear-favourable features — these have signal value in downtrends
BEAR_FEATURE_POOL = [
    "funding_zscore_24",     # positive = crowded longs = good to short
    "oi_acceleration",       # negative = OI unwinding
    "rsi_14",                # overbought bounces = short entry points
    "vol_regime",            # expanding vol = continuation
    "basis_momentum",        # negative = contango deepening
    "cvd_20",                # negative = sell pressure
    "ma_cross_5_20",         # bearish cross
    # Additional useful bear features
    "funding_momentum",      # funding rate trend
    "funding_extreme",       # extreme funding = reversal
    "atr_norm_14",           # volatility level
    "bb_pctb_20",            # Bollinger %B — overbought/oversold
    "parkinson_vol",         # high-low volatility
    "leverage_proxy",        # leverage buildup
    "vol_of_vol",            # vol-of-vol spikes in bear
    "basis",                 # spot-futures basis
    "basis_zscore_24",       # basis z-score
]

# LGBM params tuned for bear regime (fewer samples, need regularisation)
BEAR_LGBM_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,          # shallower — less data in bear regime
    "learning_rate": 0.01,
    "num_leaves": 12,
    "min_child_samples": 60,
    "reg_alpha": 0.3,        # more regularisation
    "reg_lambda": 3.0,       # more regularisation
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "regression",
    "verbosity": -1,
}


# ── Fold generation ──────────────────────────────────────────

@dataclass
class Fold:
    idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class FoldResult:
    idx: int
    period: str
    ic: float
    sharpe: float
    total_return: float
    n_bear_bars_train: int
    n_bear_bars_test: int
    n_short_signals: int
    features: List[str]
    n_train: int
    n_test: int


def generate_wf_folds(
    n_bars: int,
    min_train_bars: int = MIN_TRAIN_BARS,
    test_bars: int = TEST_BARS,
    step_bars: int = STEP_BARS,
) -> List[Fold]:
    """Generate expanding-window walk-forward folds."""
    folds = []
    fold_idx = 0
    test_start = min_train_bars

    while test_start + test_bars <= n_bars:
        folds.append(Fold(
            idx=fold_idx,
            train_start=0,
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + test_bars,
        ))
        fold_idx += 1
        test_start += step_bars

    return folds


# ── Signal pipeline (short-only) ─────────────────────────────

def _pred_to_short_signal(
    y_pred: np.ndarray,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    zscore_window: int = ZSCORE_WINDOW,
    zscore_warmup: int = ZSCORE_WARMUP,
) -> np.ndarray:
    """Convert raw prediction to short-only signal {-1, 0}.

    Pipeline:
      1. Rolling z-score normalisation
      2. Discretize: z < -deadzone -> -1, else 0 (NO longs)
      3. Min-hold enforce
    """
    z = rolling_zscore(y_pred, window=zscore_window, warmup=zscore_warmup)

    # Short-only discretization: only allow -1 or 0
    signal = np.zeros(len(z))
    for i in range(len(z)):
        if z[i] < -deadzone:
            signal[i] = -1.0
        # z > +deadzone would be long — we skip it entirely

    # Min-hold
    signal = _enforce_min_hold(signal, min_hold)

    # Clip to short-only (safety — min_hold could theoretically propagate)
    np.clip(signal, -1.0, 0.0, out=signal)

    return signal


# ── Single fold execution ────────────────────────────────────

def run_bear_fold(
    fold: Fold,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    *,
    top_k: int = TOP_K,
    horizon: int = HORIZON,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    ma_window: int = MA_WINDOW,
) -> FoldResult:
    """Train and evaluate a single bear-model fold.

    Key differences from bull model:
      - Target = NEGATIVE forward return (we profit from drops)
      - Only train on bars where close < SMA(480) (bear regime)
      - Short-only signal pipeline
      - No monthly gate (this model IS the bear regime)
    """
    import lightgbm as lgb

    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]
    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]
    n_test = len(test_closes)

    # ── Bear regime masks ──
    bear_mask_train = _compute_bear_mask(train_closes, ma_window)
    bear_mask_test = _compute_bear_mask(test_closes, ma_window)

    n_bear_train = int(bear_mask_train.sum())
    n_bear_test = int(bear_mask_test.sum())

    # ── Target: negative forward return ──
    # We negate the target so that positive predictions = expecting drops
    # This way, the model learns to predict the magnitude of drops
    y_train_full = _compute_target(train_closes, horizon, TARGET_MODE)
    y_train_neg = -y_train_full  # negate: positive = expecting price drop

    # ── Build training data: only bear regime bars ──
    X_train_full = train_df[feature_names].values.astype(np.float64)

    # Valid training samples: in bear regime + valid target + past warmup
    bear_valid_train = bear_mask_train.copy()
    bear_valid_train[:WARMUP] = False
    bear_valid_train &= ~np.isnan(y_train_neg)

    bear_train_idx = np.where(bear_valid_train)[0]

    if len(bear_train_idx) < 500:
        logger.warning(
            "Fold %d: only %d bear bars in training (need 500+), skipping",
            fold.idx, len(bear_train_idx),
        )
        return FoldResult(
            idx=fold.idx, period="", ic=0.0, sharpe=0.0, total_return=0.0,
            n_bear_bars_train=n_bear_train, n_bear_bars_test=n_bear_test,
            n_short_signals=0, features=[], n_train=len(bear_train_idx), n_test=0,
        )

    X_train_bear = np.nan_to_num(X_train_full[bear_train_idx], 0.0)
    y_train_bear = y_train_neg[bear_train_idx]

    # ── Feature selection on bear training data ──
    selected = greedy_ic_select(X_train_bear, y_train_bear, feature_names, top_k=top_k)

    if not selected:
        logger.warning("Fold %d: no features selected", fold.idx)
        return FoldResult(
            idx=fold.idx, period="", ic=0.0, sharpe=0.0, total_return=0.0,
            n_bear_bars_train=n_bear_train, n_bear_bars_test=n_bear_test,
            n_short_signals=0, features=[], n_train=len(bear_train_idx), n_test=0,
        )

    sel_idx = [feature_names.index(f) for f in selected]
    X_train_sel = X_train_bear[:, sel_idx]

    # ── Train LGBM ──
    params = dict(BEAR_LGBM_PARAMS)
    dtrain = lgb.Dataset(X_train_sel, label=y_train_bear)
    bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 400),
        callbacks=[lgb.log_evaluation(0)],
    )

    # ── Predict on test ──
    X_test = test_df[feature_names].values.astype(np.float64)
    X_test_sel = np.nan_to_num(X_test[:, sel_idx], 0.0)
    y_pred = bst.predict(X_test_sel)

    # ── IC on bear test bars ──
    y_test_neg = -_compute_target(test_closes, horizon, TARGET_MODE)
    test_valid = bear_mask_test & ~np.isnan(y_test_neg)

    ic = 0.0
    if test_valid.sum() > 10:
        y_pred_v = y_pred[test_valid]
        y_test_v = y_test_neg[test_valid]
        if np.std(y_pred_v) > 1e-12 and np.std(y_test_v) > 1e-12:
            ic = float(_spearman_ic(_rankdata(y_pred_v), _rankdata(y_test_v)))

    # ── Convert to short-only signal ──
    signal = _pred_to_short_signal(
        y_pred, deadzone=deadzone, min_hold=min_hold,
    )

    # Zero out signal on non-bear bars (this model only trades in bear regime)
    signal[~bear_mask_test[:n_test]] = 0.0

    n_short_signals = int((signal == -1.0).sum())

    # ── PnL computation ──
    ret_1bar = np.diff(test_closes) / test_closes[:-1]
    signal_for_trade = signal[:len(ret_1bar)]

    gross_pnl = signal_for_trade * ret_1bar

    # Cost: turnover-based
    turnover = np.abs(np.diff(signal_for_trade, prepend=0))
    cost = turnover * COST_PER_TRADE

    # Funding: shorts receive positive funding (benefit)
    funding_cost = np.zeros(len(signal_for_trade))
    if "funding_rate" in test_df.columns:
        fr = test_df["funding_rate"].values[:len(signal_for_trade)].astype(np.float64)
        fr = np.nan_to_num(fr, 0.0)
        funding_cost = signal_for_trade * fr / 8.0  # position * funding

    net_pnl = gross_pnl - cost - funding_cost

    # Sharpe on active bars only
    active = signal_for_trade != 0
    n_active = int(active.sum())

    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net_pnl))

    return FoldResult(
        idx=fold.idx,
        period="",
        ic=ic,
        sharpe=sharpe,
        total_return=total_return,
        n_bear_bars_train=n_bear_train,
        n_bear_bars_test=n_bear_test,
        n_short_signals=n_short_signals,
        features=selected,
        n_train=len(bear_train_idx),
        n_test=n_test,
    )


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bear-Market Short-Only Alpha — Walk-Forward Research",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to test")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Forward return horizon (bars)")
    parser.add_argument("--deadzone", type=float, default=DEADZONE, help="Z-score deadzone for short entry")
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD, help="Minimum hold period (bars)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of features to select per fold")
    parser.add_argument("--ma-window", type=int, default=MA_WINDOW, help="SMA window for bear detection")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'=' * 72}")
    print("  Bear-Market Short-Only Alpha — Walk-Forward Research")
    print(f"  Symbol: {args.symbol}  |  Horizon: {args.horizon}  |  Deadzone: {args.deadzone}")
    print(f"  Min-hold: {args.min_hold}  |  Top-K features: {args.top_k}")
    print(f"  Bear detection: close < SMA({args.ma_window})")
    print(f"{'=' * 72}\n")

    t0 = time.time()

    # ── Load data ──
    data_path = Path(f"data_files/{args.symbol}_1h.csv")
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} bars ({len(df) / 8760:.1f} years)")

    # ── Compute features ──
    print("Computing features ...")
    feat_df = _load_and_compute_features(args.symbol, df)
    if feat_df is None or len(feat_df) == 0:
        print("ERROR: Feature computation failed")
        return

    closes = df["close"].values.astype(np.float64)

    # ── Filter to bear-relevant features ──
    # Start with the curated bear pool, then allow greedy selection from all
    available_bear = [f for f in BEAR_FEATURE_POOL if f in feat_df.columns]
    all_features = [
        f for f in feat_df.columns
        if f not in BLACKLIST
        and f != "close"
        and not f.startswith("ret_")
        and feat_df[f].notna().sum() > WARMUP
    ]

    # Use bear pool features + any other available features
    # (greedy IC selection will pick the best ones per fold)
    feature_names = list(dict.fromkeys(available_bear + all_features))

    # Ensure all are in feat_df
    feature_names = [f for f in feature_names if f in feat_df.columns]

    print(f"  Bear pool: {len(available_bear)} features available of {len(BEAR_FEATURE_POOL)} target")
    print(f"  Total candidate features: {len(feature_names)}")

    # ── Bear regime stats ──
    bear_mask = _compute_bear_mask(closes, args.ma_window)
    n_bear = int(bear_mask.sum())
    pct_bear = n_bear / len(closes) * 100 if len(closes) > 0 else 0
    print(f"\n  Bear regime bars: {n_bear} / {len(closes)} ({pct_bear:.1f}%)")

    if n_bear < 1000:
        print(f"\n  WARNING: Only {n_bear} bear bars. Results may be unreliable.")

    # ── Generate folds ──
    folds = generate_wf_folds(len(closes))
    print(f"  Walk-forward folds: {len(folds)}")

    if len(folds) == 0:
        print("ERROR: Not enough data for walk-forward folds")
        return

    # ── Run folds ──
    results: List[FoldResult] = []

    for fold in folds:
        logger.info("Running fold %d/%d ...", fold.idx + 1, len(folds))

        result = run_bear_fold(
            fold, feat_df, closes, feature_names,
            top_k=args.top_k,
            horizon=args.horizon,
            deadzone=args.deadzone,
            min_hold=args.min_hold,
            ma_window=args.ma_window,
        )

        # Add period label
        ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
        if ts_col in df.columns:
            ts_start = pd.Timestamp(df[ts_col].iloc[fold.test_start], unit="ms")
            ts_end = pd.Timestamp(df[ts_col].iloc[min(fold.test_end - 1, len(df) - 1)], unit="ms")
            result.period = f"{ts_start:%Y-%m} to {ts_end:%Y-%m}"
        else:
            result.period = f"bar {fold.test_start}-{fold.test_end}"

        results.append(result)

    elapsed = time.time() - t0

    # ── Results table ──
    print(f"\n{'=' * 100}")
    print(f"  RESULTS: Bear Short-Only Model — {args.symbol}")
    print(f"{'=' * 100}")
    print(
        f"{'Fold':>4s}  {'Period':<22s}  {'IC':>7s}  {'Sharpe':>8s}  "
        f"{'Return':>8s}  {'Bear(tr)':>8s}  {'Bear(te)':>8s}  "
        f"{'Shorts':>7s}  {'Train':>6s}  {'Test':>6s}"
    )
    print("-" * 100)

    ics = []
    sharpes = []
    returns = []
    n_positive = 0
    n_with_bear = 0

    for r in results:
        verdict = ""
        if r.n_bear_bars_test == 0:
            verdict = " [NO BEAR]"
        elif r.sharpe > 0:
            verdict = " PASS"
            n_positive += 1

        if r.n_bear_bars_test > 0:
            n_with_bear += 1

        print(
            f"{r.idx:4d}  {r.period:<22s}  {r.ic:7.3f}  {r.sharpe:8.2f}  "
            f"{r.total_return:8.4f}  {r.n_bear_bars_train:8d}  {r.n_bear_bars_test:8d}  "
            f"{r.n_short_signals:7d}  {r.n_train:6d}  {r.n_test:6d}{verdict}"
        )

        ics.append(r.ic)
        sharpes.append(r.sharpe)
        returns.append(r.total_return)

    print("-" * 100)

    # ── Summary stats ──
    avg_ic = np.mean(ics) if ics else 0.0
    avg_sharpe = np.mean(sharpes) if sharpes else 0.0
    total_ret = np.sum(returns)
    median_sharpe = np.median(sharpes) if sharpes else 0.0

    print(f"\n  Avg IC:        {avg_ic:.4f}")
    print(f"  Avg Sharpe:    {avg_sharpe:.2f}")
    print(f"  Median Sharpe: {median_sharpe:.2f}")
    print(f"  Total Return:  {total_ret:.4f} ({total_ret * 100:.2f}%)")
    print(f"  Positive folds: {n_positive}/{n_with_bear} (bear folds only)")

    # ── Feature frequency ──
    feat_counts: Dict[str, int] = {}
    for r in results:
        for f in r.features:
            feat_counts[f] = feat_counts.get(f, 0) + 1

    if feat_counts:
        print("\n  Top features (by selection frequency across folds):")
        sorted_feats = sorted(feat_counts.items(), key=lambda x: -x[1])
        for fname, count in sorted_feats[:15]:
            bar = "#" * count
            in_pool = "*" if fname in BEAR_FEATURE_POOL else " "
            print(f"    {in_pool} {fname:<30s}  {count:2d}/{len(results)}  {bar}")
        print("    (* = in curated bear pool)")

    # ── Verdict ──
    print(f"\n{'=' * 72}")
    if n_with_bear == 0:
        verdict = "INCONCLUSIVE — no bear regime bars in test folds"
    elif avg_sharpe >= 2.0 and n_positive >= n_with_bear * 0.6:
        verdict = f"PASS — Sharpe {avg_sharpe:.2f}, {n_positive}/{n_with_bear} positive bear folds"
    elif avg_sharpe >= 1.0 and n_positive >= n_with_bear * 0.5:
        verdict = f"WEAK PASS — Sharpe {avg_sharpe:.2f}, {n_positive}/{n_with_bear} positive bear folds"
    elif avg_sharpe > 0:
        verdict = f"MARGINAL — Sharpe {avg_sharpe:.2f}, needs more bear data"
    else:
        verdict = f"FAIL — Sharpe {avg_sharpe:.2f}, bear short model not profitable"

    print(f"  VERDICT: {verdict}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()

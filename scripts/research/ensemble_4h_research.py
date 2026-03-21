#!/usr/bin/env python3
"""4h Ensemble Research — train independent 4h model and blend with 1h.

Hypothesis: 4h model captures slower trend signals that 1h model misses.
Blending reduces trading frequency and improves signal stability.

Usage:
    python3 -m scripts.research.ensemble_4h_research --symbol BTCUSDT
    python3 -m scripts.research.ensemble_4h_research --symbol ETHUSDT --blend-sweep
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

log = logging.getLogger("4h_ensemble")

COST_PER_TRADE = 0.0006
MIN_TRAIN = 2190  # 3 months of 4h bars (= 12 months of 1h)
TEST_BARS_4H = 548  # 3 months of 4h bars


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h bars."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)

    ohlcv = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }).dropna(subset=["close"])

    if "open_time" in df.columns:
        ohlcv["open_time"] = ohlcv.index.astype(np.int64) // 10**6

    return ohlcv.reset_index(drop=True)


def compute_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic momentum/mean-reversion features for 4h bars."""
    c = df["close"].values.astype(float)
    n = len(c)

    feats = pd.DataFrame(index=range(n))
    feats["close"] = c

    # Returns
    for h in [1, 3, 6, 12]:
        feats[f"ret_{h}"] = pd.Series(c).pct_change(h).values

    # MA crossovers
    ma20 = pd.Series(c).rolling(20).mean().values
    ma50 = pd.Series(c).rolling(50).mean().values
    feats["close_vs_ma20"] = (c - ma20) / np.where(ma20 > 0, ma20, 1)
    feats["close_vs_ma50"] = (c - ma50) / np.where(ma50 > 0, ma50, 1)

    # RSI
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-8)
    feats["rsi_14"] = (100 - 100 / (1 + rs)).values

    # Volatility
    feats["vol_20"] = pd.Series(c).pct_change().rolling(20).std().values

    # Volume ratio
    v = df["volume"].values.astype(float)
    v_ma = pd.Series(v).rolling(20).mean().values
    feats["vol_ratio_20"] = v / np.where(v_ma > 0, v_ma, 1)

    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean()
    ema26 = pd.Series(c).ewm(span=26).mean()
    feats["macd_hist"] = ((ema12 - ema26) / c).values

    return feats


def run_4h_wf(symbol: str, blend_weight: float = 0.3) -> dict:
    """Run 4h walk-forward and blend with 1h prediction proxy."""
    # Load 1h data
    df_1h = pd.read_csv(f"data_files/{symbol}_1h.csv")

    # Resample to 4h
    df_4h = resample_to_4h(df_1h)
    log.info("4h bars: %d (from %d 1h bars)", len(df_4h), len(df_1h))

    # Compute features
    feat_4h = compute_simple_features(df_4h)
    closes_4h = df_4h["close"].values.astype(float)

    # Target: 6-bar forward return (6 × 4h = 24h)
    target = pd.Series(closes_4h).shift(-6) / pd.Series(closes_4h) - 1.0
    target = target.clip(-0.1, 0.1).values  # clip outliers

    # Feature columns
    feat_cols = [c for c in feat_4h.columns if c != "close"]
    X = feat_4h[feat_cols].values.astype(float)
    X = np.nan_to_num(X, 0.0)

    n = len(X)
    step = TEST_BARS_4H

    results = []
    fold_idx = 0

    for test_start in range(MIN_TRAIN, n - step, step):
        test_end = test_start + step
        train_X = X[:test_start]
        train_y = target[:test_start]
        test_X = X[test_start:test_end]
        test_closes = closes_4h[test_start:test_end]

        # Remove NaN targets
        valid = ~np.isnan(train_y)
        if valid.sum() < 200:
            continue

        try:
            import lightgbm as lgb
            dtrain = lgb.Dataset(train_X[valid], train_y[valid])
            params = {
                "objective": "regression", "metric": "mae",
                "num_leaves": 31, "learning_rate": 0.05,
                "feature_fraction": 0.8, "verbose": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=200)
            pred_4h = model.predict(test_X)
        except ImportError:
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
            m.fit(train_X[valid], train_y[valid])
            pred_4h = m.predict(test_X)

        # IC
        test_y = target[test_start:test_end]
        valid_test = ~np.isnan(test_y)
        if valid_test.sum() < 50:
            continue
        ic, _ = spearmanr(pred_4h[valid_test], test_y[valid_test])

        # Signal: simple z-score → discretize
        from scripts.shared.signal_postprocess import pred_to_signal
        signal = np.asarray(pred_to_signal(
            pred_4h, deadzone=0.8, min_hold=6, zscore_window=180, zscore_warmup=45,
        ), dtype=float)

        # Backtest
        ret_1bar = np.zeros(len(test_closes))
        ret_1bar[1:] = test_closes[1:] / test_closes[:-1] - 1.0
        turnover = np.zeros(len(signal))
        turnover[1:] = np.abs(np.diff(signal))
        pnl = signal * ret_1bar - turnover * COST_PER_TRADE

        active = signal != 0
        n_active = int(active.sum())
        if n_active < 10:
            continue

        mean_pnl = float(np.mean(pnl[active]))
        std_pnl = float(np.std(pnl[active]))
        sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(365 * 6)  # 4h bars annualized

        total_ret = float(np.prod(1 + pnl) - 1)

        results.append({
            "fold": fold_idx,
            "ic": round(ic, 4),
            "sharpe": round(sharpe, 2),
            "return": round(total_ret * 100, 1),
            "trades": int(turnover.sum()),
        })
        fold_idx += 1

    return {"folds": results, "n_folds": len(results)}


def main():
    parser = argparse.ArgumentParser(description="4h Ensemble Research")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--blend-sweep", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_4h_wf(args.symbol)
    folds = result["folds"]

    if not folds:
        print("No valid folds — insufficient data")
        return

    print()
    print("=" * 80)
    print(f"  4h Model Walk-Forward Results — {args.symbol}")
    print("=" * 80)
    print(f"  {'Fold':>4} {'IC':>8} {'Sharpe':>8} {'Return%':>9} {'Trades':>7}")
    for f in folds:
        print(f"  {f['fold']:>4} {f['ic']:>8.4f} {f['sharpe']:>8.2f} {f['return']:>8.1f}% {f['trades']:>7}")

    ics = [f["ic"] for f in folds]
    sharpes = [f["sharpe"] for f in folds]
    pass_n = sum(1 for s in sharpes if s > 0)
    print(f"\n  Mean IC: {np.mean(ics):.4f}  Mean Sharpe: {np.mean(sharpes):.2f}")
    print(f"  Pass: {pass_n}/{len(folds)} ({pass_n/len(folds)*100:.0f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()

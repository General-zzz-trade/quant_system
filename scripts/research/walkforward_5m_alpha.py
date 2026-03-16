#!/usr/bin/env python3
"""5m Alpha Walk-Forward Validation for ETHUSDT.

Tests Ridge + LightGBM ensemble on 5-minute bars with expanding window.
Target: 12-bar (1h) forward return.

Features computed via EnrichedFeatureComputer (Python, not Rust batch).
Uses greedy IC feature selection per fold.

Usage:
    python3 -m scripts.research.walkforward_5m_alpha
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
HORIZON = 12          # 12 bars = 1 hour forward return
ZSCORE_WINDOW = 2160  # 5m bars: 2160 = 7.5 days (similar to 720 for 1h)
WARMUP = 500
FOLD_TEST_BARS = 12 * 24 * 30 * 3  # 3 months of 5m bars = 25,920
MIN_TRAIN_BARS = 12 * 24 * 30 * 3  # 3 months minimum training
DEADZONE = 0.3
MIN_HOLD = 48         # 48 bars = 4 hours
MAX_HOLD = 180        # 180 bars = 15 hours
COST_BPS = 8          # roundtrip cost


# ── Feature computation (incremental, Python) ────────────────

def compute_5m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for 5m bars using vectorized operations.

    Faster than EnrichedFeatureComputer for batch processing.
    Returns feature DataFrame aligned to input.
    """
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(df)

    feat = {}

    # Returns
    for lag in [1, 3, 6, 12, 24, 48]:
        feat[f"ret_{lag}"] = pd.Series(closes).pct_change(lag).fillna(0).values

    # Moving averages
    for w in [5, 10, 20, 50]:
        ma = pd.Series(closes).rolling(w).mean().values
        feat[f"close_vs_ma{w}"] = (closes - ma) / np.where(ma > 0, ma, 1)

    # MA cross
    ma5 = pd.Series(closes).rolling(5).mean().values
    ma20 = pd.Series(closes).rolling(20).mean().values
    feat["ma_cross_5_20"] = (ma5 - ma20) / np.where(ma20 > 0, ma20, 1)

    # Volatility
    ret_1 = feat["ret_1"]
    for w in [5, 12, 24, 60]:
        feat[f"vol_{w}"] = pd.Series(ret_1).rolling(w).std().fillna(0).values

    feat["vol_ratio"] = feat["vol_5"] / np.where(feat["vol_24"] > 1e-10, feat["vol_24"], 1)

    # ATR
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - np.roll(closes, 1)),
                               np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    feat["atr_14"] = pd.Series(tr).rolling(14).mean().fillna(0).values / np.where(closes > 0, closes, 1)

    # Parkinson volatility
    hl_log = np.log(highs / np.where(lows > 0, lows, 1))
    feat["parkinson_vol"] = pd.Series(hl_log ** 2).rolling(20).mean().fillna(0).values / (4 * np.log(2))

    # RSI
    for period in [6, 14]:
        deltas = np.diff(closes, prepend=closes[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = pd.Series(gains).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(span=period, adjust=False).mean().values
        rs = avg_gain / np.where(avg_loss > 1e-10, avg_loss, 1e-10)
        feat[f"rsi_{period}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = pd.Series(closes).ewm(span=12).mean().values
    ema26 = pd.Series(closes).ewm(span=26).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9).mean().values
    feat["macd_hist"] = (macd - signal) / np.where(closes > 0, closes, 1) * 1000

    # Bollinger Band width
    ma20_s = pd.Series(closes).rolling(20).mean()
    std20 = pd.Series(closes).rolling(20).std()
    feat["bb_width_20"] = (std20 / ma20_s).fillna(0).values
    feat["bb_pctb"] = ((closes - (ma20_s - 2 * std20)) / (4 * std20 + 1e-10)).values

    # Volume features
    vol_ma = pd.Series(volumes).rolling(20).mean().values
    feat["vol_ma_ratio"] = volumes / np.where(vol_ma > 0, vol_ma, 1)

    # Taker buy ratio
    if "taker_buy_volume" in df.columns:
        tbv = df["taker_buy_volume"].values.astype(np.float64)
        feat["taker_buy_ratio"] = tbv / np.where(volumes > 0, volumes, 1)
        feat["taker_imbalance"] = 2 * feat["taker_buy_ratio"] - 1
    else:
        feat["taker_buy_ratio"] = np.full(n, 0.5)
        feat["taker_imbalance"] = np.zeros(n)

    # Hour/DOW cyclical (from timestamp)
    if "open_time" in df.columns:
        hours = (df["open_time"].values // (3600 * 1000)) % 24
        feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        dows = (df["open_time"].values // (86400 * 1000) + 4) % 7  # 0=Mon
        feat["dow_sin"] = np.sin(2 * np.pi * dows / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * dows / 7)

    return pd.DataFrame(feat, index=df.index)


# ── Greedy IC feature selection ──────────────────────────────

def greedy_ic_select(X: np.ndarray, y: np.ndarray, feature_names: list,
                     max_features: int = 15) -> list:
    """Select features by greedy IC maximization."""
    selected = []
    remaining = list(range(len(feature_names)))
    valid = ~np.isnan(y) & (np.abs(y) < 0.1)

    for _ in range(max_features):
        best_ic = -999
        best_idx = -1
        for idx in remaining:
            col = X[:, idx]
            mask = valid & ~np.isnan(col)
            if mask.sum() < 100:
                continue
            ic = abs(stats.spearmanr(col[mask], y[mask])[0])
            if ic > best_ic:
                best_ic = ic
                best_idx = idx
        if best_idx < 0 or best_ic < 0.005:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ── Walk-forward engine ──────────────────────────────────────

def run_walkforward(df: pd.DataFrame) -> list:
    """Run expanding-window walk-forward validation."""
    from sklearn.linear_model import Ridge
    import lightgbm as lgb

    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()
    feat_df = compute_5m_features(df)
    feature_names = list(feat_df.columns)
    X = feat_df.values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    print(f"done ({len(feature_names)} features, {time.time()-t0:.1f}s)")

    # Target: HORIZON-bar forward return
    target = np.concatenate([closes[HORIZON:] / closes[:-HORIZON] - 1, np.zeros(HORIZON)])

    n = len(df)
    results = []
    fold = 0
    test_start = MIN_TRAIN_BARS

    while test_start + FOLD_TEST_BARS <= n - HORIZON:
        test_end = test_start + FOLD_TEST_BARS

        X_train = X[:test_start]
        y_train = target[:test_start]
        X_test = X[test_start:test_end]
        y_test = target[test_start:test_end]

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # Feature selection
        sel_idx = greedy_ic_select(X_train, y_train, feature_names, max_features=15)
        if len(sel_idx) < 3:
            test_start += FOLD_TEST_BARS
            fold += 1
            continue
        sel_names = [feature_names[i] for i in sel_idx]

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]

        # Train Ridge
        ridge = Ridge(alpha=1.0).fit(X_tr_sel, y_train)
        pred_ridge = ridge.predict(X_te_sel)

        # Train LGBM
        lgbm = lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, num_leaves=15,
            learning_rate=0.05, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            verbosity=-1, n_jobs=1, random_state=42,
        )
        lgbm.fit(X_tr_sel, y_train)
        pred_lgbm = lgbm.predict(X_te_sel)

        # Ensemble: Ridge 60% + LGBM 40%
        pred = 0.6 * pred_ridge + 0.4 * pred_lgbm

        # IC
        valid = ~np.isnan(y_test) & (np.abs(y_test) < 0.1)
        ic = stats.spearmanr(pred[valid], y_test[valid])[0] if valid.sum() > 50 else 0

        # Signal simulation
        # Z-score
        all_pred = np.concatenate([ridge.predict(X_train[:, sel_idx]) * 0.6 +
                                    lgbm.predict(X_train[:, sel_idx]) * 0.4,
                                    pred])
        z_scores = np.zeros(len(pred))
        for i in range(len(pred)):
            window_start = max(0, len(X_train) + i - ZSCORE_WINDOW)
            window_end = len(X_train) + i + 1
            window = all_pred[window_start:window_end]
            if len(window) > WARMUP:
                mu = np.mean(window)
                sigma = np.std(window)
                z_scores[i] = (pred[i] - mu) / sigma if sigma > 1e-12 else 0

        # Simple Sharpe from signal
        closes_test = closes[test_start:test_end]
        ret_test = np.diff(closes_test) / closes_test[:-1]

        signal = np.zeros(len(z_scores))
        hold_count = 0
        cur_sig = 0
        for i in range(len(z_scores)):
            if hold_count < MIN_HOLD and cur_sig != 0:
                signal[i] = cur_sig
                hold_count += 1
            elif z_scores[i] > DEADZONE:
                signal[i] = 1
                if cur_sig != 1:
                    hold_count = 1
                    cur_sig = 1
                else:
                    hold_count += 1
            elif z_scores[i] < -DEADZONE:
                signal[i] = -1
                if cur_sig != -1:
                    hold_count = 1
                    cur_sig = -1
                else:
                    hold_count += 1
            else:
                signal[i] = 0
                cur_sig = 0
                hold_count = 0

            if hold_count >= MAX_HOLD:
                signal[i] = 0
                cur_sig = 0
                hold_count = 0

        # PnL
        pnl_bars = signal[:-1] * ret_test
        # Cost: charged on signal change
        changes = np.diff(signal, prepend=0)
        cost = np.abs(changes) * COST_BPS / 10000
        net_pnl = pnl_bars - cost[:-1]

        total_return = np.sum(net_pnl)
        n_trades = np.sum(np.abs(changes) > 0)

        # Sharpe
        if len(net_pnl) > 100 and np.std(net_pnl) > 0:
            # Annualize: 12*24*365 bars per year
            bars_per_year = 12 * 24 * 365
            sharpe = np.mean(net_pnl) / np.std(net_pnl) * np.sqrt(bars_per_year)
        else:
            sharpe = 0

        # Period label
        ts_start = df["open_time"].iloc[test_start] if "open_time" in df.columns else 0
        ts_end = df["open_time"].iloc[min(test_end - 1, n - 1)] if "open_time" in df.columns else 0
        from datetime import datetime, timezone
        p_start = datetime.fromtimestamp(ts_start / 1000, tz=timezone.utc).strftime("%Y-%m") if ts_start else "?"
        p_end = datetime.fromtimestamp(ts_end / 1000, tz=timezone.utc).strftime("%Y-%m") if ts_end else "?"

        results.append({
            "fold": fold,
            "period": f"{p_start}->{p_end}",
            "ic": ic,
            "sharpe": sharpe,
            "return_pct": total_return * 100,
            "trades": int(n_trades),
            "features": sel_names[:3],
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        print(f"  Fold {fold:>2d}: {p_start}->{p_end} IC={ic:+.4f} Sharpe={sharpe:+.2f} "
              f"Ret={total_return*100:+.1f}% Trades={n_trades:.0f} "
              f"Feats={sel_names[:3]}")

        fold += 1
        test_start += FOLD_TEST_BARS

    return results


# ── Main ─────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 100)
    print("  5m Alpha Walk-Forward Validation — ETHUSDT")
    print("=" * 100)

    df = pd.read_csv("data_files/ETHUSDT_5m.csv")
    print(f"\n  Data: {len(df):,} bars, "
          f"{df.shape[1]} columns")

    results = run_walkforward(df)

    # Summary
    print(f"\n{'=' * 100}")
    print(f"  WALK-FORWARD SUMMARY ({len(results)} folds)")
    print(f"{'=' * 100}")

    ics = [r["ic"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    returns = [r["return_pct"] for r in results]
    pos_sharpe = sum(1 for s in sharpes if s > 0)

    print(f"  Average IC:     {np.mean(ics):+.4f}")
    print(f"  Average Sharpe: {np.mean(sharpes):+.2f}")
    print(f"  Total return:   {sum(returns):+.1f}%")
    print(f"  Positive Sharpe: {pos_sharpe}/{len(results)}")

    threshold = max(len(results) * 2 // 3, 1)
    verdict = "PASS" if pos_sharpe >= threshold else "FAIL"
    print(f"\n  VERDICT: {pos_sharpe}/{len(results)} positive Sharpe "
          f"(need >= {threshold}) -> {verdict}")

    print(f"\n  Runtime: {time.time()-t0:.1f}s")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()

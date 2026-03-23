#!/usr/bin/env python3
"""Polymarket BTC Up/Down 5m -- ML classifier Walk-Forward validation.

Framework: Logistic (40%) + LightGBM (60%) classification ensemble
Target: predict P(Up), compare with CLOB market price, Kelly bet sizing

Features:
  - RSI(5, 14) mean reversion signals
  - Momentum (1/3/6/12/24 bar returns)
  - Volatility (12/60 bar rolling std)
  - Consecutive streak (up/down)
  - Hour cyclical (sin/cos encoding)
  - Interaction features (RSI x momentum)

Evaluation:
  - Accuracy vs RSI baseline
  - Log-loss (probability calibration quality)
  - Kelly PnL simulation
  - Quarterly stability

Usage:
    python3 -m scripts.research.polymarket_binary_alpha
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# --- Constants ---------------------------------------------------------------
COST_PER_WIN = 0.495   # Polymarket: win payout after fees
COST_PER_LOSS = 0.505  # Polymarket: loss cost after fees
BARS_PER_DAY = 12 * 24  # 5-min bars per day


# --- Feature Engineering -----------------------------------------------------

def compute_rsi(prices: np.ndarray, period: int = 5) -> np.ndarray:
    """Wilder's RSI."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros(len(deltas))
    avg_loss = np.zeros(len(deltas))
    if period > len(gains):
        return np.full(len(prices), 50.0)
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rs = avg_gain / np.where(avg_loss > 1e-10, avg_loss, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return np.concatenate([[50.0] * period, rsi[period - 1:]])


def compute_streak(results: np.ndarray) -> np.ndarray:
    """Consecutive Up(+) / Down(-) streak length."""
    streak = np.zeros(len(results))
    for i in range(1, len(results)):
        if results[i - 1] == 1:
            streak[i] = max(streak[i - 1], 0) + 1
        else:
            streak[i] = min(streak[i - 1], 0) - 1
    return streak


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from 5-minute OHLC data.

    Returns DataFrame with features aligned to df index.
    Target is next bar's result (shifted).
    """
    closes = df["close"].values.astype(np.float64)
    results = (df["result"] == "Up").astype(int).values
    hours = df["hour_utc"].values.astype(np.float64)

    feat = {}

    # RSI
    feat["rsi_5"] = compute_rsi(closes, 5)
    feat["rsi_14"] = compute_rsi(closes, 14)
    feat["rsi_5_norm"] = (feat["rsi_5"] - 50.0) / 50.0  # [-1, 1]

    # Momentum (returns)
    ret = np.concatenate([[0], np.diff(closes) / np.where(closes[:-1] > 0, closes[:-1], 1)])
    feat["ret_1"] = ret
    feat["ret_3"] = pd.Series(closes).pct_change(3).fillna(0).values
    feat["ret_6"] = pd.Series(closes).pct_change(6).fillna(0).values
    feat["ret_12"] = pd.Series(closes).pct_change(12).fillna(0).values
    feat["ret_24"] = pd.Series(closes).pct_change(24).fillna(0).values

    # Volatility
    ret_s = pd.Series(ret)
    feat["vol_12"] = ret_s.rolling(12, min_periods=1).std().fillna(0).values
    feat["vol_60"] = ret_s.rolling(60, min_periods=1).std().fillna(0).values
    feat["vol_ratio"] = feat["vol_12"] / np.where(feat["vol_60"] > 1e-10, feat["vol_60"], 1)

    # Streak
    feat["streak"] = compute_streak(results)
    feat["streak_abs"] = np.abs(feat["streak"])

    # Hour cyclical encoding
    feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Price distance from round number
    feat["round_dist"] = (closes % 1000) / 1000

    # Interaction features
    feat["rsi5_x_ret3"] = feat["rsi_5_norm"] * feat["ret_3"] * 100
    feat["rsi5_x_streak"] = feat["rsi_5_norm"] * feat["streak"] / 10
    feat["vol_x_ret1"] = feat["vol_12"] * feat["ret_1"] * 1000

    # Target: next bar result
    target = np.concatenate([results[1:], [0]])

    feat_df = pd.DataFrame(feat, index=df.index)
    feat_df["target"] = target

    return feat_df


FEATURE_NAMES = [
    "rsi_5", "rsi_14", "rsi_5_norm",
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "vol_12", "vol_60", "vol_ratio",
    "streak", "streak_abs",
    "hour_sin", "hour_cos",
    "round_dist",
    "rsi5_x_ret3", "rsi5_x_streak", "vol_x_ret1",
]


# --- Walk-Forward Engine -----------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    period: str
    n_test: int
    # RSI baseline
    rsi_acc: float
    rsi_signals: int
    rsi_pnl: float
    # Logistic
    lr_acc: float
    lr_logloss: float
    lr_signals: int
    lr_pnl: float
    # LightGBM
    lgbm_acc: float
    lgbm_logloss: float
    lgbm_signals: int
    lgbm_pnl: float
    # Ensemble
    ens_acc: float
    ens_logloss: float
    ens_signals: int
    ens_pnl: float
    # Top features
    top_features: List[str]


def rsi_baseline_pnl(rsi: np.ndarray, target: np.ndarray,
                     oversold: float = 25, overbought: float = 75,
                     bet_size: float = 10.0) -> Tuple[float, int, float]:
    """RSI rule-based baseline: accuracy, signal count, PnL."""
    correct = 0
    total = 0
    pnl = 0.0
    for i in range(len(rsi)):
        if rsi[i] < oversold:
            total += 1
            won = target[i] == 1
            if won:
                correct += 1
                pnl += COST_PER_WIN * bet_size
            else:
                pnl -= COST_PER_LOSS * bet_size
        elif rsi[i] > overbought:
            total += 1
            won = target[i] == 0
            if won:
                correct += 1
                pnl += COST_PER_WIN * bet_size
            else:
                pnl -= COST_PER_LOSS * bet_size
    acc = correct / total if total > 0 else 0.5
    return acc, total, pnl


def kelly_pnl(probs: np.ndarray, target: np.ndarray,
              market_price: float = 0.50, min_edge: float = 0.02,
              kelly_frac: float = 0.5, bankroll: float = 100.0,
              bet_cap_pct: float = 0.10) -> Tuple[int, float]:
    """Simulate Kelly-sized bets. Returns (n_bets, final_pnl)."""
    equity = bankroll
    n_bets = 0
    for i in range(len(probs)):
        p_up = probs[i]
        # Decide direction
        if p_up > market_price + min_edge:
            direction = "up"
            edge = p_up - market_price
            odds = (1 - market_price) / market_price
        elif p_up < market_price - min_edge:
            direction = "down"
            edge = market_price - p_up
            odds = market_price / (1 - market_price)
        else:
            continue

        # Kelly fraction
        kelly_f = edge * odds if odds > 0 else 0
        kelly_f = max(0, min(kelly_f, 0.25))  # cap at 25%
        bet = equity * kelly_f * kelly_frac
        bet = min(bet, equity * bet_cap_pct)  # cap per bet
        if bet < 1.0:  # min $1
            continue

        n_bets += 1
        won = (direction == "up" and target[i] == 1) or \
              (direction == "down" and target[i] == 0)
        if won:
            equity += bet * COST_PER_WIN
        else:
            equity -= bet * COST_PER_LOSS

        if equity < 1.0:
            break

    return n_bets, equity - bankroll


def run_walk_forward(
    df: pd.DataFrame,
    train_months: int = 3,
    test_months: int = 1,
    bet_size: float = 10.0,
    bankroll: float = 100.0,
) -> List[FoldResult]:
    """Walk-forward validation with Logistic + LightGBM + Ensemble."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    import lightgbm as lgb

    feat_df = build_features(df)
    X = feat_df[FEATURE_NAMES].values.astype(np.float64)
    y = feat_df["target"].values.astype(int)
    rsi5 = feat_df["rsi_5"].values

    bars_per_month = BARS_PER_DAY * 30
    n = len(df)
    results = []
    fold = 0
    start = 0

    while start + (train_months + test_months) * bars_per_month < n - 1:
        train_end = start + train_months * bars_per_month
        test_end = min(train_end + test_months * bars_per_month, n - 1)

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        rsi_test = rsi5[train_end:test_end]

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # --- RSI baseline ---
        rsi_acc, rsi_signals, rsi_pnl = rsi_baseline_pnl(
            rsi_test, y_test, bet_size=bet_size
        )

        # --- Logistic Regression ---
        lr = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42
        )
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        lr_pred = (lr_probs > 0.5).astype(int)
        lr_acc = float(np.mean(lr_pred == y_test))
        lr_ll = log_loss(y_test, lr_probs)
        lr_signals, lr_pnl = kelly_pnl(lr_probs, y_test, bankroll=bankroll)

        # --- LightGBM ---
        lgbm_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_pred = (lgbm_probs > 0.5).astype(int)
        lgbm_acc = float(np.mean(lgbm_pred == y_test))
        lgbm_ll = log_loss(y_test, lgbm_probs)
        lgbm_signals, lgbm_pnl = kelly_pnl(lgbm_probs, y_test, bankroll=bankroll)

        # --- Ensemble (LR 40% + LGBM 60%) ---
        ens_probs = 0.4 * lr_probs + 0.6 * lgbm_probs
        ens_pred = (ens_probs > 0.5).astype(int)
        ens_acc = float(np.mean(ens_pred == y_test))
        ens_ll = log_loss(y_test, ens_probs)
        ens_signals, ens_pnl = kelly_pnl(ens_probs, y_test, bankroll=bankroll)

        # Top LGBM features
        importances = lgbm_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_feats = [FEATURE_NAMES[i] for i in top_idx]

        period_start = df["datetime"].iloc[train_end][:7]
        period_end = df["datetime"].iloc[min(test_end, n - 1)][:7]

        results.append(FoldResult(
            fold=fold,
            period=f"{period_start}->{period_end}",
            n_test=test_end - train_end,
            rsi_acc=rsi_acc, rsi_signals=rsi_signals, rsi_pnl=rsi_pnl,
            lr_acc=lr_acc, lr_logloss=lr_ll, lr_signals=lr_signals, lr_pnl=lr_pnl,
            lgbm_acc=lgbm_acc, lgbm_logloss=lgbm_ll,
            lgbm_signals=lgbm_signals, lgbm_pnl=lgbm_pnl,
            ens_acc=ens_acc, ens_logloss=ens_ll,
            ens_signals=ens_signals, ens_pnl=ens_pnl,
            top_features=top_feats,
        ))

        fold += 1
        start += test_months * bars_per_month

    return results


# --- Calibration Analysis ----------------------------------------------------

def calibration_analysis(
    df: pd.DataFrame,
    train_months: int = 3,
    test_months: int = 1,
    n_bins: int = 10,
) -> Dict:
    """Collect all OOS predictions and analyze calibration."""
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    feat_df = build_features(df)
    X = feat_df[FEATURE_NAMES].values.astype(np.float64)
    y = feat_df["target"].values.astype(int)

    bars_per_month = BARS_PER_DAY * 30
    n = len(df)

    all_probs: list = []
    all_targets: list = []
    start = 0

    while start + (train_months + test_months) * bars_per_month < n - 1:
        train_end = start + train_months * bars_per_month
        test_end = min(train_end + test_months * bars_per_month, n - 1)

        X_train = np.nan_to_num(X[start:train_end], nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X[train_end:test_end], nan=0, posinf=0, neginf=0)
        y_train, y_test = y[start:train_end], y[train_end:test_end]

        lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)
        lr.fit(X_train, y_train)

        lgbm_model = lgb.LGBMClassifier(
            objective="binary", n_estimators=200, max_depth=4,
            num_leaves=15, learning_rate=0.05, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=-1, n_jobs=1,
        )
        lgbm_model.fit(X_train, y_train)

        ens_probs = 0.4 * lr.predict_proba(X_test)[:, 1] + \
                    0.6 * lgbm_model.predict_proba(X_test)[:, 1]
        all_probs.extend(ens_probs.tolist())
        all_targets.extend(y_test.tolist())

        start += test_months * bars_per_month

    probs = np.array(all_probs)
    targets = np.array(all_targets)

    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_data = []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_pred = float(probs[mask].mean())
            mean_actual = float(targets[mask].mean())
            cal_data.append({
                "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "n": int(mask.sum()),
                "pred_prob": mean_pred,
                "actual_rate": mean_actual,
                "gap": mean_actual - mean_pred,
            })

    return {"calibration": cal_data, "total": len(probs)}


# --- Main --------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 100)
    print("  Polymarket BTC Up/Down 5m -- ML Classifier Walk-Forward Validation")
    print("=" * 100)

    # Load data
    df = pd.read_csv("data/polymarket/btc_5m_updown_history.csv")
    print(f"\n  Data: {len(df):,} 5-min windows, "
          f"{df['datetime'].iloc[0][:10]} -> {df['datetime'].iloc[-1][:10]}")

    # Build features
    print("  Building features...", end=" ", flush=True)
    build_features(df)
    print(f"done ({len(FEATURE_NAMES)} features)")

    # Walk-forward
    print("  Running Walk-Forward validation...\n")
    results = run_walk_forward(df, train_months=3, test_months=1,
                               bet_size=10.0, bankroll=100.0)

    # --- Report: Accuracy -----------------------------------------------------
    print(f"{'~' * 100}")
    print(f"  {'Fold':>4s} {'Period':>16s} | {'RSI':>8s} {'Logistic':>9s} "
          f"{'LightGBM':>9s} {'Ensemble':>8s} | {'RSI_n':>5s} {'Ens_n':>5s} | Top Features")
    print(f"{'~' * 100}")

    for r in results:
        print(f"  {r.fold:>4d} {r.period:>16s} | "
              f"{r.rsi_acc:>7.2%} {r.lr_acc:>8.2%} "
              f"{r.lgbm_acc:>8.2%} {r.ens_acc:>7.2%} | "
              f"{r.rsi_signals:>5d} {r.ens_signals:>5d} | "
              f"{', '.join(r.top_features[:3])}")

    # --- Summary --------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  Summary ({len(results)} folds)")
    print(f"{'=' * 100}")

    def summarize(name, accs, logloss=None, signals=None, pnls=None):
        avg_acc = np.mean(accs)
        pos_folds = sum(1 for a in accs if a > 0.5)
        edge = avg_acc - 0.5
        line = f"  {name:<14s}: acc={avg_acc:.2%}  pos_fold={pos_folds}/{len(accs)}  edge={edge:+.2%}"
        if logloss is not None:
            line += f"  log-loss={np.mean(logloss):.4f}"
        if signals is not None:
            avg_s = np.mean(signals)
            line += f"  signals/day={avg_s/30:.1f}"
        if pnls is not None:
            avg_pnl = np.mean(pnls)
            total_pnl = sum(pnls)
            line += f"  avg_pnl=${avg_pnl:+.1f}  total=${total_pnl:+.1f}"
        print(line)

    summarize("RSI 25/75",
              [r.rsi_acc for r in results],
              signals=[r.rsi_signals for r in results],
              pnls=[r.rsi_pnl for r in results])
    summarize("Logistic",
              [r.lr_acc for r in results],
              logloss=[r.lr_logloss for r in results],
              signals=[r.lr_signals for r in results],
              pnls=[r.lr_pnl for r in results])
    summarize("LightGBM",
              [r.lgbm_acc for r in results],
              logloss=[r.lgbm_logloss for r in results],
              signals=[r.lgbm_signals for r in results],
              pnls=[r.lgbm_pnl for r in results])
    summarize("Ensemble",
              [r.ens_acc for r in results],
              logloss=[r.ens_logloss for r in results],
              signals=[r.ens_signals for r in results],
              pnls=[r.ens_pnl for r in results])

    # --- ML vs RSI comparison -------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  ML vs RSI Per-Fold Comparison")
    print(f"{'=' * 100}")

    ml_wins = 0
    for r in results:
        diff = r.ens_acc - r.rsi_acc
        better = "ML" if diff > 0 else "RSI" if diff < 0 else "TIE"
        if diff > 0:
            ml_wins += 1
        print(f"  Fold {r.fold:>2d} {r.period}: RSI={r.rsi_acc:.2%} vs Ens={r.ens_acc:.2%} "
              f"  d={diff:+.2%}  [{better}]")
    print(f"\n  ML wins: {ml_wins}/{len(results)} folds")

    # --- Calibration ----------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  Probability Calibration (Ensemble)")
    print(f"{'=' * 100}")

    cal = calibration_analysis(df)
    print(f"\n  {'Bin':<12s} {'Count':>8s} {'Pred P':>8s} {'Actual':>8s} {'Gap':>8s}")
    print(f"  {'~' * 48}")
    for c in cal["calibration"]:
        print(f"  {c['bin']:<12s} {c['n']:>8d} {c['pred_prob']:>7.3f} "
              f"{c['actual_rate']:>7.3f} {c['gap']:>+7.3f}")

    # --- Feature Importance ---------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  Feature Importance (LightGBM top-5 frequency)")
    print(f"{'=' * 100}")

    feat_counts: Dict[str, int] = {}
    for r in results:
        for f in r.top_features:
            feat_counts[f] = feat_counts.get(f, 0) + 1
    sorted_feats = sorted(feat_counts.items(), key=lambda x: -x[1])
    for name, count in sorted_feats:
        bar = "#" * count
        print(f"  {name:<18s} {count:>2d}/{len(results)} {bar}")

    # --- Compounding simulation -----------------------------------------------
    print(f"\n{'=' * 100}")
    print("  $100 Compounding Simulation (Ensemble, half-Kelly)")
    print(f"{'=' * 100}")

    equity = 100.0
    equity_history = [100.0]
    total_bets = 0
    fold_equities = []

    for r in results:
        eq_before = equity
        scale = equity / 100.0
        equity += r.ens_pnl * scale
        equity = max(equity, 1.0)
        total_bets += r.ens_signals
        fold_equities.append((r.period, eq_before, equity, r.ens_signals))
        equity_history.append(equity)

    print(f"\n  {'Period':>16s} {'Start$':>8s} {'End$':>8s} {'Bets':>6s}")
    print(f"  {'~' * 44}")
    for period, eq_b, eq_a, n_bets in fold_equities:
        print(f"  {period:>16s} ${eq_b:>7.1f} ${eq_a:>7.1f} {n_bets:>6d}")

    total_return = (equity - 100) / 100 * 100
    months = len(results)
    annual_return = ((equity / 100) ** (12 / months) - 1) * 100 if months > 0 else 0
    peak = 100.0
    max_dd = 0.0
    for eq in equity_history:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    print(f"\n  Final equity: ${equity:.1f}")
    print(f"  Total return: {total_return:+.1f}%")
    print(f"  Annualized: {annual_return:+.1f}%")
    print(f"  Max drawdown: {max_dd*100:.1f}%")
    print(f"  Total bets: {total_bets}")
    print(f"  Runtime: {time.time() - t0:.1f}s")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()

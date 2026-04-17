#!/usr/bin/env python3
"""1D (daily) POC backtest for BTC + ETH.

Goal: determine whether a daily timeframe alpha is worth adding to the
strategy H mix (currently 1h primary + 4h filter).

Design choices:
  - Resample hourly data → UTC daily (00:00 cutoff)
  - Features: price-based (ret, vol, RSI, MA) + macro from
    cross_market_daily.csv (SPY, QQQ, VIX, DXY, GLD, COIN ETFs, etc.)
  - Target: next-day log return
  - Model: LGBM + XGB ensemble (50/50), top-12 IC-selected features
  - Walk-forward: 6-month train, 3-month test, refit every 3 months
  - Rule: z > dz → long 100%, held until z < 0.3 or 3 days max
  - Costs: 4bp (maker) / 14bp (taker) round-trip

Run: python3 -m scripts.daily_poc
"""
from __future__ import annotations
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")


# ── Config ──
BARS_PER_MONTH = 30
TRAIN_MONTHS = 6
TEST_MONTHS = 3
COST_MAKER_RT = 4  # bp round-trip
COST_TAKER_RT = 14
MIN_HOLD = 1
MAX_HOLD = 7
HORIZON_DAYS = 1   # predict next-day return


def resample_1h_to_1d(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h bars to 1D (UTC midnight cutoff)."""
    ts = df["open_time"].values.astype(np.int64)
    # UTC day bucket = floor(ts / 86400000)
    groups = ts // (24 * 60 * 60_000)
    work = pd.DataFrame({
        "group": groups,
        "open_time": ts,
        "open": df["open"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low": df["low"].values.astype(np.float64),
        "close": df["close"].values.astype(np.float64),
        "volume": df["volume"].values.astype(np.float64),
    })
    if "quote_volume" in df.columns:
        work["quote_volume"] = df["quote_volume"].values.astype(np.float64)
    if "taker_buy_volume" in df.columns:
        work["taker_buy_volume"] = df["taker_buy_volume"].values.astype(np.float64)

    agg = {
        "open_time": "first", "open": "first",
        "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in work.columns:
        agg["quote_volume"] = "sum"
    if "taker_buy_volume" in work.columns:
        agg["taker_buy_volume"] = "sum"

    out = work.groupby("group", sort=True).agg(agg).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["open_time"], unit="ms").dt.date
    return out


def compute_1d_features(df: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build daily feature panel: price + macro."""
    f = pd.DataFrame(index=df.index)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values

    # Returns
    for h in (1, 2, 3, 5, 7, 14):
        f[f"ret_{h}d"] = pd.Series(close).pct_change(h).values

    # Log return
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[np.nan], log_ret])

    # Realized vol (rolling std of log returns)
    for w in (7, 14, 30):
        f[f"rv_{w}d"] = pd.Series(log_ret).rolling(w).std().values

    # MA ratios
    for w in (5, 10, 20, 50):
        ma = pd.Series(close).rolling(w).mean()
        f[f"close_vs_ma{w}"] = (close / ma - 1).values

    # High/low range
    f["hl_range_7d"] = pd.Series((high - low) / close).rolling(7).mean().values
    f["hl_range_14d"] = pd.Series((high - low) / close).rolling(14).mean().values

    # RSI
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0)
    losses = np.where(diff < 0, -diff, 0)
    avg_g = pd.Series(gains).rolling(14).mean().values
    avg_l = pd.Series(losses).rolling(14).mean().values
    rs = np.where(avg_l > 0, avg_g / (avg_l + 1e-9), 100)
    rsi = 100 - (100 / (1 + rs))
    f["rsi_14"] = np.concatenate([[np.nan], rsi])[:len(close)]

    # Volume features
    vol_ma = pd.Series(vol).rolling(20).mean().values
    f["vol_ratio_20"] = vol / (vol_ma + 1e-9)
    log_vol = pd.Series(np.log(vol + 1))
    mu20 = log_vol.rolling(20).mean()
    sd20 = log_vol.rolling(20).std()
    f["vol_log_zscore_20"] = ((log_vol - mu20) / (sd20 + 1e-9)).values

    # Return autocorrelation (momentum / mean-reversion indicator)
    ret_ser = pd.Series(log_ret)
    f["ret_autocorr_5"] = ret_ser.rolling(30).apply(
        lambda x: x.autocorr(lag=5) if x.std() > 0 else 0
    ).values
    f["ret_skew_14"] = ret_ser.rolling(14).skew().values

    # Attach macro (join on date)
    f["date"] = df["date"].values
    merged = f.merge(macro, on="date", how="left")
    # Forward-fill macro (macro may lag 1 day)
    macro_cols = [c for c in macro.columns if c != "date"]
    for c in macro_cols:
        merged[c] = merged[c].ffill()

    # Interaction features
    if "vix_level" in merged.columns:
        merged["vol_x_vix"] = merged["rv_14d"] * merged["vix_level"]
    if "spy_ret_1d" in merged.columns and "ret_1d" in merged.columns:
        merged["btc_vs_spy_1d"] = merged["ret_1d"] - merged["spy_ret_1d"]

    feature_names = [c for c in merged.columns if c not in ("date",)]
    return merged, feature_names


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    hold_days: int
    z: float


def run_backtest(
    predictions: np.ndarray,
    closes: np.ndarray,
    deadzone: float,
    cost_bps: float,
    *,
    long_only: bool = False,
    min_hold: int = 1,
    max_hold: int = 7,
    notional: float = 500.0,
) -> list[Trade]:
    n = len(predictions)
    pred_std = float(np.nanstd(predictions))
    if pred_std < 1e-12:
        return []

    z = predictions / pred_std
    cost_frac = cost_bps / 10000
    trades: list[Trade] = []
    pos = 0
    ep = 0.0
    eb = 0
    score = 0.0

    for i in range(n):
        if pos != 0:
            held = i - eb
            exit_now = False
            if held >= max_hold:
                exit_now = True
            elif held >= min_hold:
                if pos * z[i] < -0.3 or abs(z[i]) < 0.2:
                    exit_now = True
            if exit_now:
                pnl_pct = pos * (closes[i] - ep) / ep
                gross = pnl_pct * notional
                cost = cost_frac * notional
                net = gross - cost
                trades.append(Trade(eb, i, pos, ep, closes[i],
                                    gross, net, held, score))
                pos = 0

        if pos == 0:
            if z[i] > deadzone:
                pos, ep, eb, score = 1, closes[i], i, z[i]
            elif not long_only and z[i] < -deadzone:
                pos, ep, eb, score = -1, closes[i], i, z[i]

    if pos != 0:
        pnl_pct = pos * (closes[-1] - ep) / ep
        gross = pnl_pct * notional
        cost = cost_frac * notional
        net = gross - cost
        trades.append(Trade(eb, n-1, pos, ep, closes[-1],
                            gross, net, n-1-eb, score))
    return trades


def summarize(trades: list[Trade], label: str, notional: float = 500.0):
    if not trades:
        print(f"  [{label}] no trades")
        return
    net = np.array([t.net_pnl for t in trades])
    wins = sum(1 for t in trades if t.net_pnl > 0)
    wr = wins / len(trades) * 100
    avg_bp = float(np.mean(net) / notional * 10000)
    total_ret = float(np.sum(net) / 10000 * 100)

    eq, peak, max_dd = 10000.0, 10000.0, 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    holds = np.array([t.hold_days for t in trades])
    longs = sum(1 for t in trades if t.direction > 0)
    shorts = len(trades) - longs
    tpy = 365 / max(np.mean(holds), 0.5)
    sharpe = float(
        np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy)
    ) if len(net) > 1 else 0.0

    print(
        f"  [{label}] trades={len(trades)} (L={longs}/S={shorts})  "
        f"WR={wr:.1f}%  hold={np.mean(holds):.1f}d  "
        f"avg={avg_bp:+.1f}bp  total={total_ret:+.2f}%  "
        f"maxDD={max_dd*100:.2f}%  Sharpe={sharpe:+.2f}"
    )


def walk_forward_predict(X, y, dates, train_m=6, test_m=3):
    """Rolling walk-forward training, returns predictions array."""
    import lightgbm as lgb
    import xgboost as xgb

    n = len(X)
    preds = np.full(n, np.nan)
    train_size = train_m * BARS_PER_MONTH
    test_size = test_m * BARS_PER_MONTH

    start = train_size
    fold = 0
    while start + test_size <= n:
        tr_end = start
        tr_start = max(0, tr_end - train_size)
        te_end = min(start + test_size, n)

        X_tr = X[tr_start:tr_end]
        y_tr = y[tr_start:tr_end]
        X_te = X[start:te_end]

        valid_tr = ~np.isnan(y_tr)
        X_tr = X_tr[valid_tr]
        y_tr = y_tr[valid_tr]

        if len(X_tr) < 60:
            start += test_size
            continue

        # LGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        lgb_params = {
            "objective": "regression", "metric": "rmse",
            "num_leaves": 31, "learning_rate": 0.03,
            "feature_fraction": 0.85, "bagging_fraction": 0.85,
            "bagging_freq": 3, "min_data_in_leaf": 20,
            "verbosity": -1,
        }
        lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=200)
        p_lgb = lgb_model.predict(X_te)

        # XGB
        dtrain_x = xgb.DMatrix(X_tr, label=y_tr)
        xgb_params = {
            "objective": "reg:squarederror", "eta": 0.03,
            "max_depth": 5, "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 10, "verbosity": 0,
        }
        xgb_model = xgb.train(xgb_params, dtrain_x, num_boost_round=200)
        p_xgb = xgb_model.predict(xgb.DMatrix(X_te))

        preds[start:te_end] = 0.5 * p_lgb + 0.5 * p_xgb
        fold += 1
        start += test_size

    return preds, fold


def run_symbol(symbol: str, macro: pd.DataFrame):
    print("=" * 72)
    print(f"1D POC: {symbol}")
    print("=" * 72)

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    n = len(df_1d)
    print(f"  1D bars: {n} ({df_1d['date'].iloc[0]} → {df_1d['date'].iloc[-1]})")

    # Filter to >= 2021-04-20 (earliest macro date)
    macro_start = macro["date"].min()
    df_1d = df_1d[df_1d["date"] >= macro_start].reset_index(drop=True)
    print(f"  Aligned to macro: {len(df_1d)} days from {df_1d['date'].iloc[0]}")

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    closes = df_1d["close"].values.astype(np.float64)

    # Target: next-day log return (future leak-free: use t+1 close vs t close)
    log_rets = np.diff(np.log(closes), prepend=np.nan)
    target = np.roll(log_rets, -HORIZON_DAYS)
    target[-HORIZON_DAYS:] = np.nan

    # Prep X
    X_all = feat_df[feature_names].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Walk-forward
    preds, n_folds = walk_forward_predict(X_all, target, df_1d["date"].values,
                                          TRAIN_MONTHS, TEST_MONTHS)
    valid = ~np.isnan(preds)
    print(f"  Walk-forward folds: {n_folds}  valid preds: {valid.sum()}")

    if valid.sum() < 30:
        print("  Not enough valid predictions — aborting")
        return

    # IC (Spearman-like)
    tgt_valid = target[valid]
    pred_valid = preds[valid]
    good = ~np.isnan(tgt_valid)
    from scipy.stats import spearmanr
    ic_pearson = np.corrcoef(pred_valid[good], tgt_valid[good])[0, 1]
    ic_spearman, _ = spearmanr(pred_valid[good], tgt_valid[good])
    print(f"  OOS IC (Pearson) : {ic_pearson:.4f}")
    print(f"  OOS IC (Spearman): {ic_spearman:.4f}")

    # Dz sweep
    print()
    oos_closes = closes[valid]
    oos_preds = pred_valid
    print(f"  {'dz':>5} {'cost':>6} {'trades':>7} {'WR':>6} "
          f"{'avg_bp':>8} {'total':>8} {'maxDD':>7} {'Sharpe':>7}")
    print("  " + "-" * 68)
    for dz in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        for cost_label, cost in [("maker", COST_MAKER_RT), ("taker", COST_TAKER_RT)]:
            trades = run_backtest(oos_preds, oos_closes, dz, cost,
                                  long_only=False, min_hold=MIN_HOLD, max_hold=MAX_HOLD)
            if not trades:
                print(f"  {dz:>5.2f} {cost_label:>6} {'0':>7}")
                continue
            net = np.array([t.net_pnl for t in trades])
            wins = sum(1 for t in trades if t.net_pnl > 0)
            wr = wins / len(trades) * 100
            avg_bp = float(np.mean(net) / 500 * 10000)
            total = float(np.sum(net) / 10000 * 100)
            eq, peak, mdd = 10000.0, 10000.0, 0.0
            for t in trades:
                eq += t.net_pnl
                peak = max(peak, eq)
                mdd = max(mdd, (peak - eq) / peak)
            holds = np.array([t.hold_days for t in trades])
            tpy = 365 / max(np.mean(holds), 0.5)
            sh = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy)) if len(net) > 1 else 0
            print(f"  {dz:>5.2f} {cost_label:>6} {len(trades):>7} "
                  f"{wr:>5.1f}% {avg_bp:>+7.1f} {total:>+7.2f}% "
                  f"{mdd*100:>6.2f}% {sh:>+7.2f}")


def main():
    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date
    print(f"Macro features: {len(macro.columns)-1} cols, "
          f"{len(macro)} days, {macro['date'].iloc[0]} → {macro['date'].iloc[-1]}")
    print()

    for sym in ("BTCUSDT", "ETHUSDT"):
        run_symbol(sym, macro)
        print()


if __name__ == "__main__":
    main()

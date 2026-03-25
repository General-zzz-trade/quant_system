#!/usr/bin/env python3
"""Explore 4h and daily alpha: IC scan, WF validation, cost analysis.

Key hypothesis: longer timeframes should have:
1. Higher IC (less noise, more signal)
2. Lower cost impact (fewer trades)
3. Better cross-market feature alignment (daily features → daily bars = no forward-fill)
4. But less data (fewer bars for training/validation)

Compares: 15m, 1h, 4h, daily side-by-side.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data_files")
MODEL_DIR = Path("models_v8")

FEE = 0.0004
SLIP = 0.0003
COST = FEE + SLIP


def load_1h(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}_1h.csv")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def load_cross_market() -> pd.DataFrame:
    path = DATA_DIR / "cross_market_daily.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h bars to 4h."""
    df = df_1h.copy()
    df = df.set_index("datetime")
    ohlcv = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    if "quote_volume" in df.columns:
        ohlcv["quote_volume"] = df["quote_volume"].resample("4h").sum()
    if "taker_buy_volume" in df.columns:
        ohlcv["taker_buy_volume"] = df["taker_buy_volume"].resample("4h").sum()
    if "trades" in df.columns:
        ohlcv["trades"] = df["trades"].resample("4h").sum()

    ohlcv = ohlcv.reset_index()
    return ohlcv


def resample_to_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h bars to daily."""
    df = df_1h.copy()
    df = df.set_index("datetime")
    ohlcv = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    if "quote_volume" in df.columns:
        ohlcv["quote_volume"] = df["quote_volume"].resample("1D").sum()
    if "taker_buy_volume" in df.columns:
        ohlcv["taker_buy_volume"] = df["taker_buy_volume"].resample("1D").sum()
    if "trades" in df.columns:
        ohlcv["trades"] = df["trades"].resample("1D").sum()

    ohlcv = ohlcv.reset_index()
    return ohlcv


def compute_features(df: pd.DataFrame, cross_market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute comprehensive features for any timeframe."""
    feats = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    # Returns
    for w in [1, 3, 5, 10, 20, 50]:
        feats[f"ret_{w}"] = close.pct_change(w)

    # Volatility
    ret = close.pct_change()
    for w in [5, 10, 20, 50]:
        feats[f"vol_{w}"] = ret.rolling(w).std()

    feats["vol_ratio_5_20"] = feats["vol_5"] / feats["vol_20"]
    feats["vol_ratio_10_50"] = feats["vol_10"] / feats["vol_50"]

    # RSI
    for w in [6, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        feats[f"rsi_{w}"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Moving average features
    for w in [5, 10, 20, 50]:
        ma = close.rolling(w).mean()
        feats[f"close_ma{w}_ratio"] = close / ma - 1

    # Bollinger
    for w in [20]:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        feats[f"bb_pos_{w}"] = (close - ma) / std.replace(0, np.nan)

    # Volume features
    feats["vol_ma5_ratio"] = volume / volume.rolling(5).mean()
    feats["vol_ma20_ratio"] = volume / volume.rolling(20).mean()

    # Taker features
    if "taker_buy_volume" in df.columns:
        tbr = df["taker_buy_volume"] / volume.replace(0, np.nan)
        feats["taker_buy_ratio"] = tbr
        feats["taker_imbalance"] = 2 * tbr - 1
        feats["taker_imb_cum5"] = feats["taker_imbalance"].rolling(5).sum()

    # Range
    bar_range = (df["high"] - df["low"]) / close
    feats["bar_range"] = bar_range
    feats["bar_range_ma5_ratio"] = bar_range / bar_range.rolling(5).mean()

    # Distance from rolling high/low
    for w in [10, 20, 50]:
        feats[f"dist_high_{w}"] = close / df["high"].rolling(w).max() - 1
        feats[f"dist_low_{w}"] = close / df["low"].rolling(w).min() - 1

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()
    feats["macd_norm"] = macd / close
    feats["macd_hist_norm"] = (macd - macd_signal) / close

    # Cross-market features (merge by date, T-1 shifted)
    if cross_market is not None and len(cross_market) > 0:
        df_dates = df["datetime"].dt.date if hasattr(df["datetime"].dt, "date") else df["datetime"]
        cm = cross_market.copy()
        cm["date"] = pd.to_datetime(cm["date"]).dt.date
        cm = cm.set_index("date")
        # T-1 shift: shift index forward by 1 day so bar on date D uses data from D-1
        cm.index = [d + pd.Timedelta(days=1) for d in cm.index]

        for col in cm.columns:
            if col == "date":
                continue
            mapped = pd.Series(df_dates).map(
                lambda d: cm[col].get(d, np.nan) if d in cm.index else np.nan
            )
            # Forward-fill (weekends/holidays)
            mapped = mapped.ffill()
            feats[f"cm_{col}"] = mapped.values

    return feats


def ic_scan(feats: pd.DataFrame, close: pd.Series, horizons: list[int],
            timeframe_label: str) -> pd.DataFrame:
    """IC scan for features against forward returns."""
    results = []
    for h in horizons:
        fwd = close.pct_change(h).shift(-h)
        for col in feats.columns:
            vals = feats[col]
            mask = vals.notna() & fwd.notna() & np.isfinite(vals) & np.isfinite(fwd)
            if mask.sum() < 200:
                continue
            ic, pval = stats.spearmanr(vals[mask], fwd[mask])
            results.append({
                "feature": col,
                "horizon": h,
                "ic": ic,
                "abs_ic": abs(ic),
                "pval": pval,
                "n": int(mask.sum()),
                "timeframe": timeframe_label,
            })
    return pd.DataFrame(results)


def signal_pipeline(close: pd.Series, feats: pd.DataFrame, dz: float,
                     min_hold: int, max_hold: int, long_only: bool,
                     monthly_gate: bool, sma_window: int = 480) -> pd.Series:
    """Generate signals from momentum composite + z-score pipeline."""
    # Momentum composite
    pred = pd.Series(0.0, index=close.index)
    for w in [5, 10, 20, 50]:
        col = f"ret_{w}"
        if col in feats.columns:
            pred += feats[col].fillna(0) * (0.3 if w <= 10 else 0.2)

    # Z-score
    z = (pred - pred.rolling(max(100, min(720, len(close) // 3)), min_periods=50).mean()) / \
        pred.rolling(max(100, min(720, len(close) // 3)), min_periods=50).std()

    signal = pd.Series(0, index=close.index)
    signal[z > dz] = 1
    signal[z < -dz] = -1

    if long_only:
        signal[signal < 0] = 0

    if monthly_gate:
        sma = close.rolling(sma_window, min_periods=sma_window // 2).mean()
        signal[close < sma] = signal[close < sma].clip(upper=0)

    # Min hold
    current = 0
    hold = 0
    for i in range(len(signal)):
        s = signal.iloc[i]
        if s != current and s != 0:
            current = s
            hold = 1
        elif current != 0 and hold < min_hold:
            signal.iloc[i] = current
            hold += 1
        elif s != current:
            current = s
            hold = 1 if s != 0 else 0

    # Max hold
    current = 0
    hold = 0
    for i in range(len(signal)):
        if signal.iloc[i] != 0:
            if signal.iloc[i] == current:
                hold += 1
                if hold > max_hold:
                    signal.iloc[i] = 0
                    current = 0
                    hold = 0
            else:
                current = signal.iloc[i]
                hold = 1
        else:
            current = 0
            hold = 0

    return signal


def backtest(close: pd.Series, signal: pd.Series, leverage: float = 10.0,
             initial_equity: float = 500.0) -> dict:
    """Simple backtest with costs."""
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    pos = 0
    entry = 0.0
    trades = []

    for i in range(1, len(close)):
        s = signal.iloc[i]
        c = close.iloc[i]

        if s != pos:
            if pos != 0 and entry > 0:
                raw = pos * (c / entry - 1)
                net = raw - 2 * COST
                pnl = equity * leverage * 0.15 * net  # 15% of equity per trade
                equity += pnl
                trades.append({"ret": net, "pnl": pnl})

            if s != 0:
                pos = s
                entry = c
            else:
                pos = 0
                entry = 0

        peak = max(peak, equity)
        dd = (equity - peak) / peak * 100 if peak > 0 else 0
        max_dd = min(max_dd, dd)

    if not trades:
        return {"sharpe": 0, "total_ret": 0, "max_dd": 0, "n_trades": 0,
                "win_rate": 0, "final_equity": equity, "trades_per_year": 0}

    rets = [t["ret"] * leverage for t in trades]
    n_bars = len(close)
    bars_per_year = {"15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}

    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(len(trades)) if np.std(rets) > 0 else 0
    win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100

    return {
        "sharpe": sharpe,
        "total_ret": (equity / initial_equity - 1) * 100,
        "max_dd": max_dd,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "final_equity": equity,
    }


def walkforward(close: pd.Series, signal: pd.Series, min_bars: int,
                fold_bars: int, leverage: float = 10.0) -> dict:
    """Walk-forward OOS validation."""
    n_folds = (len(close) - min_bars) // fold_bars
    if n_folds < 3:
        return {"n_folds": 0, "pass_rate": 0, "mean_sharpe": 0, "mean_dd": 0}

    fold_results = []
    for f in range(n_folds):
        start = min_bars + f * fold_bars
        end = min(start + fold_bars, len(close))
        c = close.iloc[start:end].reset_index(drop=True)
        s = signal.iloc[start:end].reset_index(drop=True)
        r = backtest(c, s, leverage)
        fold_results.append(r)

    sharpes = [r["sharpe"] for r in fold_results if r["n_trades"] > 0]
    dds = [r["max_dd"] for r in fold_results if r["n_trades"] > 0]
    n_pass = sum(1 for s in sharpes if s > 0)

    return {
        "n_folds": len(sharpes),
        "pass_rate": n_pass / len(sharpes) if sharpes else 0,
        "mean_sharpe": np.mean(sharpes) if sharpes else 0,
        "median_sharpe": np.median(sharpes) if sharpes else 0,
        "mean_dd": np.mean(dds) if dds else 0,
        "worst_dd": min(dds) if dds else 0,
        "total_trades": sum(r["n_trades"] for r in fold_results),
    }


def explore_timeframe(df: pd.DataFrame, tf_label: str, symbol: str,
                       cross_market: pd.DataFrame, bars_per_day: int):
    """Full exploration for one timeframe."""
    print(f"\n  --- {tf_label} ({len(df)} bars, {len(df)/bars_per_day:.0f} days) ---")

    feats = compute_features(df, cross_market)

    # Forward return horizons (normalized to equivalent real-time)
    # All horizons in bars at this TF
    if tf_label == "15m":
        horizons = [4, 16, 48, 96, 192, 384]  # 1h, 4h, 12h, 1d, 2d, 4d
        horizon_labels = ["1h", "4h", "12h", "1d", "2d", "4d"]
    elif tf_label == "1h":
        horizons = [1, 4, 12, 24, 48, 96]
        horizon_labels = ["1h", "4h", "12h", "1d", "2d", "4d"]
    elif tf_label == "4h":
        horizons = [1, 3, 6, 12, 24, 42]
        horizon_labels = ["4h", "12h", "1d", "2d", "4d", "7d"]
    else:  # daily
        horizons = [1, 2, 3, 5, 10, 20]
        horizon_labels = ["1d", "2d", "3d", "5d", "10d", "20d"]

    ic_results = ic_scan(feats, df["close"], horizons, tf_label)

    if len(ic_results) == 0:
        print("    No IC results (insufficient data)")
        return None

    # Top features by IC
    best = ic_results.loc[ic_results.groupby("feature")["abs_ic"].idxmax()]
    best = best.sort_values("abs_ic", ascending=False)

    # Separate crypto-native vs cross-market
    crypto_feats = best[~best["feature"].str.startswith("cm_")]
    cm_feats = best[best["feature"].str.startswith("cm_")]

    print("\n    Top 10 crypto-native features:")
    for _, r in crypto_feats.head(10).iterrows():
        sig = "***" if r["pval"] < 0.001 else "**" if r["pval"] < 0.01 else "*" if r["pval"] < 0.05 else ""
        h_label = horizon_labels[horizons.index(r["horizon"])] if r["horizon"] in horizons else f"{r['horizon']}b"
        print(f"      {r['abs_ic']:.5f} {sig:3s}  {r['feature']:30s}  h={h_label}")

    if len(cm_feats) > 0:
        print("\n    Top 10 cross-market features:")
        for _, r in cm_feats.head(10).iterrows():
            sig = "***" if r["pval"] < 0.001 else "**" if r["pval"] < 0.01 else "*" if r["pval"] < 0.05 else ""
            h_label = horizon_labels[horizons.index(r["horizon"])] if r["horizon"] in horizons else f"{r['horizon']}b"
            print(f"      {r['abs_ic']:.5f} {sig:3s}  {r['feature']:30s}  h={h_label}")

    # Best overall IC
    overall_best = best.head(1).iloc[0]
    print(f"\n    Overall best: {overall_best['feature']} IC={overall_best['ic']:+.5f}")

    # Signal pipeline + WF validation
    # Adjust parameters by timeframe
    if tf_label == "4h":
        dz, mh, maxh = 0.8, 6, 36  # 6 bars = 24h, 36 bars = 6d
        sma_w = 120  # 120 x 4h = 20 days
        wf_min, wf_fold = 1000, 180  # ~167d warmup, 30d folds
    elif tf_label == "1d":
        dz, mh, maxh = 0.6, 2, 10  # 2 days min, 10 days max
        sma_w = 20  # 20 days
        wf_min, wf_fold = 200, 30  # ~200d warmup, 30d folds
    elif tf_label == "1h":
        dz, mh, maxh = 1.0, 24, 144
        sma_w = 480
        wf_min, wf_fold = 4000, 720
    else:  # 15m
        dz, mh, maxh = 0.8, 16, 128
        sma_w = 1920
        wf_min, wf_fold = 8000, 2880

    long_only = True
    monthly_gate = "BTC" in symbol

    sig = signal_pipeline(df["close"], feats, dz, mh, maxh, long_only, monthly_gate, sma_w)

    n_active = (sig != 0).sum()
    n_changes = (sig.diff().abs() > 0).sum()
    print(f"\n    Signal: {n_changes} changes, {n_active} active bars ({100*n_active/len(sig):.1f}%)")

    # Full backtest
    full = backtest(df["close"], sig, leverage=10.0)
    print(f"    Full backtest: Sharpe={full['sharpe']:.2f}, Ret={full['total_ret']:.1f}%, "
          f"MaxDD={full['max_dd']:.1f}%, Trades={full['n_trades']}, WR={full['win_rate']:.1f}%")

    # Walk-forward
    wf = walkforward(df["close"], sig, wf_min, wf_fold, leverage=10.0)
    if wf["n_folds"] > 0:
        print(f"    Walk-forward: {int(wf['pass_rate']*wf['n_folds'])}/{wf['n_folds']} PASS "
              f"({wf['pass_rate']*100:.0f}%), Mean Sharpe={wf['mean_sharpe']:.2f}, "
              f"MeanDD={wf['mean_dd']:.1f}%, WorstDD={wf['worst_dd']:.1f}%")
    else:
        print("    Walk-forward: insufficient data for folds")

    # Cost analysis
    bars_per_year_map = {"15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
    bpy = bars_per_year_map[tf_label]
    avg_hold = n_active / max(n_changes // 2, 1)  # bars per trade
    trades_per_year = bpy / avg_hold if avg_hold > 0 else 0
    annual_cost_bps = trades_per_year * 2 * (FEE + SLIP) * 10000  # round trip

    print("\n    Cost profile:")
    print(f"      Avg hold: {avg_hold:.1f} bars ({avg_hold * 24/bars_per_day:.1f}h)")
    print(f"      Trades/year: {trades_per_year:.0f}")
    print(f"      Annual cost: {annual_cost_bps:.0f} bps")

    return {
        "tf": tf_label,
        "bars": len(df),
        "best_ic": overall_best["abs_ic"],
        "best_feature": overall_best["feature"],
        "sharpe": full["sharpe"],
        "total_ret": full["total_ret"],
        "max_dd": full["max_dd"],
        "n_trades": full["n_trades"],
        "wf_pass": wf["pass_rate"] if wf["n_folds"] > 0 else 0,
        "wf_folds": wf["n_folds"],
        "wf_sharpe": wf["mean_sharpe"],
        "trades_per_year": trades_per_year,
        "annual_cost_bps": annual_cost_bps,
    }


def main():
    cross_market = load_cross_market()
    print(f"Cross-market data: {len(cross_market)} days, {len(cross_market.columns)} columns")

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n{'='*70}")
        print(f"  {symbol} — Multi-Timeframe Alpha Exploration")
        print(f"{'='*70}")

        df_1h = load_1h(symbol)
        print(f"  1h data: {len(df_1h)} bars")

        # Load 15m if available
        path_15m = DATA_DIR / f"{symbol}_15m.csv"
        if path_15m.exists():
            df_15m = pd.read_csv(path_15m)
            df_15m["datetime"] = pd.to_datetime(df_15m["open_time"], unit="ms")
            df_15m = df_15m.sort_values("open_time").reset_index(drop=True)
        else:
            df_15m = None

        # Resample
        df_4h = resample_to_4h(df_1h)
        df_1d = resample_to_daily(df_1h)

        print(f"  4h data: {len(df_4h)} bars")
        print(f"  Daily data: {len(df_1d)} bars")

        results = []

        # 15m
        if df_15m is not None and len(df_15m) > 10000:
            r = explore_timeframe(df_15m, "15m", symbol, cross_market, 96)
            if r:
                results.append(r)

        # 1h
        r = explore_timeframe(df_1h, "1h", symbol, cross_market, 24)
        if r:
            results.append(r)

        # 4h
        r = explore_timeframe(df_4h, "4h", symbol, cross_market, 6)
        if r:
            results.append(r)

        # Daily
        r = explore_timeframe(df_1d, "1d", symbol, cross_market, 1)
        if r:
            results.append(r)

        # Comparison table
        if results:
            print(f"\n  {'='*70}")
            print(f"  {symbol} — Timeframe Comparison")
            print(f"  {'='*70}")
            print(f"  {'TF':>4} {'Bars':>7} {'BestIC':>8} {'Sharpe':>8} {'Ret%':>8} "
                  f"{'MaxDD%':>8} {'Trades':>7} {'WF%':>5} {'WF_S':>6} "
                  f"{'T/yr':>6} {'Cost':>6}")
            print(f"  {'-'*82}")
            for r in results:
                print(f"  {r['tf']:>4} {r['bars']:>7} {r['best_ic']:>8.5f} "
                      f"{r['sharpe']:>8.2f} {r['total_ret']:>8.1f} "
                      f"{r['max_dd']:>8.1f} {r['n_trades']:>7} "
                      f"{r['wf_pass']*100:>5.0f} {r['wf_sharpe']:>6.2f} "
                      f"{r['trades_per_year']:>6.0f} {r['annual_cost_bps']:>6.0f}")


if __name__ == "__main__":
    main()

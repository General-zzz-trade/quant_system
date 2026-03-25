#!/usr/bin/env python3
# ruff: noqa: E402,E501,E701,E702,E741,F841,E401
"""Backtest: 5m entry timing + position scaling for 1h/15m alpha.

Tests 3 approaches vs baseline:
1. Baseline: immediate entry on 1h signal
2. Entry Timer: wait for 5m oversold/overbought before entry
3. Position Scaler: scale size by 5m conditions
4. Combined: Timer + Scaler

Uses momentum-composite 1h signals with 5m execution.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

MODEL_DIR = Path("models_v8")
DATA_DIR = Path("data_files")

FEE_BPS = 0.0004       # 4 bps taker
SLIPPAGE_BPS = 0.0003  # 3 bps
COST = FEE_BPS + SLIPPAGE_BPS


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    size_scale: float = 1.0  # position scaling factor
    entry_delay_bars: int = 0  # how many 5m bars we waited

    @property
    def gross_ret(self) -> float:
        return self.direction * (self.exit_price / self.entry_price - 1) * self.size_scale

    @property
    def net_ret(self) -> float:
        return self.gross_ret - 2 * COST * self.size_scale  # round trip


def load_1h_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1h.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def load_5m_data(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_5m.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def generate_1h_signals(df_1h: pd.DataFrame, symbol: str) -> pd.Series:
    """Generate 1h alpha signals using z-score of momentum composite."""
    model_dir = "BTCUSDT_gate_v2" if "BTC" in symbol else "ETHUSDT_gate_v2"
    cfg_path = MODEL_DIR / model_dir / "config.json"
    cfg = json.load(open(cfg_path))
    dz = cfg.get("deadzone", 1.0)
    min_hold = cfg.get("min_hold", 24)
    long_only = cfg.get("long_only", False)
    monthly_gate = cfg.get("monthly_gate", False)

    close = df_1h["close"]

    # Momentum composite as prediction proxy
    feats = pd.DataFrame(index=df_1h.index)
    for w in [5, 10, 20, 50]:
        feats[f"ret_{w}"] = close.pct_change(w)

    pred = (feats["ret_5"] * 0.3 + feats["ret_10"] * 0.3 +
            feats["ret_20"] * 0.2 + feats["ret_50"] * 0.2)

    # Z-score
    z_window = 720
    z = (pred - pred.rolling(z_window, min_periods=180).mean()) / \
        pred.rolling(z_window, min_periods=180).std()

    # Discretize
    signal = pd.Series(0, index=df_1h.index)
    signal[z > dz] = 1
    signal[z < -dz] = -1

    if long_only:
        signal[signal < 0] = 0

    if monthly_gate:
        sma480 = close.rolling(480).mean()
        signal[close < sma480] = signal[close < sma480].clip(upper=0)

    # Min hold
    current_sig = 0
    hold_count = 0
    for i in range(len(signal)):
        new_sig = signal.iloc[i]
        if new_sig != current_sig and new_sig != 0:
            current_sig = new_sig
            hold_count = 1
        elif current_sig != 0 and hold_count < min_hold:
            signal.iloc[i] = current_sig
            hold_count += 1
        elif new_sig != current_sig:
            current_sig = new_sig
            hold_count = 1 if new_sig != 0 else 0

    return signal


def compute_5m_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Compute 5m microstructure features for entry timing."""
    feats = pd.DataFrame(index=df_5m.index)

    close = df_5m["close"]
    volume = df_5m["volume"]
    tbv = df_5m["taker_buy_volume"]
    tsv = volume - tbv

    # Bollinger band position (best MR feature)
    for w in [12, 24]:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        feats[f"bb_pos_{w}"] = (close - ma) / std.replace(0, np.nan)

    # VWAP deviation
    for w in [12, 24]:
        vwap = df_5m["quote_volume"].rolling(w).sum() / volume.rolling(w).sum()
        feats[f"vwap_dev_{w}"] = (close - vwap) / vwap

    # CVD
    vd = tbv - tsv
    feats["cvd_6"] = vd.rolling(6).sum()
    feats["cvd_12"] = vd.rolling(12).sum()

    # Taker imbalance
    tbr = tbv / volume.replace(0, np.nan)
    feats["taker_imb_cum6"] = (2 * tbr - 1).rolling(6).sum()
    feats["taker_imb_cum12"] = (2 * tbr - 1).rolling(12).sum()

    # RSI 6 bar
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(6).mean()
    loss = (-delta.clip(upper=0)).rolling(6).mean()
    feats["rsi_6"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Momentum
    feats["mom_6"] = close.pct_change(6)
    feats["mom_12"] = close.pct_change(12)

    return feats


def align_1h_to_5m(sig_1h: pd.Series, dt_1h: pd.Series, dt_5m: pd.Series) -> pd.Series:
    """Forward-fill 1h signals to 5m timeframe."""
    sig_map = pd.Series(sig_1h.values, index=dt_1h.values)
    sig_5m = sig_map.reindex(dt_5m.values, method="ffill").fillna(0).astype(int)
    sig_5m.index = range(len(sig_5m))
    return sig_5m


def backtest_baseline(sig_5m: pd.Series, df_5m: pd.DataFrame, leverage: float = 10.0) -> dict:
    """Baseline: immediate entry/exit on 1h signal."""
    trades = []
    pos = 0
    entry_price = 0.0
    entry_time = None

    for i in range(1, len(df_5m)):
        new_sig = sig_5m.iloc[i]
        if new_sig != pos:
            if pos != 0:
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=df_5m["datetime"].iloc[i],
                    direction=pos,
                    entry_price=entry_price,
                    exit_price=df_5m["close"].iloc[i],
                ))
            if new_sig != 0:
                pos = new_sig
                entry_price = df_5m["close"].iloc[i]
                entry_time = df_5m["datetime"].iloc[i]
            else:
                pos = 0

    return _calc_stats(trades, df_5m, leverage, "Baseline (immediate)")


def backtest_entry_timer(sig_5m: pd.Series, df_5m: pd.DataFrame, feats_5m: pd.DataFrame,
                          max_wait: int = 12, threshold: float = 0.5,
                          leverage: float = 10.0) -> dict:
    """Entry Timer: wait for favorable 5m condition before entering."""
    trades = []
    pos = 0
    entry_price = 0.0
    entry_time = None
    waiting = False
    wait_direction = 0
    wait_start = 0

    bb = feats_5m["bb_pos_12"]

    for i in range(1, len(df_5m)):
        new_sig = sig_5m.iloc[i]

        if new_sig != pos and not waiting:
            if pos != 0:
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=df_5m["datetime"].iloc[i],
                    direction=pos,
                    entry_price=entry_price,
                    exit_price=df_5m["close"].iloc[i],
                ))
                pos = 0

            if new_sig != 0:
                waiting = True
                wait_direction = new_sig
                wait_start = i

        if waiting:
            if sig_5m.iloc[i] != wait_direction and sig_5m.iloc[i] != pos:
                waiting = False
                continue

            bars_waited = i - wait_start
            bb_val = bb.iloc[i] if pd.notna(bb.iloc[i]) else 0

            favorable = False
            if wait_direction == 1 and bb_val < -threshold:
                favorable = True
            elif wait_direction == -1 and bb_val > threshold:
                favorable = True

            if favorable or bars_waited >= max_wait:
                pos = wait_direction
                entry_price = df_5m["close"].iloc[i]
                entry_time = df_5m["datetime"].iloc[i]
                waiting = False

    if pos != 0:
        trades.append(Trade(
            entry_time=entry_time,
            exit_time=df_5m["datetime"].iloc[-1],
            direction=pos,
            entry_price=entry_price,
            exit_price=df_5m["close"].iloc[-1],
        ))

    return _calc_stats(trades, df_5m, leverage, f"Entry Timer (wait≤{max_wait}, thr={threshold})")


def backtest_position_scaler(sig_5m: pd.Series, df_5m: pd.DataFrame, feats_5m: pd.DataFrame,
                              leverage: float = 10.0) -> dict:
    """Position Scaler: adjust size based on 5m MR conditions."""
    trades = []
    pos = 0
    entry_price = 0.0
    entry_time = None
    entry_scale = 1.0

    bb = feats_5m["bb_pos_12"]

    for i in range(1, len(df_5m)):
        new_sig = sig_5m.iloc[i]

        if new_sig != pos:
            if pos != 0:
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=df_5m["datetime"].iloc[i],
                    direction=pos,
                    entry_price=entry_price,
                    exit_price=df_5m["close"].iloc[i],
                    size_scale=entry_scale,
                ))

            if new_sig != 0:
                bb_val = bb.iloc[i] if pd.notna(bb.iloc[i]) else 0

                if new_sig == 1:  # long
                    if bb_val < -1.0:
                        scale = 1.2
                    elif bb_val < -0.5:
                        scale = 1.0
                    elif bb_val < 0:
                        scale = 0.7
                    elif bb_val < 0.5:
                        scale = 0.5
                    else:
                        scale = 0.3
                else:  # short
                    if bb_val > 1.0:
                        scale = 1.2
                    elif bb_val > 0.5:
                        scale = 1.0
                    elif bb_val > 0:
                        scale = 0.7
                    elif bb_val > -0.5:
                        scale = 0.5
                    else:
                        scale = 0.3

                pos = new_sig
                entry_price = df_5m["close"].iloc[i]
                entry_time = df_5m["datetime"].iloc[i]
                entry_scale = scale
            else:
                pos = 0

    return _calc_stats(trades, df_5m, leverage, "Position Scaler")


def backtest_combined(sig_5m: pd.Series, df_5m: pd.DataFrame, feats_5m: pd.DataFrame,
                       max_wait: int = 12, threshold: float = 0.5,
                       leverage: float = 10.0) -> dict:
    """Combined: Entry Timer + Position Scaler."""
    trades = []
    pos = 0
    entry_price = 0.0
    entry_time = None
    entry_scale = 1.0
    waiting = False
    wait_direction = 0
    wait_start = 0

    bb = feats_5m["bb_pos_12"]

    for i in range(1, len(df_5m)):
        new_sig = sig_5m.iloc[i]

        if new_sig != pos and not waiting:
            if pos != 0:
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=df_5m["datetime"].iloc[i],
                    direction=pos,
                    entry_price=entry_price,
                    exit_price=df_5m["close"].iloc[i],
                    size_scale=entry_scale,
                ))
                pos = 0

            if new_sig != 0:
                waiting = True
                wait_direction = new_sig
                wait_start = i

        if waiting:
            if sig_5m.iloc[i] != wait_direction and sig_5m.iloc[i] != pos:
                waiting = False
                continue

            bars_waited = i - wait_start
            bb_val = bb.iloc[i] if pd.notna(bb.iloc[i]) else 0

            favorable = False
            if wait_direction == 1 and bb_val < -threshold:
                favorable = True
            elif wait_direction == -1 and bb_val > threshold:
                favorable = True

            if favorable or bars_waited >= max_wait:
                if wait_direction == 1:
                    if bb_val < -1.0:
                        scale = 1.2
                    elif bb_val < -0.5:
                        scale = 1.0
                    elif bb_val < 0:
                        scale = 0.7
                    elif bb_val < 0.5:
                        scale = 0.5
                    else:
                        scale = 0.3
                else:
                    if bb_val > 1.0:
                        scale = 1.2
                    elif bb_val > 0.5:
                        scale = 1.0
                    elif bb_val > 0:
                        scale = 0.7
                    elif bb_val > -0.5:
                        scale = 0.5
                    else:
                        scale = 0.3

                pos = wait_direction
                entry_price = df_5m["close"].iloc[i]
                entry_time = df_5m["datetime"].iloc[i]
                entry_scale = scale
                waiting = False

    if pos != 0:
        trades.append(Trade(
            entry_time=entry_time,
            exit_time=df_5m["datetime"].iloc[-1],
            direction=pos,
            entry_price=entry_price,
            exit_price=df_5m["close"].iloc[-1],
            size_scale=entry_scale,
        ))

    return _calc_stats(trades, df_5m, leverage, f"Combined (wait≤{max_wait}, thr={threshold})")


def _calc_stats(trades: list[Trade], df_5m: pd.DataFrame, leverage: float, label: str) -> dict:
    """Calculate strategy statistics."""
    if not trades:
        return {"label": label, "n_trades": 0, "sharpe": 0, "net_total": 0,
                "gross_total": 0, "win_rate": 0, "max_dd": 0, "avg_scale": 1.0,
                "trades_per_day": 0, "daily_mean_bps": 0, "daily_std_bps": 0,
                "avg_entry_delay": 0}

    n_days = (df_5m["datetime"].iloc[-1] - df_5m["datetime"].iloc[0]).days

    gross_rets = [t.gross_ret * leverage for t in trades]
    net_rets = [t.net_ret * leverage for t in trades]
    scales = [t.size_scale for t in trades]

    # Daily returns for Sharpe
    daily_pnl = {}
    for t in trades:
        day = t.entry_time.date()
        if day not in daily_pnl:
            daily_pnl[day] = 0.0
        daily_pnl[day] += t.net_ret * leverage

    daily_series = pd.Series(daily_pnl).sort_index()
    all_days = pd.date_range(daily_series.index[0], daily_series.index[-1], freq="D")
    daily_series = daily_series.reindex(all_days, fill_value=0.0)

    sharpe = daily_series.mean() / daily_series.std() * np.sqrt(365) if daily_series.std() > 0 else 0
    win_rate = sum(1 for r in net_rets if r > 0) / len(net_rets) * 100
    cum = daily_series.cumsum()
    dd = (cum - cum.cummax()).min() * 100

    return {
        "label": label,
        "n_trades": len(trades),
        "trades_per_day": len(trades) / max(n_days, 1),
        "sharpe": sharpe,
        "gross_total": sum(gross_rets) * 100,
        "net_total": sum(net_rets) * 100,
        "win_rate": win_rate,
        "max_dd": dd,
        "avg_scale": np.mean(scales),
        "daily_mean_bps": daily_series.mean() * 10000,
        "daily_std_bps": daily_series.std() * 10000,
        "avg_entry_delay": 0,
    }


def run_parameter_sweep(sig_5m, df_5m, feats_5m, symbol, leverage=10.0):
    """Sweep entry timer parameters."""
    print(f"\n  --- {symbol} Parameter Sweep ---")

    results = []
    for max_wait in [6, 12, 18, 24]:
        for threshold in [0.3, 0.5, 0.7, 1.0]:
            r = backtest_combined(sig_5m, df_5m, feats_5m, max_wait, threshold, leverage)
            r["max_wait"] = max_wait
            r["threshold"] = threshold
            results.append(r)

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values("sharpe", ascending=False)
    print("\n  Top 10 parameter combinations:")
    print(f"  {'Wait':>5} {'Thr':>5} {'Sharpe':>8} {'Net%':>8} {'WR%':>6} {'MaxDD%':>8} {'Trades':>7} {'AvgScale':>9}")
    for _, r in rdf.head(10).iterrows():
        print(f"  {r['max_wait']:5.0f} {r['threshold']:5.1f} {r['sharpe']:8.2f} {r['net_total']:8.1f} "
              f"{r['win_rate']:6.1f} {r['max_dd']:8.1f} {r['n_trades']:7.0f} {r['avg_scale']:9.2f}")

    return rdf


def main():
    leverage = 10.0

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n{'='*70}")
        print(f"  {symbol} — 5m Entry Optimization Backtest (leverage={leverage}x)")
        print(f"{'='*70}")

        df_1h = load_1h_data(symbol)
        df_5m = load_5m_data(symbol)
        print(f"  1h bars: {len(df_1h)}, 5m bars: {len(df_5m)}")

        sig_1h = generate_1h_signals(df_1h, symbol)
        sig_changes = (sig_1h.diff().abs() > 0).sum()
        print(f"  1h signals: {sig_changes} changes, {(sig_1h != 0).sum()} active bars")

        sig_5m = align_1h_to_5m(sig_1h, df_1h["datetime"], df_5m["datetime"])
        feats_5m = compute_5m_features(df_5m)

        start = 300
        sig_5m = sig_5m.iloc[start:].reset_index(drop=True)
        df_5m_trim = df_5m.iloc[start:].reset_index(drop=True)
        feats_5m = feats_5m.iloc[start:].reset_index(drop=True)

        print("\n  Running backtests...")
        results = []

        r1 = backtest_baseline(sig_5m, df_5m_trim, leverage)
        results.append(r1)

        r2 = backtest_entry_timer(sig_5m, df_5m_trim, feats_5m, max_wait=12, threshold=0.5, leverage=leverage)
        results.append(r2)

        r3 = backtest_position_scaler(sig_5m, df_5m_trim, feats_5m, leverage=leverage)
        results.append(r3)

        r4 = backtest_combined(sig_5m, df_5m_trim, feats_5m, max_wait=12, threshold=0.5, leverage=leverage)
        results.append(r4)

        print(f"\n  {'Strategy':<45} {'Sharpe':>8} {'Net%':>9} {'WR%':>6} {'MaxDD%':>8} {'Trades':>7} {'AvgScale':>9} {'DailyBps':>10}")
        print(f"  {'-'*103}")
        for r in results:
            print(f"  {r['label']:<45} {r['sharpe']:8.2f} {r['net_total']:9.1f} {r['win_rate']:6.1f} "
                  f"{r['max_dd']:8.1f} {r['n_trades']:7.0f} {r['avg_scale']:9.2f} {r['daily_mean_bps']:10.1f}")

        if r1["sharpe"] != 0:
            for r in results[1:]:
                sharpe_imp = (r["sharpe"] / r1["sharpe"] - 1) * 100 if r1["sharpe"] != 0 else 0
                ret_imp = r["net_total"] - r1["net_total"]
                dd_imp = r["max_dd"] - r1["max_dd"]
                print(f"\n  {r['label']}: Sharpe {sharpe_imp:+.1f}%, Return {ret_imp:+.1f}%, MaxDD {dd_imp:+.1f}%")

        run_parameter_sweep(sig_5m, df_5m_trim, feats_5m, symbol, leverage)


if __name__ == "__main__":
    main()

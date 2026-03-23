#!/usr/bin/env python3
"""
Backtest V2 Multi-Timeframe Tick Strategy from 1m Kline Data
=============================================================
Simulates aggTrade-like signals from 1-minute OHLCV bars:
  - Flow imbalance (3s/10s/30s windows → approximated from 1m volume/close direction)
  - Large trade clustering
  - EMA price momentum (fast/medium/slow)
  - Volume spike detection
  - Confidence filtering (count of agreeing signals)

Runs parameter sweep and compares V1 (single-window) vs V2 (multi-TF ensemble).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

DATA_PATH = "/quant_system/data_files/BTCUSDT_1m.csv"

# ─── Signal Computation ──────────────────────────────────────────────

def ema(series: np.ndarray, halflife_bars: float) -> np.ndarray:
    """EMA with given halflife in bars."""
    alpha = 1 - np.exp(-np.log(2) / max(halflife_bars, 0.01))
    out = np.zeros_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute V2-style tick signals from 1m bars."""
    close = df["close"].values
    volume = df["volume"].values
    df["high"].values
    df["low"].values
    open_ = df["open"].values
    n = len(df)

    # --- Direction: +1 if close > open, -1 if close < open ---
    direction = np.sign(close - open_)
    ret = np.diff(close, prepend=close[0])
    zero_mask = direction == 0
    direction[zero_mask] = np.sign(ret[zero_mask])

    # --- Signed flow (volume * direction) ---
    signed_flow = volume * direction

    # --- Multi-TF flow imbalance (window in bars: 3s≈1bar, 10s≈1bar, 30s≈1bar for 1m data) ---
    # Since we have 1m bars, we approximate:
    # 3s window ≈ current bar only (flow_1)
    # 10s window ≈ current bar (flow_1) — same at 1m resolution
    # 30s window ≈ current bar (flow_1) — same at 1m resolution
    # Better approximation: use rolling windows of 1, 3, 10 bars
    def rolling_flow_imbalance(signed_flow, volume, window):
        buy_vol = np.where(signed_flow > 0, volume, 0)
        sell_vol = np.where(signed_flow < 0, volume, 0)
        cum_buy = pd.Series(buy_vol).rolling(window, min_periods=1).sum().values
        cum_sell = pd.Series(sell_vol).rolling(window, min_periods=1).sum().values
        total = cum_buy + cum_sell
        imbalance = np.where(total > 0, (cum_buy - cum_sell) / total, 0.0)
        return imbalance

    flow_fast = rolling_flow_imbalance(signed_flow, volume, 1)   # ~3s (1 bar)
    flow_med  = rolling_flow_imbalance(signed_flow, volume, 3)   # ~10s (3 bars)
    flow_slow = rolling_flow_imbalance(signed_flow, volume, 10)  # ~30s (10 bars)

    # --- Multi-TF ensemble ---
    # All agree
    all_agree = (np.sign(flow_fast) == np.sign(flow_med)) & (np.sign(flow_med) == np.sign(flow_slow)) & (flow_fast != 0)
    # Fast+med agree
    fast_med_agree = (np.abs(flow_fast) > 0.3) & (np.sign(flow_fast) == np.sign(flow_med)) & ~all_agree

    ensemble = np.zeros(n)
    ensemble[all_agree] = 0.5 * flow_fast[all_agree] + 0.3 * flow_med[all_agree] + 0.2 * flow_slow[all_agree]
    ensemble[fast_med_agree] = 0.6 * flow_fast[fast_med_agree] + 0.4 * flow_med[fast_med_agree]

    # --- V1 flow (single 10-bar window for comparison) ---
    flow_v1 = rolling_flow_imbalance(signed_flow, volume, 5)

    # --- Price momentum (EMA-based) ---
    ret_1 = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
    ema_fast = ema(ret_1, 1.0)    # ~1s halflife
    ema_med  = ema(ret_1, 5.0)    # ~5s halflife
    ema_slow = ema(ret_1, 30.0)   # ~30s halflife
    momentum = 0.5 * ema_fast + 0.3 * ema_med + 0.2 * ema_slow
    # Normalize to [-1, 1] range
    mom_std = pd.Series(momentum).rolling(100, min_periods=10).std().values
    mom_std = np.maximum(mom_std, 1e-10)
    momentum_norm = np.clip(momentum / (3 * mom_std), -1, 1)

    # --- Volume spike ---
    vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_spike = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_spike_signal = np.where(vol_spike > 3.0, np.sign(direction) * np.minimum(vol_spike / 5.0, 1.0), 0.0)

    # --- Large trade clustering (approximated: high volume + strong direction) ---
    large_threshold = 5.0
    is_large = vol_spike > large_threshold
    # Cluster: 2+ consecutive large trades in same direction
    cluster = np.zeros(n)
    for i in range(1, n):
        if is_large[i] and is_large[i-1] and direction[i] == direction[i-1]:
            cluster[i] = direction[i] * min(vol_spike[i] / large_threshold, 1.0)

    # --- Combined V2 tick score ---
    # 0.45*ensemble + 0.25*momentum + 0.20*cluster + 0.10*vol_spike
    tick_score_v2 = 0.45 * ensemble + 0.25 * momentum_norm + 0.20 * cluster + 0.10 * vol_spike_signal

    # --- V1 tick score (simple: flow + volume) ---
    tick_score_v1 = 0.7 * flow_v1 + 0.3 * vol_spike_signal

    # --- Confidence (count of agreeing signals / total) ---
    signals = np.column_stack([
        np.sign(ensemble),
        np.sign(momentum_norm),
        np.sign(cluster),
        np.sign(vol_spike_signal)
    ])
    dominant_sign = np.sign(tick_score_v2)
    agreement = np.zeros(n)
    for i in range(n):
        if dominant_sign[i] != 0:
            non_zero = np.sum(signals[i] != 0)
            if non_zero > 0:
                agreement[i] = np.sum(signals[i] == dominant_sign[i]) / max(non_zero, 1)

    df = df.copy()
    df["flow_fast"] = flow_fast
    df["flow_med"] = flow_med
    df["flow_slow"] = flow_slow
    df["ensemble"] = ensemble
    df["flow_v1"] = flow_v1
    df["momentum"] = momentum_norm
    df["vol_spike"] = vol_spike_signal
    df["cluster"] = cluster
    df["tick_v2"] = tick_score_v2
    df["tick_v1"] = tick_score_v1
    df["confidence"] = agreement

    return df

# ─── Backtest Engine ──────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    max_dd: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    monthly_pnl: dict = field(default_factory=dict)

def run_backtest(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    min_confidence: float,
    min_hold_bars: int,
    max_hold_bars: int,
    cost_bps: float,
    notional: float = 300.0,
    min_interval_bars: int = 1,
) -> BacktestResult:
    """Run backtest using precomputed signals."""
    close = df["close"].values
    scores = df[score_col].values
    confidence = df["confidence"].values
    n = len(close)

    result = BacktestResult()
    equity = 10000.0
    peak_equity = equity
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_bar = 0
    last_trade_bar = -min_interval_bars

    equity_list = [equity]

    for i in range(1, n):
        month_key = str(df.index[i])[:7] if hasattr(df.index[i], 'strftime') else str(i // 43200)

        # --- Check exit ---
        if position != 0:
            bars_held = i - entry_bar

            # Exit conditions
            should_exit = False
            if bars_held >= max_hold_bars:
                should_exit = True  # max hold
            elif bars_held >= min_hold_bars:
                # Exit on signal reversal or score going flat
                if position == 1 and scores[i] < -threshold * 0.5:
                    should_exit = True
                elif position == -1 and scores[i] > threshold * 0.5:
                    should_exit = True
                elif abs(scores[i]) < threshold * 0.3:
                    should_exit = True  # signal faded

            if should_exit:
                # Close position
                pnl_pct = position * (close[i] - entry_price) / entry_price
                trade_cost = 2 * cost_bps / 10000  # entry + exit
                gross = pnl_pct * notional
                cost = trade_cost * notional
                net = gross - cost

                result.total_trades += 1
                result.gross_pnl += gross
                result.costs += cost
                result.net_pnl += net
                if gross > 0:
                    result.wins += 1

                equity += net
                if month_key not in result.monthly_pnl:
                    result.monthly_pnl[month_key] = 0.0
                result.monthly_pnl[month_key] += net

                position = 0
                last_trade_bar = i

        # --- Check entry ---
        if position == 0 and (i - last_trade_bar) >= min_interval_bars:
            score = scores[i]
            conf = confidence[i] if score_col == "tick_v2" else 1.0  # V1 has no confidence

            if abs(score) >= threshold and conf >= min_confidence:
                position = 1 if score > 0 else -1
                entry_price = close[i]
                entry_bar = i

        # Track equity
        equity_list.append(equity)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        result.max_dd = max(result.max_dd, dd)

    # Close any open position
    if position != 0:
        pnl_pct = position * (close[-1] - entry_price) / entry_price
        trade_cost = 2 * cost_bps / 10000
        gross = pnl_pct * notional
        cost = trade_cost * notional
        result.total_trades += 1
        result.gross_pnl += gross
        result.costs += cost
        result.net_pnl += gross - cost
        equity += gross - cost
        equity_list.append(equity)

    result.equity_curve = equity_list
    return result

# ─── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("V2 Multi-Timeframe Tick Strategy Backtest")
    print("=" * 70)

    # Load data
    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Expect columns: timestamp, open, high, low, close, volume
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"

    print(f"  Loaded {len(df):,} bars ({len(df)/1440:.0f} days)")
    print(f"  Date range: {df.iloc[0].get('timestamp', 'N/A')} → {df.iloc[-1].get('timestamp', 'N/A')}")

    # Use last 3 months for backtest (avoid overfitting on full dataset)
    n_bars = min(len(df), 3 * 30 * 1440)  # ~3 months of 1m bars
    df = df.tail(n_bars).reset_index(drop=True)
    print(f"  Using last {len(df):,} bars for backtest")

    # Compute signals
    print("\nComputing V1 and V2 signals ...")
    df = compute_signals(df)

    # Signal statistics
    print("\n--- Signal Statistics ---")
    print(f"  V2 tick_score: mean={df['tick_v2'].mean():.4f}, std={df['tick_v2'].std():.4f}, "
          f"|score|>0.2: {(df['tick_v2'].abs() > 0.2).sum():,} bars ({(df['tick_v2'].abs() > 0.2).mean()*100:.1f}%)")
    print(f"  V1 tick_score: mean={df['tick_v1'].mean():.4f}, std={df['tick_v1'].std():.4f}, "
          f"|score|>0.2: {(df['tick_v1'].abs() > 0.2).sum():,} bars ({(df['tick_v1'].abs() > 0.2).mean()*100:.1f}%)")
    print(f"  Confidence≥0.5: {(df['confidence'] >= 0.5).sum():,} bars ({(df['confidence'] >= 0.5).mean()*100:.1f}%)")
    print(f"  Ensemble≠0: {(df['ensemble'] != 0).sum():,} bars ({(df['ensemble'] != 0).mean()*100:.1f}%)")

    # ─── Parameter Sweep ───
    print("\n" + "=" * 70)
    print("Parameter Sweep")
    print("=" * 70)

    configs = [
        # (label, score_col, threshold, min_conf, min_hold, max_hold, cost_bps)
        ("V1-baseline-7bps",    "tick_v1", 0.20, 0.0, 1, 30, 7.0),
        ("V1-baseline-2bps",    "tick_v1", 0.20, 0.0, 1, 30, 2.0),
        ("V2-default-7bps",     "tick_v2", 0.20, 0.5, 1, 30, 7.0),
        ("V2-default-2bps",     "tick_v2", 0.20, 0.5, 1, 30, 2.0),
        ("V2-tight-7bps",       "tick_v2", 0.30, 0.6, 2, 20, 7.0),
        ("V2-tight-2bps",       "tick_v2", 0.30, 0.6, 2, 20, 2.0),
        ("V2-wide-7bps",        "tick_v2", 0.15, 0.4, 1, 60, 7.0),
        ("V2-wide-2bps",        "tick_v2", 0.15, 0.4, 1, 60, 2.0),
        ("V2-aggressive-7bps",  "tick_v2", 0.10, 0.3, 1, 15, 7.0),
        ("V2-aggressive-2bps",  "tick_v2", 0.10, 0.3, 1, 15, 2.0),
        ("V2-conservative-7bps","tick_v2", 0.40, 0.7, 3, 30, 7.0),
        ("V2-conservative-2bps","tick_v2", 0.40, 0.7, 3, 30, 2.0),
    ]

    print(f"\n{'Config':<25} {'Trades':>7} {'WinR':>6} {'Gross$':>9} {'Cost$':>9} {'Net$':>9} {'Net%':>7} {'MaxDD':>7}")
    print("-" * 85)

    results = {}
    for label, score_col, thresh, min_conf, min_hold, max_hold, cost in configs:
        r = run_backtest(df, score_col, thresh, min_conf, min_hold, max_hold, cost)
        results[label] = r
        win_rate = r.wins / max(r.total_trades, 1) * 100
        net_pct = r.net_pnl / 10000 * 100
        print(f"{label:<25} {r.total_trades:>7,} {win_rate:>5.1f}% {r.gross_pnl:>+9.0f} {r.costs:>9.0f} {r.net_pnl:>+9.0f} {net_pct:>+6.1f}% {r.max_dd:>6.1f}%")  # noqa: E501

    # ─── Monthly Breakdown for best V2 config ───
    print("\n" + "=" * 70)
    print("Monthly P&L Breakdown (V2-default-2bps)")
    print("=" * 70)

    best_key = "V2-default-2bps"
    if best_key in results and results[best_key].monthly_pnl:
        monthly = results[best_key].monthly_pnl
        print(f"\n{'Month':<12} {'P&L':>10} {'Cum P&L':>10}")
        print("-" * 35)
        cum = 0
        for month in sorted(monthly.keys()):
            cum += monthly[month]
            print(f"{month:<12} {monthly[month]:>+10.1f} {cum:>+10.1f}")

    # ─── V1 vs V2 Comparison ───
    print("\n" + "=" * 70)
    print("V1 vs V2 Comparison (2bps cost)")
    print("=" * 70)

    v1 = results.get("V1-baseline-2bps")
    v2 = results.get("V2-default-2bps")
    if v1 and v2:
        print(f"\n{'Metric':<25} {'V1':>12} {'V2':>12} {'Delta':>12}")
        print("-" * 65)
        print(f"{'Trades':<25} {v1.total_trades:>12,} {v2.total_trades:>12,} {v2.total_trades - v1.total_trades:>+12,}")
        v1wr = v1.wins/max(v1.total_trades,1)*100
        v2wr = v2.wins/max(v2.total_trades,1)*100
        print(f"{'Win Rate':<25} {v1wr:>11.1f}% {v2wr:>11.1f}% {v2wr-v1wr:>+11.1f}%")
        print(f"{'Gross P&L':<25} {v1.gross_pnl:>+12.0f} {v2.gross_pnl:>+12.0f} {v2.gross_pnl-v1.gross_pnl:>+12.0f}")
        print(f"{'Costs':<25} {v1.costs:>12.0f} {v2.costs:>12.0f} {v2.costs-v1.costs:>+12.0f}")
        print(f"{'Net P&L':<25} {v1.net_pnl:>+12.0f} {v2.net_pnl:>+12.0f} {v2.net_pnl-v1.net_pnl:>+12.0f}")
        print(f"{'Max Drawdown':<25} {v1.max_dd:>11.1f}% {v2.max_dd:>11.1f}% {v2.max_dd-v1.max_dd:>+11.1f}%")

        # Per-trade metrics
        v1_avg = v1.net_pnl / max(v1.total_trades, 1)
        v2_avg = v2.net_pnl / max(v2.total_trades, 1)
        print(f"{'Avg P&L/Trade':<25} {v1_avg:>+12.2f} {v2_avg:>+12.2f} {v2_avg-v1_avg:>+12.2f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()

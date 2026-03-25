#!/usr/bin/env python3
"""Deep analysis of 5m native features:
1. Ensemble IC (top features combined)
2. Cost-adjusted expected edge
3. Mean reversion vs momentum decomposition
4. Cross-exchange features (Bybit vs Binance implied)
5. Funding rate tick features
6. Walk-forward IC stability test
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data_files")


def load_5m(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_5m.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_volume"]
    return df


def compute_top_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the top 20 genuine 5m features identified in scan."""
    feats = pd.DataFrame(index=df.index)

    # --- MICROSTRUCTURE (genuine 5m) ---
    tbr = df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
    ti = 2 * tbr - 1
    vd = df["taker_buy_volume"] - df["taker_sell_volume"]

    feats["taker_buy_ratio_ma6"] = tbr.rolling(6).mean()
    feats["taker_imbalance_cum6"] = ti.rolling(6).sum()
    feats["taker_imbalance_cum12"] = ti.rolling(12).sum()
    feats["cvd_6"] = vd.rolling(6).sum()
    feats["cvd_12"] = vd.rolling(12).sum()
    feats["cvd_24"] = vd.rolling(24).sum()

    # CVD-price divergence
    for w in [6, 12]:
        cvd_dir = np.sign(vd.rolling(w).sum())
        price_dir = np.sign(df["close"].diff(w))
        feats[f"cvd_price_div_{w}"] = cvd_dir * price_dir

    # Trade intensity
    feats["trade_intensity_ratio_12"] = df["trades"] / df["trades"].rolling(12).mean()

    # Avg trade size
    avg_size = df["quote_volume"] / df["trades"].replace(0, np.nan)
    feats["avg_trade_size_ratio_12"] = avg_size / avg_size.rolling(12).mean()

    # VPIN proxy
    abs_imb = (df["taker_buy_volume"] - df["taker_sell_volume"]).abs()
    feats["vpin_proxy_24"] = abs_imb.rolling(24).sum() / df["volume"].rolling(24).sum()

    # --- MULTIBAR (mean reversion) ---
    # VWAP deviation
    for w in [12, 24, 48]:
        vwap = df["quote_volume"].rolling(w).sum() / df["volume"].rolling(w).sum()
        feats[f"close_vs_vwap_{w}"] = (df["close"] - vwap) / vwap

    # Bollinger position
    for w in [12, 24]:
        ma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        feats[f"bb_pos_{w}"] = (df["close"] - ma) / std.replace(0, np.nan)

    # Momentum
    for w in [3, 6, 12]:
        feats[f"mom_{w}"] = df["close"].pct_change(w)

    # RSI
    delta = df["ret"].copy()
    gain = delta.clip(lower=0).rolling(6).mean()
    loss = (-delta.clip(upper=0)).rolling(6).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_6"] = 100 - 100 / (1 + rs)

    # Distance from rolling high
    feats["dist_high_48"] = df["close"] / df["high"].rolling(48).max() - 1

    # Bar run length
    direction = np.sign(df["ret"])
    runs = direction.copy().values.astype(float)
    for i in range(1, len(runs)):
        if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
            runs[i] = runs[i - 1] + direction.iloc[i]
        else:
            runs[i] = direction.iloc[i]
    feats["bar_run_length"] = runs

    # --- VOL DYNAMICS ---
    feats["rv_48"] = df["log_ret"].rolling(48).std() * np.sqrt(288)
    feats["vol_price_corr_12"] = df["volume"].rolling(12).corr(df["ret"])
    bar_range = (df["high"] - df["low"]) / df["close"]
    feats["bar_range_ratio_12"] = bar_range / bar_range.rolling(12).mean()

    # --- INTRADAY ---
    hour = df["datetime"].dt.hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Intraday return
    midnight_close = df.groupby(df["datetime"].dt.date)["close"].transform("first")
    feats["intraday_ret"] = df["close"] / midnight_close - 1

    return feats


def ensemble_ic_test(df: pd.DataFrame, feats: pd.DataFrame, symbol: str):
    """Test ensemble signal IC vs individual features."""
    print(f"\n{'='*60}")
    print(f"  {symbol} — Ensemble IC Analysis")
    print(f"{'='*60}")

    # Rank-normalize all features
    feats_norm = feats.rank(pct=True) - 0.5  # centered around 0

    # Forward returns
    for h in [3, 6, 12, 24]:
        fwd = df["close"].pct_change(h).shift(-h)
        fwd_name = f"{h*5}m"

        # Individual ICs
        ics = {}
        for col in feats_norm.columns:
            mask = feats_norm[col].notna() & fwd.notna()
            if mask.sum() < 1000:
                continue
            ic, _ = stats.spearmanr(feats_norm[col][mask], fwd[mask])
            ics[col] = ic

        # Equal-weight ensemble
        ensemble = feats_norm.mean(axis=1)
        mask = ensemble.notna() & fwd.notna()
        ens_ic, ens_p = stats.spearmanr(ensemble[mask], fwd[mask])

        # IC-weighted ensemble (use sign from individual ICs)
        weights = pd.Series(ics)
        ic_weighted = (feats_norm * weights).sum(axis=1) / weights.abs().sum()
        mask2 = ic_weighted.notna() & fwd.notna()
        icw_ic, icw_p = stats.spearmanr(ic_weighted[mask2], fwd[mask2])

        # Top 10 only
        top10 = weights.abs().nlargest(10).index
        top10_ens = feats_norm[top10].mean(axis=1)
        mask3 = top10_ens.notna() & fwd.notna()
        t10_ic, t10_p = stats.spearmanr(top10_ens[mask3], fwd[mask3])

        best_individual = max(abs(v) for v in ics.values())
        print(f"\n  h={fwd_name}:")
        print(f"    Best individual |IC|: {best_individual:.5f}")
        print(f"    Equal-weight ensemble IC: {ens_ic:+.5f} (p={ens_p:.1e})")
        print(f"    IC-weighted ensemble IC:  {icw_ic:+.5f} (p={icw_p:.1e})")
        print(f"    Top-10 ensemble IC:       {t10_ic:+.5f} (p={t10_p:.1e})")


def cost_analysis(df: pd.DataFrame, feats: pd.DataFrame, symbol: str):
    """Analyze whether the edge survives transaction costs."""
    print(f"\n{'='*60}")
    print(f"  {symbol} — Cost-Adjusted Edge Analysis")
    print(f"{'='*60}")

    FEE_BPS = 4  # taker fee (bps)
    SLIPPAGE_BPS = 3  # estimated slippage
    TOTAL_COST_BPS = FEE_BPS + SLIPPAGE_BPS  # 7 bps round trip

    # Use IC-weighted ensemble for signal
    for h in [3, 6, 12, 24, 48]:
        fwd = df["close"].pct_change(h).shift(-h)
        hold_time = h * 5  # minutes

        feats_norm = feats.rank(pct=True) - 0.5
        ics = {}
        for col in feats_norm.columns:
            mask = feats_norm[col].notna() & fwd.notna()
            if mask.sum() < 1000:
                continue
            ic, _ = stats.spearmanr(feats_norm[col][mask], fwd[mask])
            ics[col] = ic

        weights = pd.Series(ics)
        signal = (feats_norm * weights).sum(axis=1) / weights.abs().sum()

        # Discretize signal: long/short/flat
        mask = signal.notna() & fwd.notna()
        sig = signal[mask]
        ret = fwd[mask]

        # IC-based expected edge
        ic, _ = stats.spearmanr(sig, ret)
        ret_std = ret.std()
        expected_edge_bps = abs(ic) * ret_std * 10000  # in bps

        # Approximate trade frequency (how often signal flips)
        sig_discrete = np.sign(sig)
        flips = (sig_discrete.diff().abs() > 0).sum()
        total_bars = len(sig_discrete)
        trades_per_day = flips / (total_bars / 288) * 2  # entry + exit
        cost_per_day = trades_per_day * TOTAL_COST_BPS

        # Expected daily edge
        signals_per_day = 288 / h  # how many h-bar periods per day
        gross_edge_per_day = expected_edge_bps * signals_per_day
        net_edge_per_day = gross_edge_per_day - cost_per_day

        print(f"\n  Hold={hold_time}m ({h} bars):")
        print(f"    IC: {ic:+.5f}, Ret σ: {ret_std*100:.3f}%")
        print(f"    Expected edge/trade: {expected_edge_bps:.2f} bps")
        print(f"    Cost/trade: {TOTAL_COST_BPS} bps (fee {FEE_BPS} + slip {SLIPPAGE_BPS})")
        print(f"    Trades/day: {trades_per_day:.1f}")
        print(f"    Gross edge/day: {gross_edge_per_day:.1f} bps")
        print(f"    Net edge/day: {net_edge_per_day:.1f} bps")
        status = "PROFITABLE ✓" if net_edge_per_day > 0 else "UNPROFITABLE ✗"
        print(f"    → {status}")

        # What IC would be needed to break even?
        if net_edge_per_day <= 0:
            needed_ic = TOTAL_COST_BPS / (ret_std * 10000)
            print(f"    Break-even IC needed: {needed_ic:.5f} (current: {abs(ic):.5f})")


def walkforward_ic_stability(df: pd.DataFrame, feats: pd.DataFrame, symbol: str):
    """Test IC stability across time using rolling windows."""
    print(f"\n{'='*60}")
    print(f"  {symbol} — Walk-Forward IC Stability")
    print(f"{'='*60}")

    h = 6  # 30m hold = best horizon
    fwd = df["close"].pct_change(h).shift(-h)

    feats_norm = feats.rank(pct=True) - 0.5
    # IC-weighted ensemble
    ics = {}
    for col in feats_norm.columns:
        mask = feats_norm[col].notna() & fwd.notna()
        if mask.sum() < 1000:
            continue
        ic, _ = stats.spearmanr(feats_norm[col][mask], fwd[mask])
        ics[col] = ic
    weights = pd.Series(ics)
    signal = (feats_norm * weights).sum(axis=1) / weights.abs().sum()

    # Rolling 30-day IC
    window = 288 * 30  # 30 days of 5m bars
    step = 288 * 7  # weekly step

    monthly_ics = []
    for start in range(0, len(df) - window, step):
        end = start + window
        mask = signal.iloc[start:end].notna() & fwd.iloc[start:end].notna()
        if mask.sum() < 500:
            continue
        s = signal.iloc[start:end][mask]
        f = fwd.iloc[start:end][mask]
        ic, p = stats.spearmanr(s, f)
        date = df["datetime"].iloc[start + window // 2]
        monthly_ics.append({"date": date, "ic": ic, "n": int(mask.sum())})

    mdf = pd.DataFrame(monthly_ics)
    pos = (mdf["ic"] < 0).sum()  # negative IC = mean reversion works
    total = len(mdf)

    print(f"  Rolling 30-day IC (h={h*5}m hold):")
    print(f"  Total windows: {total}")
    print(f"  Mean IC: {mdf['ic'].mean():+.5f}")
    print(f"  Median IC: {mdf['ic'].median():+.5f}")
    print(f"  Std IC: {mdf['ic'].std():.5f}")
    print(f"  IC<0 (MR works): {pos}/{total} ({100*pos/total:.0f}%)")
    print(f"  Min IC: {mdf['ic'].min():+.5f}")
    print(f"  Max IC: {mdf['ic'].max():+.5f}")
    print(f"  ICIR: {mdf['ic'].mean()/mdf['ic'].std():.3f}")

    # Show by quarter
    mdf["quarter"] = mdf["date"].dt.to_period("Q")
    qsum = mdf.groupby("quarter")["ic"].agg(["mean", "std", "count"])
    print("\n  Quarterly IC:")
    for q, row in qsum.iterrows():
        bar = "█" * int(abs(row["mean"]) * 500)
        sign = "+" if row["mean"] > 0 else "-"
        print(f"    {q}: IC={row['mean']:+.5f} ± {row['std']:.4f} (n={int(row['count'])}) {sign}{bar}")


def mean_reversion_strategy_sim(df: pd.DataFrame, feats: pd.DataFrame, symbol: str):
    """Simple mean reversion strategy simulation with costs."""
    print(f"\n{'='*60}")
    print(f"  {symbol} — Mean Reversion Strategy Simulation")
    print(f"{'='*60}")

    FEE = 0.0004  # 4 bps taker
    SLIP = 0.0003  # 3 bps slippage
    COST = FEE + SLIP  # 7 bps

    # Use bb_pos_12 + cvd_6 as signal (top MR features)
    bb = feats["bb_pos_12"]
    cvd = feats["cvd_6"]

    # Rank and combine
    bb_rank = bb.rank(pct=True) - 0.5
    cvd_rank = cvd.rank(pct=True) - 0.5
    combo = (bb_rank + cvd_rank) / 2

    # Entry threshold
    for threshold in [0.2, 0.3, 0.4]:
        pos = pd.Series(0.0, index=df.index)
        # Enter short when combo > threshold (overbought)
        pos[combo > threshold] = -1.0
        # Enter long when combo < -threshold (oversold)
        pos[combo < -threshold] = 1.0

        # Minimum hold period: 6 bars (30m)
        min_hold = 6
        actual_pos = pos.copy()
        hold_count = 0
        current = 0.0
        for i in range(len(actual_pos)):
            if actual_pos.iloc[i] != 0 and actual_pos.iloc[i] != current:
                current = actual_pos.iloc[i]
                hold_count = 0
            elif hold_count < min_hold and current != 0:
                actual_pos.iloc[i] = current
                hold_count += 1
            else:
                current = actual_pos.iloc[i]
                hold_count = 0

        # P&L
        ret = df["close"].pct_change()
        raw_pnl = actual_pos.shift(1) * ret
        trades = actual_pos.diff().abs()
        cost_series = trades * COST
        net_pnl = raw_pnl - cost_series

        # Stats
        total_ret = net_pnl.sum()
        n_trades = (trades > 0).sum()
        n_days = len(df) / 288
        sharpe = net_pnl.mean() / net_pnl.std() * np.sqrt(288 * 365) if net_pnl.std() > 0 else 0
        gross_sharpe = raw_pnl.mean() / raw_pnl.std() * np.sqrt(288 * 365) if raw_pnl.std() > 0 else 0
        win_rate = (net_pnl[net_pnl != 0] > 0).mean() * 100
        max_dd = (net_pnl.cumsum() - net_pnl.cumsum().cummax()).min() * 100

        print(f"\n  Threshold={threshold}:")
        print(f"    Trades: {n_trades} ({n_trades/n_days:.1f}/day)")
        print(f"    Gross Sharpe: {gross_sharpe:.2f}")
        print(f"    Net Sharpe (after 7bps cost): {sharpe:.2f}")
        print(f"    Total return: {total_ret*100:.2f}%")
        print(f"    Win rate: {win_rate:.1f}%")
        print(f"    Max drawdown: {max_dd:.2f}%")
        print(f"    Avg daily P&L: {net_pnl.sum()/n_days*10000:.1f} bps")

    # What about maker fees? (0 bps or rebate)
    print("\n  --- With MAKER fees (0 bps fee + 1 bps slip) ---")
    MAKER_COST = 0.0001  # 1 bps total
    threshold = 0.3
    pos = pd.Series(0.0, index=df.index)
    pos[combo > threshold] = -1.0
    pos[combo < -threshold] = 1.0
    ret = df["close"].pct_change()
    raw_pnl = pos.shift(1) * ret
    trades = pos.diff().abs()
    cost_series = trades * MAKER_COST
    net_pnl = raw_pnl - cost_series
    sharpe = net_pnl.mean() / net_pnl.std() * np.sqrt(288 * 365) if net_pnl.std() > 0 else 0
    n_trades = (trades > 0).sum()
    n_days = len(df) / 288
    print(f"    Trades: {n_trades} ({n_trades/n_days:.1f}/day)")
    print(f"    Net Sharpe (maker): {sharpe:.2f}")
    print(f"    Total return: {net_pnl.sum()*100:.2f}%")
    print(f"    Avg daily P&L: {net_pnl.sum()/n_days*10000:.1f} bps")


def main():
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = load_5m(symbol)
        print(f"\nLoaded {symbol}: {len(df)} bars")

        feats = compute_top_features(df)
        ensemble_ic_test(df, feats, symbol)
        cost_analysis(df, feats, symbol)
        walkforward_ic_stability(df, feats, symbol)
        mean_reversion_strategy_sim(df, feats, symbol)


if __name__ == "__main__":
    main()

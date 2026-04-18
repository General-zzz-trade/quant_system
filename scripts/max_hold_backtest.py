"""Test dynamic max_hold strategies on BTC/ETH 1h.

Compares 5 max_hold policies on the SAME walk-forward predictions:

  A. STATIC          max_hold = 120  (current production baseline)
  B. VOL_INVERSE     max_hold * clip(1/sqrt(vf), 0.5, 1.5) — high vol → cut faster
  C. PROFIT_TRAIL    extend if profit > 1*ATR; shrink if loss > 0.5*ATR
  D. ATR_INVERSE     max_hold * clip(0.01/atr_pct, 0.5, 2.0) — low vol → let run
  E. Z_STRENGTH      max_hold * clip(|z_entry|/dz, 0.7, 1.8) — strong signal → longer
  F. COMBINED        VOL × Z_STRENGTH × profit-trail (best of all worlds?)

Uses simple features + LGBM walk-forward (we want to test the
max_hold policy in isolation, not prediction quality). Same
model + same preds across all 6 variants — only max_hold differs.

Run:
  python3 -m scripts.max_hold_backtest                # both BTC + ETH
  python3 -m scripts.max_hold_backtest --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

# Production thresholds (match BTCUSDT_gate_v2 / ETHUSDT_gate_v2)
DZ = {"BTCUSDT": 1.5, "ETHUSDT": 2.0}
MIN_HOLD = {"BTCUSDT": 21, "ETHUSDT": 9}
MAX_HOLD_BASE = 120
NOTIONAL = 500.0
COST_BPS = 14
BARS_PER_DAY = 24
TRAIN_DAYS = 90      # 90 * 24 = 2160 bars
TEST_DAYS = 30       # 30 * 24 = 720 bars


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    z_entry: float
    bars_held: int
    net_pnl: float
    exit_reason: str


def _build_features(closes, highs, lows, vols):
    """Simple feature set: returns + RSI + ATR + vol-z."""
    n = len(closes)
    f = np.zeros((n, 12))
    s = pd.Series(closes)
    h = pd.Series(highs)
    l_ = pd.Series(lows)
    v = pd.Series(vols)

    # 0-3: returns
    for i, h_ in enumerate([1, 4, 12, 24]):
        f[:, i] = s.pct_change(h_).fillna(0).values

    # 4-5: RSI 14, 28
    for i, w in enumerate([14, 28]):
        delta = s.diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / (loss + 1e-9)
        f[:, 4 + i] = (100 - 100 / (1 + rs)).fillna(50).values

    # 6: ATR 14 / close (vol)
    tr = pd.concat([h - l_, (h - s.shift()).abs(), (l_ - s.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f[:, 6] = (atr / s).fillna(0).values

    # 7: realized vol 24
    f[:, 7] = s.pct_change().rolling(24).std().fillna(0).values

    # 8: log-vol z-score
    log_vol = np.log(v + 1)
    f[:, 8] = ((log_vol - log_vol.rolling(20).mean()) / (log_vol.rolling(20).std() + 1e-9)).fillna(0).values

    # 9: close vs MA
    f[:, 9] = (closes / s.rolling(20).mean() - 1).fillna(0).values

    # 10: high-low range
    f[:, 10] = ((h - l_) / s).rolling(7).mean().fillna(0).values

    # 11: return autocorr lag-1 over 30 bars
    f[:, 11] = s.pct_change().rolling(30).apply(
        lambda x: x.autocorr(lag=1) if x.std() > 0 else 0
    ).fillna(0).values

    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def _walk_forward_predict(X, closes, train_bars, test_bars):
    """Simple LGBM walk-forward with 1h forward return target."""
    import lightgbm as lgb
    n = len(X)
    preds = np.full(n, np.nan)
    target = np.diff(np.log(closes), prepend=np.nan)
    target = np.roll(target, -1)
    target[-1] = np.nan

    start = train_bars
    while start + test_bars <= n:
        tr_start = max(0, start - train_bars)
        X_tr = X[tr_start:start]
        y_tr = target[tr_start:start]
        valid = ~np.isnan(y_tr)
        X_tr = X_tr[valid]
        y_tr = y_tr[valid]
        if len(X_tr) < 100:
            start += test_bars
            continue
        model = lgb.train(
            {"objective": "regression", "metric": "rmse",
             "num_leaves": 31, "learning_rate": 0.05,
             "feature_fraction": 0.85, "verbosity": -1,
             "min_data_in_leaf": 50},
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=100,
        )
        end = min(start + test_bars, n)
        preds[start:end] = model.predict(X[start:end])
        start += test_bars
    return preds


def _atr(highs, lows, closes, window=14):
    h = pd.Series(highs)
    l_ = pd.Series(lows)
    c = pd.Series(closes)
    tr = pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
    return (tr.rolling(window).mean() / c).fillna(0).values


def _vol_factor(closes, window=20, baseline_window=480):
    """vol_factor = recent vol / baseline vol (production-style)."""
    rets = pd.Series(closes).pct_change()
    recent = rets.rolling(window).std()
    baseline = rets.rolling(baseline_window).std()
    vf = (recent / (baseline + 1e-9)).fillna(1.0).clip(0.3, 3.0).values
    return vf


def _max_hold_for_strategy(strategy, *, base, vf, atr_pct, z_entry, dz,
                           bars_held, profit_pct, atr_at_entry):
    """Return effective max_hold given current state."""
    if strategy == "STATIC":
        return base
    if strategy == "VOL_INVERSE":
        scale = np.clip(1.0 / np.sqrt(vf), 0.5, 1.5)
        return int(base * scale)
    if strategy == "ATR_INVERSE":
        scale = np.clip(0.01 / max(atr_pct, 1e-4), 0.5, 2.0)
        return int(base * scale)
    if strategy == "Z_STRENGTH":
        scale = np.clip(abs(z_entry) / max(dz, 1e-4), 0.7, 1.8)
        return int(base * scale)
    if strategy == "PROFIT_TRAIL":
        eff = base
        if profit_pct > atr_at_entry:
            eff = int(base * 1.5)  # let winners run
        elif profit_pct < -atr_at_entry * 0.5:
            eff = int(base * 0.7)  # cut losers
        return eff
    if strategy == "COMBINED":
        s_vol = np.clip(1.0 / np.sqrt(vf), 0.5, 1.5)
        s_z = np.clip(abs(z_entry) / max(dz, 1e-4), 0.7, 1.8)
        s_profit = 1.5 if profit_pct > atr_at_entry else (
            0.7 if profit_pct < -atr_at_entry * 0.5 else 1.0
        )
        return int(base * s_vol * s_z * s_profit)
    return base


def _backtest(strategy, preds, closes, vf, atr_pct, dz, min_hold, base_max):
    n = len(preds)
    pred_std = float(np.nanstd(preds))
    if pred_std < 1e-10:
        return []
    z = preds / pred_std
    cost = COST_BPS / 10000

    trades: list[Trade] = []
    pos = 0
    eb = 0
    ep = 0.0
    z_entry = 0.0
    atr_entry = 0.01
    for i in range(n):
        if pos != 0:
            held = i - eb
            profit_pct = pos * (closes[i] - ep) / ep
            mh = _max_hold_for_strategy(
                strategy,
                base=base_max, vf=vf[i] if i < len(vf) else 1.0,
                atr_pct=atr_pct[i] if i < len(atr_pct) else 0.01,
                z_entry=z_entry, dz=dz, bars_held=held,
                profit_pct=profit_pct, atr_at_entry=atr_entry,
            )
            exit_now = False
            reason = ""
            if held >= mh:
                exit_now = True; reason = "max_hold"
            elif held >= min_hold:
                if pos * z[i] < -0.3:
                    exit_now = True; reason = "z_reversal"
                elif abs(z[i]) < 0.2:
                    exit_now = True; reason = "z_decay"
            if exit_now:
                pnl_pct = pos * (closes[i] - ep) / ep
                gross = pnl_pct * NOTIONAL
                net = gross - cost * NOTIONAL
                trades.append(Trade(eb, i, pos, ep, closes[i], z_entry,
                                    held, net, reason))
                pos = 0

        if pos == 0 and not np.isnan(z[i]):
            if z[i] > dz:
                pos = 1; ep = closes[i]; eb = i
                z_entry = z[i]
                atr_entry = atr_pct[i] if i < len(atr_pct) else 0.01
            elif z[i] < -dz:
                pos = -1; ep = closes[i]; eb = i
                z_entry = z[i]
                atr_entry = atr_pct[i] if i < len(atr_pct) else 0.01

    return trades


def _summarize(trades, label):
    if not trades:
        print(f"  [{label:14s}] no trades"); return None
    nets = np.array([t.net_pnl for t in trades])
    holds = np.array([t.bars_held for t in trades])
    wins = sum(1 for t in trades if t.net_pnl > 0)
    wr = wins / len(trades) * 100
    avg_bp = float(np.mean(nets) / NOTIONAL * 10000)
    total = float(np.sum(nets) / 10000 * 100)

    eq, peak, mdd = 10000.0, 10000.0, 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)

    avg_hold_h = np.mean(holds)
    tpy = 365 * 24 / max(avg_hold_h, 1.0)
    sharpe = float(
        np.mean(nets) / max(np.std(nets, ddof=1), 1e-10) * np.sqrt(tpy)
    ) if len(nets) > 1 else 0.0

    # Reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    print(f"  [{label:14s}] n={len(trades):>3d}  WR={wr:>4.0f}%  "
          f"hold={avg_hold_h:>4.0f}h  avg={avg_bp:>+5.1f}bp  "
          f"total={total:>+5.2f}%  DD={mdd*100:>4.2f}%  "
          f"Sh={sharpe:>+5.2f}  | {reasons}")
    return {"n": len(trades), "wr": wr, "hold_h": avg_hold_h,
            "avg_bp": avg_bp, "total": total, "mdd": mdd*100,
            "sharpe": sharpe}


def run_symbol(symbol):
    df = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    print(f"\n{symbol} 1h backtest")
    print(f"  Bars: {len(df):,}  range: "
          f"{pd.Timestamp(int(df['open_time'].iloc[0]), unit='ms').date()} → "
          f"{pd.Timestamp(int(df['open_time'].iloc[-1]), unit='ms').date()}")

    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    vols = df["volume"].values.astype(np.float64)

    # Use last 18 months for OOS test
    n_total = len(closes)
    oos_bars = min(BARS_PER_DAY * 540, n_total - BARS_PER_DAY * TRAIN_DAYS)
    start_idx = max(0, n_total - oos_bars - BARS_PER_DAY * TRAIN_DAYS)
    closes = closes[start_idx:]
    highs = highs[start_idx:]
    lows = lows[start_idx:]
    vols = vols[start_idx:]

    print(f"  Trim to {len(closes):,} bars for backtest")

    X = _build_features(closes, highs, lows, vols)
    preds = _walk_forward_predict(X, closes,
                                  train_bars=BARS_PER_DAY * TRAIN_DAYS,
                                  test_bars=BARS_PER_DAY * TEST_DAYS)

    valid = ~np.isnan(preds)
    print(f"  Walk-forward: {valid.sum():,} valid preds")

    closes_oos = closes[valid]
    preds_oos = preds[valid]
    highs_oos = highs[valid]
    lows_oos = lows[valid]
    vf = _vol_factor(closes_oos)
    atr_pct = _atr(highs_oos, lows_oos, closes_oos)

    dz = DZ[symbol]
    mh = MIN_HOLD[symbol]
    print()
    results = {}
    for strategy in ("STATIC", "VOL_INVERSE", "ATR_INVERSE",
                     "Z_STRENGTH", "PROFIT_TRAIL", "COMBINED"):
        trades = _backtest(strategy, preds_oos, closes_oos,
                           vf, atr_pct, dz, mh, MAX_HOLD_BASE)
        r = _summarize(trades, strategy)
        if r:
            results[strategy] = r
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT", "BOTH"], default="BOTH")
    args = p.parse_args()

    print("=" * 90)
    print("Dynamic max_hold backtest — 1h, last ~18 months OOS")
    print(f"  Base max_hold = {MAX_HOLD_BASE}h, dz=BTC {DZ['BTCUSDT']}/ETH {DZ['ETHUSDT']}")
    print(f"  Walk-forward: train={TRAIN_DAYS}d test={TEST_DAYS}d, refit every {TEST_DAYS}d")
    print("=" * 90)

    syms = ["BTCUSDT", "ETHUSDT"] if args.symbol == "BOTH" else [args.symbol]
    all_results = {}
    for s in syms:
        all_results[s] = run_symbol(s)

    # Comparative ranking
    if all_results:
        print("\n" + "=" * 90)
        print("RANKING by Sharpe (per symbol)")
        print("=" * 90)
        for sym, results in all_results.items():
            print(f"\n{sym}:")
            ranked = sorted(results.items(), key=lambda kv: -kv[1]["sharpe"])
            for s, r in ranked:
                marker = "🏆" if s == ranked[0][0] else "  "
                print(f"  {marker} {s:14s} Sharpe={r['sharpe']:+.2f} "
                      f"return={r['total']:+.2f}% DD={r['mdd']:.2f}% "
                      f"trades={r['n']}")


if __name__ == "__main__":
    main()

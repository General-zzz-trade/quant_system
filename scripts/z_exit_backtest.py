"""Test asymmetric z_exit thresholds (let winners run, cut losers fast).

Current production exits when:
  pos * z[i] < -0.3   (z_reversal — model flipped opposite)
  abs(z[i]) < 0.2     (z_decay — signal weakened)

These thresholds apply uniformly regardless of trade PnL. Hypothesis
from the max_hold test: the same trades that benefit from extended
max_hold (PROFIT_TRAIL was the only positive-Sharpe variant) should
also benefit from looser z_decay when winning — give trends time
to develop. And tighter z_reversal when losing — cut faster.

Variants tested:
  A. STATIC          decay=0.2 reversal=-0.3  (current production)
  B. WIN_RELAX_DECAY decay=0.10 if profit > 0.5*atr else 0.2
  C. LOSE_TIGHT_REV  reversal=-0.15 if loss > 0.5*atr else -0.3
  D. ATR_TRAIL       same as PROFIT_TRAIL but on z thresholds, not max_hold
                     decay=0.10 if winning, reversal=-0.15 if losing
  E. BOTH            B + C combined

Same walk-forward LGBM as max_hold_backtest.py for apples-to-apples.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from scripts.max_hold_backtest import (  # noqa: E402
    BARS_PER_DAY, COST_BPS, DZ, MIN_HOLD, MAX_HOLD_BASE, NOTIONAL,
    TRAIN_DAYS, TEST_DAYS, _atr, _build_features, _vol_factor,
    _walk_forward_predict,
)


@dataclass
class Trade:
    entry_bar: int; exit_bar: int; direction: int
    entry_price: float; exit_price: float
    z_entry: float; bars_held: int; net_pnl: float
    exit_reason: str


def _z_exit_thresholds(strategy, *, profit_pct, atr_at_entry):
    """Return (decay_threshold, reversal_threshold) for current trade state."""
    if strategy == "STATIC":
        return 0.20, -0.30
    if strategy == "WIN_RELAX_DECAY":
        decay = 0.10 if profit_pct > 0.5 * atr_at_entry else 0.20
        return decay, -0.30
    if strategy == "LOSE_TIGHT_REV":
        reversal = -0.15 if profit_pct < -0.5 * atr_at_entry else -0.30
        return 0.20, reversal
    if strategy == "ATR_TRAIL":
        # Asymmetric: winners get loose decay, losers get tight reversal
        decay = 0.10 if profit_pct > 0.5 * atr_at_entry else 0.20
        reversal = -0.15 if profit_pct < -0.5 * atr_at_entry else -0.30
        return decay, reversal
    if strategy == "BOTH":
        decay = 0.10 if profit_pct > 0.5 * atr_at_entry else 0.20
        reversal = -0.15 if profit_pct < -0.5 * atr_at_entry else -0.30
        return decay, reversal
    return 0.20, -0.30


def _backtest(strategy, preds, closes, atr_pct, dz, min_hold, max_hold):
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
            decay_th, reversal_th = _z_exit_thresholds(
                strategy,
                profit_pct=profit_pct, atr_at_entry=atr_entry,
            )
            exit_now = False
            reason = ""
            if held >= max_hold:
                exit_now = True; reason = "max_hold"
            elif held >= min_hold:
                if pos * z[i] < reversal_th:
                    exit_now = True; reason = "z_reversal"
                elif abs(z[i]) < decay_th:
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
        print(f"  [{label:18s}] no trades"); return None
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

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    print(f"  [{label:18s}] n={len(trades):>3d}  WR={wr:>4.0f}%  "
          f"hold={avg_hold_h:>4.0f}h  avg={avg_bp:>+6.1f}bp  "
          f"total={total:>+5.2f}%  DD={mdd*100:>4.2f}%  "
          f"Sh={sharpe:>+5.2f}  | {reasons}")
    return {"n": len(trades), "wr": wr, "hold_h": avg_hold_h,
            "avg_bp": avg_bp, "total": total, "mdd": mdd*100,
            "sharpe": sharpe}


def run_symbol(symbol):
    df = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    vols = df["volume"].values.astype(np.float64)

    n_total = len(closes)
    oos_bars = min(BARS_PER_DAY * 540, n_total - BARS_PER_DAY * TRAIN_DAYS)
    start_idx = max(0, n_total - oos_bars - BARS_PER_DAY * TRAIN_DAYS)
    closes = closes[start_idx:]; highs = highs[start_idx:]
    lows = lows[start_idx:]; vols = vols[start_idx:]

    print(f"\n{symbol} 1h backtest (z_exit policy sweep)")
    print(f"  Bars: {len(closes):,}")

    X = _build_features(closes, highs, lows, vols)
    preds = _walk_forward_predict(X, closes,
                                  train_bars=BARS_PER_DAY * TRAIN_DAYS,
                                  test_bars=BARS_PER_DAY * TEST_DAYS)

    valid = ~np.isnan(preds)
    closes_oos = closes[valid]; preds_oos = preds[valid]
    highs_oos = highs[valid]; lows_oos = lows[valid]
    atr_pct = _atr(highs_oos, lows_oos, closes_oos)

    print(f"  Valid preds: {valid.sum():,}\n")
    results = {}
    for strategy in ("STATIC", "WIN_RELAX_DECAY", "LOSE_TIGHT_REV",
                     "ATR_TRAIL", "BOTH"):
        trades = _backtest(strategy, preds_oos, closes_oos, atr_pct,
                           DZ[symbol], MIN_HOLD[symbol], MAX_HOLD_BASE)
        r = _summarize(trades, strategy)
        if r:
            results[strategy] = r
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT", "BOTH"], default="BOTH")
    args = p.parse_args()

    print("=" * 90)
    print("Asymmetric z_exit threshold backtest — 1h, ~18 month OOS")
    print("=" * 90)

    syms = ["BTCUSDT", "ETHUSDT"] if args.symbol == "BOTH" else [args.symbol]
    all_results = {}
    for s in syms:
        all_results[s] = run_symbol(s)

    print("\n" + "=" * 90)
    print("RANKING by Sharpe")
    print("=" * 90)
    for sym, results in all_results.items():
        print(f"\n{sym}:")
        ranked = sorted(results.items(), key=lambda kv: -kv[1]["sharpe"])
        baseline = results.get("STATIC", {}).get("sharpe", 0)
        for s, r in ranked:
            marker = "🏆" if s == ranked[0][0] else "  "
            improvement = r["sharpe"] - baseline
            print(f"  {marker} {s:18s} Sharpe={r['sharpe']:+.2f} "
                  f"(Δ={improvement:+.2f})  return={r['total']:+.2f}% "
                  f"DD={r['mdd']:.2f}% trades={r['n']}")


if __name__ == "__main__":
    main()

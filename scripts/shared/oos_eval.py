"""Shared OOS evaluation utilities for backtesting credibility.

Separates prediction quality (IC/direction accuracy on 5-bar target)
from trading profitability (1-bar PnL with costs and threshold).

Cost model matches live BacktestExecutionAdapter:
  - fee_bps: flat fee per leg (default 4 bps = Binance taker)
  - slippage_bps: price impact per leg (default 2 bps)
  - Total per leg: (fee_bps + slippage_bps) / 10000

Sharpe is computed on active bars only (signal != 0), ddof=1.
Win rate is trade-level (consecutive same-signal bars grouped).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def compute_1bar_returns(
    closes_all: np.ndarray,
    test_orig_idx: np.ndarray,
) -> np.ndarray:
    """Compute non-overlapping 1-bar forward returns for test set rows.

    ret_1bar[i] = closes_all[idx+1] / closes_all[idx] - 1
    NaN if idx+1 is out of bounds.
    """
    n = len(test_orig_idx)
    ret = np.full(n, np.nan)
    max_idx = len(closes_all) - 1
    for i in range(n):
        idx = test_orig_idx[i]
        if idx + 1 <= max_idx:
            ret[i] = closes_all[idx + 1] / closes_all[idx] - 1.0
    return ret


def apply_threshold(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous predictions to discrete signal {-1, 0, +1}.

    |pred| <= threshold → 0 (flat), matching live MLDecisionModule.
    """
    signal = np.zeros_like(y_pred)
    signal[y_pred > threshold] = 1.0
    signal[y_pred < -threshold] = -1.0
    return signal


def compute_signal_costs(
    signal: np.ndarray,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> np.ndarray:
    """Compute per-bar transaction costs based on signal changes.

    Matches live BacktestExecutionAdapter cost structure:
      fee_bps:      taker fee per leg (default 4 bps)
      slippage_bps: price impact per leg (default 2 bps)

    Cost per leg = (fee_bps + slippage_bps) / 10000.
    Reversal = 2 legs, open/close = 1 leg, hold = 0.
    """
    leg_cost = (fee_bps + slippage_bps) / 10_000
    costs = np.zeros(len(signal))
    prev = 0.0
    for i in range(len(signal)):
        cur = signal[i]
        if cur != prev:
            if prev != 0.0 and cur != 0.0:
                costs[i] = 2.0 * leg_cost
            else:
                costs[i] = leg_cost
        prev = cur
    return costs


def _group_trades(signal: np.ndarray, net_pnl: np.ndarray) -> List[float]:
    """Group consecutive same-direction bars into trades, return per-trade net PnL."""
    trades: List[float] = []
    trade_pnl = 0.0
    prev_sig = 0.0
    for i in range(len(signal)):
        cur = signal[i]
        if cur != 0:
            if prev_sig != 0 and cur != prev_sig:
                # Reversal — close previous trade (up to previous bar)
                trades.append(trade_pnl)
                trade_pnl = 0.0
            trade_pnl += net_pnl[i]
        else:
            if prev_sig != 0:
                # Went flat — close trade
                trades.append(trade_pnl)
                trade_pnl = 0.0
        prev_sig = cur
    if prev_sig != 0:
        trades.append(trade_pnl)
    return trades


def evaluate_oos(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    ret_1bar: np.ndarray,
    thresholds: Sequence[float] = (0.0, 0.0005, 0.001, 0.002, 0.005),
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> Dict:
    """Full OOS evaluation with threshold scan.

    Prediction quality (IC, direction accuracy, MSE) uses 5-bar y_test.
    Trading metrics (PnL, Sharpe, win rate, costs) use 1-bar returns.

    Sharpe: computed on active bars only (signal != 0), ddof=1.
    Win rate: trade-level (consecutive same-signal bars grouped).
    Cost model: fee_bps + slippage_bps per leg, matching live adapter.
    """
    # --- Prediction quality (uses 5-bar target) ---
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
    ic = float(np.corrcoef(y_pred, y_test)[0, 1]) if len(y_pred) > 1 else 0.0
    mse = float(np.mean((y_pred - y_test) ** 2))

    prediction_quality = {
        "direction_accuracy": dir_acc,
        "ic": ic,
        "mse": mse,
    }

    # --- Threshold scan (uses 1-bar returns) ---
    valid = ~np.isnan(ret_1bar)
    ret_valid = ret_1bar[valid]
    pred_valid = y_pred[valid]

    scan_results: List[Dict] = []
    best_sharpe = -np.inf
    best_thr = thresholds[0]

    for thr in thresholds:
        signal = apply_threshold(pred_valid, thr)
        costs = compute_signal_costs(signal, fee_bps, slippage_bps)

        gross_pnl = signal * ret_valid
        net_pnl = gross_pnl - costs

        n_trades = int(np.sum(np.diff(np.concatenate([[0], signal])) != 0))
        exposure = float(np.mean(signal != 0))

        gross_ret = float(np.sum(gross_pnl))
        net_ret = float(np.sum(net_pnl))
        total_costs = float(np.sum(costs))

        # Sharpe: active bars only, ddof=1
        active_mask = signal != 0
        n_active = int(np.sum(active_mask))
        if n_active > 1:
            active_net = net_pnl[active_mask]
            std_active = float(np.std(active_net, ddof=1))
            if std_active > 0:
                sharpe_raw = float(np.mean(active_net)) / std_active
                sharpe_annual = sharpe_raw * np.sqrt(8760)
            else:
                sharpe_annual = 0.0
        else:
            sharpe_annual = 0.0

        # Win rate: trade-level
        trades = _group_trades(signal, net_pnl)
        if trades:
            wins = sum(1 for t in trades if t > 0)
            win_rate = float(wins / len(trades))
        else:
            win_rate = 0.0

        row = {
            "threshold": float(thr),
            "gross_return": gross_ret,
            "net_return": net_ret,
            "total_costs": total_costs,
            "sharpe_annual": sharpe_annual,
            "win_rate": win_rate,
            "n_trades": n_trades,
            "exposure": exposure,
        }
        scan_results.append(row)

        if sharpe_annual > best_sharpe:
            best_sharpe = sharpe_annual
            best_thr = float(thr)

    return {
        "prediction_quality": prediction_quality,
        "threshold_scan": scan_results,
        "best_threshold": best_thr,
    }


def print_evaluation(eval_result: Dict, label: str = "OOS") -> None:
    """Pretty-print evaluate_oos results."""
    pq = eval_result["prediction_quality"]
    print(f"\n  {label} Prediction Quality (5-bar target):")
    print(f"    Direction accuracy: {pq['direction_accuracy']:.4f} ({pq['direction_accuracy']*100:.1f}%)")
    print(f"    IC:                {pq['ic']:.4f}")
    print(f"    MSE:               {pq['mse']:.6f}")

    print(f"\n  {label} Trading Performance (1-bar, fee=4bps+slip=2bps per leg):")
    print(f"    {'Threshold':>10} {'GrossRet':>10} {'NetRet':>10} {'Costs':>10} {'Sharpe':>8} {'WinRate':>8} {'Trades':>7} {'Exposure':>8}")  # noqa: E501
    print(f"    {'-'*71}")
    for row in eval_result["threshold_scan"]:
        print(f"    {row['threshold']:>10.4f} "
              f"{row['gross_return']*100:>9.2f}% "
              f"{row['net_return']*100:>9.2f}% "
              f"{row['total_costs']*100:>9.2f}% "
              f"{row['sharpe_annual']:>8.2f} "
              f"{row['win_rate']*100:>7.1f}% "
              f"{row['n_trades']:>7d} "
              f"{row['exposure']*100:>7.1f}%")
    print(f"\n    Best threshold: {eval_result['best_threshold']:.4f}")

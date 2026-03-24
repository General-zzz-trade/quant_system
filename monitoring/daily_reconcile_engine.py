"""Reconciliation engine for daily_reconcile.

Extracted from daily_reconcile.py to keep file sizes manageable.
Contains the core reconcile_symbol() logic.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reconcile_symbol(session) -> dict:
    """Run full reconciliation for one symbol.

    Returns a structured report dict with all metrics.
    """
    from monitoring.daily_reconcile import (
        replay_signals, _extract_bt_trades, _find_closest_bar, _parse_ts,
    )

    bars = session.bars
    fills = session.fills

    if not bars:
        return {
            "symbol": session.symbol,
            "status": "no_data",
            "error": "No bar entries found in log",
        }

    # Effective deadzone
    dz_values = [b.dz for b in bars if b.dz > 0]
    effective_dz = float(np.mean(dz_values)) if dz_values else session.deadzone
    if effective_dz <= 0:
        effective_dz = 0.3

    bt_results = replay_signals(
        bars, deadzone=effective_dz,
        min_hold=session.min_hold, max_hold=session.max_hold,
    )

    # 1. Signal agreement
    n_bars = len(bars)
    agreements = 0
    disagreements = []
    live_signals = []
    bt_signals = []

    for bar, bt in zip(bars, bt_results):
        live_sig = bar.signal
        bt_sig = bt["signal"]
        live_signals.append(live_sig)
        bt_signals.append(bt_sig)

        if live_sig == bt_sig:
            agreements += 1
        else:
            disagreements.append({
                "bar_num": bar.bar_num,
                "timestamp": str(bar.timestamp),
                "close": bar.close,
                "z": bar.z,
                "live_signal": live_sig,
                "bt_signal": bt_sig,
                "live_hold": bar.hold,
                "regime": bar.regime,
                "dz": bar.dz,
            })

    signal_match_rate = agreements / n_bars if n_bars > 0 else 0.0

    live_dist = {-1: 0, 0: 0, 1: 0}
    bt_dist = {-1: 0, 0: 0, 1: 0}
    for s in live_signals:
        live_dist[s] = live_dist.get(s, 0) + 1
    for s in bt_signals:
        bt_dist[s] = bt_dist.get(s, 0) + 1

    live_changes = sum(1 for i in range(1, len(live_signals))
                       if live_signals[i] != live_signals[i - 1])
    bt_changes = sum(1 for i in range(1, len(bt_signals))
                     if bt_signals[i] != bt_signals[i - 1])

    # 2. Execution slippage
    open_fills = [f for f in fills if f.fill_type == "open" and f.price > 0]
    slippage_bps_list = []
    for fill in open_fills:
        closest_bar = _find_closest_bar(bars, fill.timestamp)
        if closest_bar and closest_bar.close > 0:
            slip_bps = (fill.price - closest_bar.close) / closest_bar.close * 10000
            if fill.side in ("short", "sell"):
                slip_bps = -slip_bps
            slippage_bps_list.append(slip_bps)

    avg_fill_slippage_bps = (
        float(np.mean(slippage_bps_list)) if slippage_bps_list else 0.0
    )

    # 3. Timing delay
    timing_delays = []
    for i in range(1, min(len(live_signals), len(bt_signals))):
        bt_changed = bt_signals[i] != bt_signals[i - 1]
        live_changed = live_signals[i] != live_signals[i - 1]
        if bt_changed and not live_changed:
            for j in range(i + 1, min(i + 10, len(live_signals))):
                if live_signals[j] == bt_signals[i]:
                    timing_delays.append(j - i)
                    break

    timing_delay_bars = float(np.mean(timing_delays)) if timing_delays else 0.0

    # 4. PnL comparison
    close_fills = [f for f in fills if f.fill_type == "close" and f.pnl is not None]
    live_total_pnl = sum(f.pnl for f in close_fills)

    bt_trades = _extract_bt_trades(bt_results)
    bt_total_pnl_pct = sum(t["pnl_pct"] for t in bt_trades)

    pnl_gap_pct = None
    live_pnl_pct_list = [f.pnl_pct for f in close_fills if f.pnl_pct is not None]
    live_total_pnl_pct = sum(live_pnl_pct_list)
    if abs(bt_total_pnl_pct) > 0.001:
        pnl_gap_pct = round(
            (live_total_pnl_pct - bt_total_pnl_pct) / abs(bt_total_pnl_pct) * 100, 2
        )

    # 5. Disagreement analysis
    divergence_causes = {
        "regime_filter": 0, "hold_timing": 0,
        "z_reversal_exit": 0, "stop_loss": 0, "unknown": 0,
    }
    for d in disagreements:
        if d["regime"] != "active":
            divergence_causes["regime_filter"] += 1
        elif d["live_signal"] == 0 and d["bt_signal"] != 0:
            stop_near = any(
                f.exit_type == "stop"
                and abs((f.timestamp - _parse_ts(d["timestamp"])).total_seconds()) < 7200
                for f in fills if f.fill_type == "close"
            )
            if stop_near:
                divergence_causes["stop_loss"] += 1
            else:
                divergence_causes["z_reversal_exit"] += 1
        elif abs(d.get("live_hold", 0)) > 0:
            divergence_causes["hold_timing"] += 1
        else:
            divergence_causes["unknown"] += 1

    return {
        "symbol": session.symbol,
        "status": "ok",
        "model": session.model,
        "config": {
            "deadzone": session.deadzone,
            "effective_dz": round(effective_dz, 4),
            "min_hold": session.min_hold,
            "max_hold": session.max_hold,
        },
        "period": {
            "start": str(bars[0].timestamp),
            "end": str(bars[-1].timestamp),
            "n_bars": n_bars,
            "n_days": round((bars[-1].timestamp - bars[0].timestamp).total_seconds() / 86400, 1),
        },
        "metrics": {
            "signal_match_rate": round(signal_match_rate, 4),
            "avg_fill_slippage_bps": round(avg_fill_slippage_bps, 2),
            "timing_delay_bars": round(timing_delay_bars, 2),
            "pnl_gap_pct": pnl_gap_pct,
        },
        "signal_distribution": {"live": live_dist, "backtest": bt_dist},
        "signal_changes": {"live": live_changes, "backtest": bt_changes},
        "fills": {
            "open_count": len(open_fills),
            "close_count": len(close_fills),
            "live_total_pnl_usd": round(live_total_pnl, 4),
            "live_total_pnl_pct": round(live_total_pnl_pct, 4),
            "bt_total_pnl_pct": round(bt_total_pnl_pct, 4),
            "bt_trade_count": len(bt_trades),
        },
        "slippage_samples": len(slippage_bps_list),
        "timing_delay_samples": len(timing_delays),
        "divergence_causes": divergence_causes,
        "disagreements_sample": disagreements[:20],
    }

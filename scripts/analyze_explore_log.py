#!/usr/bin/env python3
"""Analyze quant_trader dry-run logs to evaluate signal quality.

Usage:
    python scripts/analyze_explore_log.py ext/rust/1s_explore.log
    python scripts/analyze_explore_log.py ext/rust/3m_explore.log

Computes:
    - Direction hit rate: does signal predict next bar's price direction?
    - Per-signal analysis: flow, large_trade, vol_spike, micro, ml
    - Threshold sweep: optimal signal threshold for each component
    - Combined signal analysis: ML + micro confirmation vs contradiction
"""

import re
import sys
from collections import defaultdict

def parse_bar_lines(filepath):
    """Parse 'bar' log lines into structured records."""
    bars = []
    # Match bar log lines — handle both log formats
    bar_re = re.compile(
        r'bar\s+'
        r'symbol=(\S+)\s+'
        r'close="([^"]+)"\s+'
        r'ml(?:_score)?="([^"]+)"\s+'
        r'raw="([^"]+)"\s+'
        r'micro="([^"]+)"\s+'
        r'flow="([^"]+)"\s+'
        r'vol_spike="([^"]+)"\s+'
        r'large="([^"]+)"\s+'
        r'idx=(\d+)\s+'
        r'ticks=(\d+)'
    )

    with open(filepath) as f:
        for line in f:
            # Strip ANSI codes
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            m = bar_re.search(line)
            if m:
                bars.append({
                    'symbol': m.group(1),
                    'close': float(m.group(2)),
                    'ml': float(m.group(3)),
                    'raw': float(m.group(4)),
                    'micro': float(m.group(5)),
                    'flow': float(m.group(6)),
                    'vol_spike': float(m.group(7)),
                    'large': float(m.group(8)),
                    'idx': int(m.group(9)),
                    'ticks': int(m.group(10)),
                })
    return bars


def direction_hit_rate(signals, returns):
    """Compute hit rate: fraction of times signal direction matches return direction."""
    if len(signals) == 0:
        return 0.0, 0
    hits = sum(1 for s, r in zip(signals, returns) if s * r > 0)
    total = sum(1 for s, r in zip(signals, returns) if abs(s) > 1e-9)
    return hits / total if total > 0 else 0.0, total


def threshold_sweep(signals, returns, thresholds):
    """Find optimal threshold for a signal."""
    best_hr, best_thresh, best_n = 0.0, 0.0, 0
    for t in thresholds:
        filtered = [(s, r) for s, r in zip(signals, returns) if abs(s) >= t]
        if len(filtered) < 10:
            continue
        hits = sum(1 for s, r in filtered if s * r > 0)
        hr = hits / len(filtered)
        if hr > best_hr:
            best_hr = hr
            best_thresh = t
            best_n = len(filtered)
    return best_hr, best_thresh, best_n


def analyze(bars):
    """Run full signal analysis."""
    if len(bars) < 20:
        print(f"Only {len(bars)} bars — need at least 20 for analysis. Keep collecting data.")
        return

    # Compute forward returns (next bar close / current close - 1)
    returns = []
    for i in range(len(bars) - 1):
        r = (bars[i + 1]['close'] - bars[i]['close']) / bars[i]['close']
        returns.append(r)

    # Align signals with forward returns (drop last bar)
    n = len(returns)
    signals = {
        'ml': [bars[i]['ml'] for i in range(n)],
        'raw': [bars[i]['raw'] for i in range(n)],
        'micro': [bars[i]['micro'] for i in range(n)],
        'flow': [bars[i]['flow'] for i in range(n)],
        'large': [bars[i]['large'] for i in range(n)],
    }

    print(f"\n{'='*60}")
    print(f"Signal Analysis — {n} bars")
    print(f"Price range: {min(b['close'] for b in bars):.2f} - {max(b['close'] for b in bars):.2f}")
    avg_abs_ret = sum(abs(r) for r in returns) / len(returns)
    print(f"Avg |return|: {avg_abs_ret*100:.4f}%")
    print(f"{'='*60}\n")

    # 1. Raw hit rates
    print("--- Direction Hit Rates (signal direction vs next bar direction) ---")
    print(f"{'Signal':<12} {'Hit Rate':>10} {'N signals':>10} {'Edge vs 50%':>12}")
    for name, sig in signals.items():
        hr, total = direction_hit_rate(sig, returns)
        edge = hr - 0.5
        marker = " ***" if abs(edge) > 0.02 and total >= 20 else ""
        print(f"{name:<12} {hr*100:>9.1f}% {total:>10d} {edge*100:>+11.1f}%{marker}")

    # 2. Threshold sweep
    print(f"\n--- Optimal Thresholds (best hit rate at threshold) ---")
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"{'Signal':<12} {'Best HR':>10} {'Threshold':>10} {'N signals':>10}")
    for name, sig in signals.items():
        hr, thresh, total = threshold_sweep(sig, returns, thresholds)
        print(f"{name:<12} {hr*100:>9.1f}% {thresh:>10.2f} {total:>10d}")

    # 3. ML + micro agreement analysis
    ml_sigs = signals['ml']
    micro_sigs = signals['micro']
    agree, disagree, ml_only, micro_only = [], [], [], []
    agree_ret, disagree_ret, ml_only_ret, micro_only_ret = [], [], [], []

    for i in range(n):
        ml, mi, r = ml_sigs[i], micro_sigs[i], returns[i]
        has_ml = abs(ml) > 1e-9
        has_micro = abs(mi) > 0.1

        if has_ml and has_micro:
            if ml * mi > 0:  # same direction
                agree.append(ml)
                agree_ret.append(r)
            else:
                disagree.append(ml)
                disagree_ret.append(r)
        elif has_ml:
            ml_only.append(ml)
            ml_only_ret.append(r)
        elif has_micro:
            micro_only.append(mi)
            micro_only_ret.append(r)

    print(f"\n--- ML + Micro Confirmation Analysis ---")
    scenarios = [
        ("ML+Micro agree", agree, agree_ret),
        ("ML+Micro disagree", disagree, disagree_ret),
        ("ML only", ml_only, ml_only_ret),
        ("Micro only", micro_only, micro_only_ret),
    ]
    print(f"{'Scenario':<22} {'Hit Rate':>10} {'N':>6} {'Avg |ret|':>10}")
    for name, sigs, rets in scenarios:
        if len(sigs) == 0:
            print(f"{name:<22} {'n/a':>10} {0:>6}")
            continue
        hr, total = direction_hit_rate(sigs, rets)
        avg_r = sum(abs(r) for r in rets) / len(rets) * 100
        print(f"{name:<22} {hr*100:>9.1f}% {total:>6d} {avg_r:>9.4f}%")

    # 4. Volume spike interaction
    print(f"\n--- Volume Spike Interaction ---")
    vol_spikes = [bars[i]['vol_spike'] for i in range(n)]
    for vol_thresh in [1.5, 2.0, 3.0]:
        high_vol = [(signals['micro'][i], returns[i])
                     for i in range(n) if vol_spikes[i] >= vol_thresh]
        low_vol = [(signals['micro'][i], returns[i])
                    for i in range(n) if vol_spikes[i] < vol_thresh]
        if len(high_vol) >= 5:
            hr_high, n_high = direction_hit_rate(
                [s for s, _ in high_vol], [r for _, r in high_vol])
            hr_low, n_low = direction_hit_rate(
                [s for s, _ in low_vol], [r for _, r in low_vol])
            print(f"  vol_spike >= {vol_thresh}: micro HR = {hr_high*100:.1f}% (n={n_high})"
                  f"  |  < {vol_thresh}: {hr_low*100:.1f}% (n={n_low})")

    # 5. Signal correlation with returns
    print(f"\n--- Signal-Return Correlation ---")
    for name, sig in signals.items():
        active = [(s, r) for s, r in zip(sig, returns) if abs(s) > 1e-9]
        if len(active) < 10:
            print(f"  {name}: insufficient data")
            continue
        sx = [s for s, _ in active]
        sy = [r for _, r in active]
        mx, my = sum(sx)/len(sx), sum(sy)/len(sy)
        cov = sum((x-mx)*(y-my) for x, y in zip(sx, sy)) / len(sx)
        std_x = (sum((x-mx)**2 for x in sx) / len(sx)) ** 0.5
        std_y = (sum((y-my)**2 for y in sy) / len(sy)) ** 0.5
        corr = cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0
        print(f"  {name}: corr = {corr:+.4f} (n={len(active)})")

    print(f"\n{'='*60}")
    # Summary recommendation
    best_signal = max(signals.keys(),
                      key=lambda k: abs(direction_hit_rate(signals[k], returns)[0] - 0.5))
    best_hr, best_n = direction_hit_rate(signals[best_signal], returns)
    print(f"Best single signal: {best_signal} (HR={best_hr*100:.1f}%, n={best_n})")

    if best_hr > 0.52 and best_n >= 50:
        print("PROMISING: Hit rate > 52% with sufficient samples")
    elif best_hr > 0.52:
        print("TENTATIVE: Hit rate looks good but need more samples")
    else:
        print("INSUFFICIENT EDGE: No signal > 52% hit rate yet")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_explore_log.py <log_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    bars = parse_bar_lines(filepath)
    print(f"Parsed {len(bars)} bar records from {filepath}")
    analyze(bars)

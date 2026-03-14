"""V1 rule-based signal generation for crypto prediction markets."""
from __future__ import annotations
from typing import Dict
import math


def generate_signal(
    features: Dict[str, float],
    threshold: float = 0.15,
    min_hours: float = 6.0,
) -> float:
    """Generate directional signal for a prediction market.

    Returns signal in [-1, +1]:
      positive = buy YES (probability underpriced)
      negative = buy NO (probability overpriced)
      0 = no signal
    """
    hours = features.get("hours_to_expiry", 0)
    if hours < min_hours:
        return 0.0

    # Core signal: agreement between BTC price direction and probability mispricing
    btc_vs_strike = features.get("btc_price_vs_strike", 0)
    prob_zscore = features.get("prob_zscore_24h", 0)

    if math.isnan(btc_vs_strike) or math.isnan(prob_zscore):
        return 0.0

    # BTC above strike + probability below average -> YES underpriced
    # BTC below strike + probability above average -> NO underpriced
    signal = 0.0
    if btc_vs_strike > 0.02 and prob_zscore < -0.5:
        signal = min(1.0, abs(prob_zscore) * 0.3 + abs(btc_vs_strike) * 2.0)
    elif btc_vs_strike < -0.02 and prob_zscore > 0.5:
        signal = -min(1.0, abs(prob_zscore) * 0.3 + abs(btc_vs_strike) * 2.0)

    # Amplify by depth imbalance (confirming signal)
    depth_imb = features.get("depth_imbalance", 0)
    if not math.isnan(depth_imb):
        if (signal > 0 and depth_imb > 0.2) or (signal < 0 and depth_imb < -0.2):
            signal *= 1.2

    # Dampen near extreme probabilities
    prob_level = features.get("prob_level", 0.5)
    if prob_level < 0.15 or prob_level > 0.85:
        signal *= 0.3

    return signal if abs(signal) >= threshold else 0.0

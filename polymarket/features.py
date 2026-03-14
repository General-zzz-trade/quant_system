"""Probability market feature computation (~30 features)."""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional


def compute_features(
    prob_history: np.ndarray,
    orderbook: Dict[str, Any],
    trades: Dict[str, Any],
    expiry_hours: float,
    btc_price: float = 0.0,
    btc_strike: float = 0.0,
) -> Dict[str, float]:
    """Compute features for a single prediction market.

    Returns dict of ~30 named features. Missing data -> NaN.
    """
    feats: Dict[str, float] = {}
    n = len(prob_history)

    # Probability momentum (returns over different horizons)
    for h, name in [(1, "1h"), (4, "4h"), (12, "12h"), (24, "24h")]:
        if n > h:
            feats[f"prob_ret_{name}"] = float(prob_history[-1] - prob_history[-1-h])
        else:
            feats[f"prob_ret_{name}"] = float("nan")

    # Mean reversion (z-scores)
    for w, name in [(24, "24h"), (72, "72h")]:
        if n >= w:
            window = prob_history[-w:]
            mu = float(np.mean(window))
            std = float(np.std(window))
            feats[f"prob_zscore_{name}"] = (float(prob_history[-1]) - mu) / std if std > 1e-8 else 0.0
        else:
            feats[f"prob_zscore_{name}"] = float("nan")

    # Volatility
    for w, name in [(12, "12h"), (24, "24h")]:
        if n >= w:
            feats[f"prob_vol_{name}"] = float(np.std(np.diff(prob_history[-w:])))
        else:
            feats[f"prob_vol_{name}"] = float("nan")

    # Orderbook features
    bid = orderbook.get("best_bid", 0)
    ask = orderbook.get("best_ask", 0)
    feats["bid_ask_spread"] = float(ask - bid) if ask and bid else float("nan")
    feats["mid_price"] = float((bid + ask) / 2) if ask and bid else float(prob_history[-1]) if n > 0 else float("nan")
    bid_depth = orderbook.get("bid_depth", 0)
    ask_depth = orderbook.get("ask_depth", 0)
    total_depth = bid_depth + ask_depth
    feats["depth_imbalance"] = float((bid_depth - ask_depth) / total_depth) if total_depth > 0 else 0.0
    feats["total_depth"] = float(total_depth)

    # Trade flow
    feats["trade_intensity"] = float(trades.get("count_1h", 0))
    buy_vol = trades.get("buy_volume", 0)
    sell_vol = trades.get("sell_volume", 0)
    total_vol = buy_vol + sell_vol
    feats["buy_sell_ratio"] = float(buy_vol / total_vol) if total_vol > 0 else 0.5
    feats["large_trade_flag"] = 1.0 if trades.get("max_trade_size", 0) > 500 else 0.0
    feats["net_flow"] = float(buy_vol - sell_vol)

    # Expiry
    feats["hours_to_expiry"] = float(expiry_hours)
    feats["time_decay_rate"] = 1.0 / max(expiry_hours, 1.0)
    feats["log_hours_to_expiry"] = float(np.log1p(max(expiry_hours, 0)))

    # Cross-market (BTC price vs market strike)
    if btc_strike > 0 and btc_price > 0:
        feats["btc_price_vs_strike"] = float((btc_price - btc_strike) / btc_strike)
    else:
        feats["btc_price_vs_strike"] = 0.0

    # BTC momentum (placeholder -- would need BTC price history in production)
    feats["btc_momentum_12h"] = 0.0  # TODO: wire to Binance data

    # Probability level features
    current_prob = float(prob_history[-1]) if n > 0 else 0.5
    feats["prob_level"] = current_prob
    feats["prob_from_center"] = abs(current_prob - 0.5)
    feats["prob_extreme"] = 1.0 if current_prob < 0.15 or current_prob > 0.85 else 0.0

    # Trend features
    if n >= 24:
        ma_24 = float(np.mean(prob_history[-24:]))
        feats["prob_vs_ma24"] = current_prob - ma_24
    else:
        feats["prob_vs_ma24"] = float("nan")

    if n >= 6:
        recent = prob_history[-6:]
        feats["prob_acceleration"] = float(np.mean(np.diff(recent)))
    else:
        feats["prob_acceleration"] = float("nan")

    return feats

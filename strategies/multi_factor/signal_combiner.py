from __future__ import annotations

from dataclasses import dataclass
from math import tanh

from strategies.multi_factor.feature_computer import MultiFactorFeatures
from strategies.multi_factor.regime import Regime


@dataclass(frozen=True)
class CombinedSignal:
    direction: int  # +1 long, -1 short, 0 flat
    strength: float  # 0.0~1.0
    regime: Regime
    components: dict


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _trend_score(features: MultiFactorFeatures) -> tuple[float, dict]:
    components: dict = {}

    # MA cross: linear mapping of (fast - slow) / slow
    if features.sma_fast is not None and features.sma_slow is not None and features.sma_slow != 0:
        ma_diff = (features.sma_fast - features.sma_slow) / features.sma_slow
        ma_score = _clamp(ma_diff * 100)  # 1% diff → score 1.0
    else:
        ma_score = 0.0
    components["ma_cross"] = ma_score

    # MACD: tanh(histogram / atr) for normalization
    if features.macd_hist is not None and features.atr is not None and features.atr > 0:
        macd_score = tanh(features.macd_hist / features.atr)
    elif features.macd_hist is not None:
        macd_score = _clamp(features.macd_hist * 10)
    else:
        macd_score = 0.0
    components["macd"] = macd_score

    # RSI: 50-70 confirms long, >80 overbought warning, <30 oversold warning
    if features.rsi is not None:
        if features.rsi > 80:
            rsi_score = -0.5  # overbought, caution
        elif features.rsi > 50:
            rsi_score = (features.rsi - 50) / 30  # 50→0, 80→1
        elif features.rsi < 20:
            rsi_score = 0.5  # oversold, caution for shorts
        elif features.rsi < 50:
            rsi_score = (features.rsi - 50) / 30  # 50→0, 20→-1
        else:
            rsi_score = 0.0
    else:
        rsi_score = 0.0
    components["rsi"] = rsi_score

    composite = 0.35 * ma_score + 0.35 * macd_score + 0.30 * rsi_score
    return composite, components


def _range_score(features: MultiFactorFeatures) -> tuple[float, dict]:
    components: dict = {}

    # BB %B: <0.2 strong buy, >0.8 strong sell
    if features.bb_pct is not None:
        bb_score = _clamp(1.0 - 2.0 * features.bb_pct)  # 0→+1, 0.5→0, 1→-1
    else:
        bb_score = 0.0
    components["bb_pct"] = bb_score

    # RSI: <30 buy +1, >70 sell -1
    if features.rsi is not None:
        if features.rsi < 30:
            rsi_score = (30 - features.rsi) / 30  # 30→0, 0→1
        elif features.rsi > 70:
            rsi_score = -(features.rsi - 70) / 30  # 70→0, 100→-1
        else:
            rsi_score = 0.0
    else:
        rsi_score = 0.0
    components["rsi"] = rsi_score

    # MACD histogram direction
    if features.macd_hist is not None and features.atr is not None and features.atr > 0:
        macd_score = tanh(features.macd_hist / features.atr)
    elif features.macd_hist is not None:
        macd_score = _clamp(features.macd_hist * 10)
    else:
        macd_score = 0.0
    components["macd"] = macd_score

    composite = 0.50 * bb_score + 0.30 * rsi_score + 0.20 * macd_score
    return composite, components


def combine_signals(
    features: MultiFactorFeatures,
    regime: Regime,
    trend_threshold: float = 0.5,
    range_threshold: float = 0.4,
) -> CombinedSignal:
    if regime == Regime.HIGH_VOL:
        return CombinedSignal(direction=0, strength=0.0, regime=regime, components={"reason": "high_vol"})

    if regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
        composite, components = _trend_score(features)
        threshold = trend_threshold
    else:
        composite, components = _range_score(features)
        threshold = range_threshold

    abs_score = abs(composite)
    if abs_score < threshold:
        return CombinedSignal(direction=0, strength=abs_score, regime=regime, components=components)

    direction = 1 if composite > 0 else -1
    return CombinedSignal(direction=direction, strength=abs_score, regime=regime, components=components)

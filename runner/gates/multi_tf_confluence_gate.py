# runner/gates/multi_tf_confluence_gate.py
"""Multi-timeframe confluence gate — scale by 1h vs 4h trend alignment.

When 1h signal direction aligns with 4h trend:
  - Boost position (scale=1.2) — confluent move
When 1h signal opposes 4h trend:
  - Reduce position (scale=0.5) — counter-trend, higher risk
When 4h is neutral/ranging:
  - Normal position (scale=1.0)

Uses tf4h_close_vs_ma20, tf4h_rsi_14, tf4h_macd_hist from context
(populated by EnrichedFeatureComputer 4h resampling).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from runner.gate_chain import GateResult

_log = logging.getLogger(__name__)


@dataclass
class MultiTFConfluenceConfig:
    """Configuration for multi-timeframe confluence gate."""
    enabled: bool = True

    # 4h trend thresholds
    ma_dev_threshold: float = 0.005     # |close_vs_ma20| > 0.5% = trending
    rsi_overbought: float = 65.0        # RSI > 65 = bullish
    rsi_oversold: float = 35.0          # RSI < 35 = bearish
    macd_threshold: float = 0.0         # MACD > 0 = bullish

    # Scaling factors
    aligned_scale: float = 1.2          # 1h + 4h agree → boost
    opposed_scale: float = 0.5          # 1h opposes 4h → reduce
    neutral_scale: float = 1.0          # 4h neutral → no change

    # Require at least N of 3 indicators to agree for trend classification
    min_confirming: int = 2


class MultiTFConfluenceGate:
    """Gate: scale position by 1h vs 4h trend alignment.

    Classifies 4h trend direction from three indicators:
      - close_vs_ma20: price relative to 20-period MA on 4h
      - rsi_14: RSI on 4h (>65 bullish, <35 bearish)
      - macd_hist: MACD histogram on 4h (>0 bullish, <0 bearish)

    Requires min_confirming indicators to agree for a trend signal.
    Then compares with 1h alpha signal direction.
    """

    name = "MultiTFConfluence"

    def __init__(self, cfg: MultiTFConfluenceConfig | None = None) -> None:
        self._cfg = cfg or MultiTFConfluenceConfig()
        self._total_checks = 0
        self._aligned_count = 0
        self._opposed_count = 0

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._cfg.enabled:
            return GateResult(allowed=True, scale=1.0)

        self._total_checks += 1
        cfg = self._cfg

        # Get 1h signal direction
        signal = 0
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            try:
                signal = int(meta.get("signal", 0))
            except (ValueError, TypeError):
                signal = 0
        if signal == 0:
            try:
                signal = int(context.get("signal", 0))
            except (ValueError, TypeError):
                signal = 0

        if signal == 0:
            return GateResult(allowed=True, scale=1.0)

        # Get 4h indicators from context
        ma_dev = _safe_float(context.get("tf4h_close_vs_ma20"))
        rsi = _safe_float(context.get("tf4h_rsi_14"))
        macd = _safe_float(context.get("tf4h_macd_hist"))

        # If no 4h data available, pass through
        if ma_dev is None and rsi is None and macd is None:
            return GateResult(allowed=True, scale=1.0)

        # Classify 4h trend direction via voting
        bullish_votes = 0
        bearish_votes = 0
        total_votes = 0

        if ma_dev is not None:
            total_votes += 1
            if ma_dev > cfg.ma_dev_threshold:
                bullish_votes += 1
            elif ma_dev < -cfg.ma_dev_threshold:
                bearish_votes += 1

        if rsi is not None:
            total_votes += 1
            if rsi > cfg.rsi_overbought:
                bullish_votes += 1
            elif rsi < cfg.rsi_oversold:
                bearish_votes += 1

        if macd is not None:
            total_votes += 1
            if macd > cfg.macd_threshold:
                bullish_votes += 1
            elif macd < -cfg.macd_threshold:
                bearish_votes += 1

        # Determine 4h trend
        trend_4h = 0  # 0 = neutral
        if bullish_votes >= cfg.min_confirming:
            trend_4h = 1
        elif bearish_votes >= cfg.min_confirming:
            trend_4h = -1

        # If 4h is neutral, no adjustment
        if trend_4h == 0:
            return GateResult(allowed=True, scale=cfg.neutral_scale)

        # Check alignment
        aligned = (signal == trend_4h)

        if aligned:
            self._aligned_count += 1
            scale = cfg.aligned_scale
            reason = f"tf_aligned signal={signal} trend_4h={trend_4h} scale={scale}"
        else:
            self._opposed_count += 1
            scale = cfg.opposed_scale
            reason = f"tf_opposed signal={signal} trend_4h={trend_4h} scale={scale}"

        _log.debug(
            "MultiTFConfluence: signal=%d trend_4h=%d bull=%d bear=%d → scale=%.2f",
            signal, trend_4h, bullish_votes, bearish_votes, scale,
        )
        return GateResult(allowed=True, scale=scale, reason=reason)

    @property
    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "aligned": self._aligned_count,
            "opposed": self._opposed_count,
            "aligned_rate": self._aligned_count / max(self._total_checks, 1),
        }


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None for NaN/None/missing."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None

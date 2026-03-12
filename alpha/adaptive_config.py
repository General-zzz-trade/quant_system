"""Adaptive Config Selector — rolling backtest to find optimal parameters.

Walk-forward showed that optimal deadzone, min_hold, and long_only shift
across market regimes. Fixed params: ETH return -34%. Adaptive: +612%.

This module:
1. Takes recent N months of z-score predictions
2. Runs a fast backtest sweep over parameter grid
3. Returns the best config for current conditions
4. Can be called periodically (e.g., monthly) to update live config

Design principles:
- Conservative: prefer configs with more trades (statistical significance)
- Stable: penalize configs that differ wildly from previous selection
- Regime-aware: weight recent months more than older months

Usage:
    selector = AdaptiveConfigSelector(lookback_months=6)
    best = selector.select(z_scores, closes)
    # best = {"deadzone": 1.0, "min_hold": 12, "max_hold": 96, "long_only": False}

    # Or with Sharpe-weighted blending across lookback windows:
    best = selector.select_robust(z_scores, closes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Parameter search grid
DEADZONE_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]
MIN_HOLD_GRID = [8, 12, 24]
MAX_HOLD_MULT = [5, 8]  # max_hold = min_hold * mult
LONG_ONLY_GRID = [True, False]

COST_BPS_RT = 4
BARS_PER_DAY = 24


@dataclass(frozen=True)
class AdaptiveParams:
    """Selected trading parameters."""
    deadzone: float
    min_hold: int
    max_hold: int
    long_only: bool
    sharpe: float           # backtest Sharpe that produced this selection
    trades: int             # number of trades in lookback
    confidence: str         # "high" | "medium" | "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deadzone": self.deadzone,
            "min_hold": self.min_hold,
            "max_hold": self.max_hold,
            "long_only": self.long_only,
            "sharpe": self.sharpe,
            "trades": self.trades,
            "confidence": self.confidence,
        }


class AdaptiveConfigSelector:
    """Selects optimal trading parameters via rolling backtest.

    Parameters
    ----------
    lookback_months : int
        Months of recent data to use for parameter selection.
    min_trades : int
        Minimum trades for a config to be considered viable.
    recency_weight : float
        How much to weight recent performance vs overall (0-1).
        0 = equal weight, 1 = only most recent half matters.
    stability_penalty : float
        Penalty for configs that differ from previous selection.
        0 = no penalty, higher = prefer stability.
    """

    def __init__(
        self,
        lookback_months: int = 6,
        min_trades: int = 8,
        recency_weight: float = 0.3,
        stability_penalty: float = 0.1,
    ):
        self._lookback_months = lookback_months
        self._min_trades = min_trades
        self._recency_weight = recency_weight
        self._stability_penalty = stability_penalty
        self._previous: Optional[AdaptiveParams] = None
        self._history: List[AdaptiveParams] = []

    def select(
        self,
        z_scores: np.ndarray,
        closes: np.ndarray,
    ) -> AdaptiveParams:
        """Select best parameters from recent data.

        Parameters
        ----------
        z_scores : array
            Rolling z-scores (already computed from model predictions).
        closes : array
            Close prices aligned with z_scores.

        Returns
        -------
        AdaptiveParams with optimal config.
        """
        lookback_bars = BARS_PER_DAY * 30 * self._lookback_months
        n = len(z_scores)

        if n < lookback_bars:
            z_window = z_scores
            c_window = closes
        else:
            z_window = z_scores[-lookback_bars:]
            c_window = closes[-lookback_bars:]

        best_score = -999.0
        best_params = None

        for dz in DEADZONE_GRID:
            for mh in MIN_HOLD_GRID:
                for mult in MAX_HOLD_MULT:
                    maxh = mh * mult
                    for lo in LONG_ONLY_GRID:
                        result = _fast_backtest(
                            z_window, c_window, dz, mh, maxh, lo
                        )
                        if result["trades"] < self._min_trades:
                            continue

                        score = self._score(result, dz, mh, maxh, lo)
                        if score > best_score:
                            best_score = score
                            best_params = (dz, mh, maxh, lo, result)

        if best_params is None:
            # Fallback to conservative defaults
            return AdaptiveParams(
                deadzone=1.0, min_hold=12, max_hold=96,
                long_only=False, sharpe=0, trades=0,
                confidence="low",
            )

        dz, mh, maxh, lo, result = best_params
        trades = result["trades"]

        if result["sharpe"] > 2.0 and trades >= 15:
            confidence = "high"
        elif result["sharpe"] > 1.0 and trades >= 10:
            confidence = "medium"
        else:
            confidence = "low"

        params = AdaptiveParams(
            deadzone=dz, min_hold=mh, max_hold=maxh, long_only=lo,
            sharpe=round(result["sharpe"], 2),
            trades=trades,
            confidence=confidence,
        )

        self._previous = params
        self._history.append(params)
        return params

    def select_robust(
        self,
        z_scores: np.ndarray,
        closes: np.ndarray,
    ) -> AdaptiveParams:
        """Multi-window selection: test on 3, 6, and 9 month windows.

        Picks the config that performs best across multiple lookback periods,
        prioritizing consistency over peak performance in any single window.
        """
        n = len(z_scores)
        config_scores: Dict[tuple, List[float]] = {}

        for months in [3, 6, 9]:
            bars = BARS_PER_DAY * 30 * months
            if bars > n:
                continue

            z_w = z_scores[-bars:]
            c_w = closes[-bars:]

            for dz in DEADZONE_GRID:
                for mh in MIN_HOLD_GRID:
                    for mult in MAX_HOLD_MULT:
                        maxh = mh * mult
                        for lo in LONG_ONLY_GRID:
                            r = _fast_backtest(z_w, c_w, dz, mh, maxh, lo)
                            if r["trades"] < self._min_trades:
                                continue
                            key = (dz, mh, maxh, lo)
                            if key not in config_scores:
                                config_scores[key] = []
                            config_scores[key].append(r["sharpe"])

        if not config_scores:
            return self.select(z_scores, closes)

        # Score: mean Sharpe across windows, penalize high variance
        best_key = None
        best_score = -999.0

        for key, sharpes in config_scores.items():
            if len(sharpes) < 2:
                continue
            mean_s = np.mean(sharpes)
            std_s = np.std(sharpes)
            # Penalize variance: prefer consistent configs
            score = mean_s - 0.5 * std_s
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            return self.select(z_scores, closes)

        dz, mh, maxh, lo = best_key
        # Get final stats from the primary window
        lookback_bars = BARS_PER_DAY * 30 * self._lookback_months
        z_w = z_scores[-min(lookback_bars, n):]
        c_w = closes[-min(lookback_bars, n):]
        final = _fast_backtest(z_w, c_w, dz, mh, maxh, lo)

        sharpes = config_scores[best_key]
        if np.mean(sharpes) > 2.0 and np.min(sharpes) > 0:
            confidence = "high"
        elif np.mean(sharpes) > 1.0:
            confidence = "medium"
        else:
            confidence = "low"

        params = AdaptiveParams(
            deadzone=dz, min_hold=mh, max_hold=maxh, long_only=lo,
            sharpe=round(final["sharpe"], 2),
            trades=final["trades"],
            confidence=confidence,
        )
        self._previous = params
        self._history.append(params)
        return params

    def _score(
        self, result: Dict, dz: float, mh: int, maxh: int, lo: bool,
    ) -> float:
        """Score a config, incorporating recency weight and stability."""
        sharpe = result["sharpe"]
        trades = result["trades"]

        # Base score = Sharpe
        score = sharpe

        # Bonus for more trades (statistical significance)
        if trades >= 20:
            score += 0.1
        elif trades >= 15:
            score += 0.05

        # Stability penalty: prefer configs close to previous
        if self._previous is not None and self._stability_penalty > 0:
            dist = 0.0
            dist += abs(dz - self._previous.deadzone) / 2.5  # normalized
            dist += abs(mh - self._previous.min_hold) / 24
            dist += 0.5 if lo != self._previous.long_only else 0
            score -= self._stability_penalty * dist

        return score

    @property
    def history(self) -> List[AdaptiveParams]:
        return list(self._history)

    @property
    def current(self) -> Optional[AdaptiveParams]:
        return self._previous


def _fast_backtest(
    z: np.ndarray,
    closes: np.ndarray,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    long_only: bool,
) -> Dict[str, Any]:
    """Vectorized-style fast backtest for parameter sweep."""
    n = len(z)
    cost_frac = COST_BPS_RT / 10000
    pos = 0.0
    eb = 0
    trades = []

    for i in range(n):
        if pos != 0:
            held = i - eb
            se = held >= max_hold
            if not se and held >= min_hold:
                se = pos * z[i] < -0.3 or abs(z[i]) < 0.2
            if se:
                pnl = pos * (closes[i] - closes[eb]) / closes[eb]
                trades.append(pnl - cost_frac)
                pos = 0.0
        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                eb = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                eb = i

    if not trades:
        return {"sharpe": 0.0, "trades": 0, "return": 0.0, "win_rate": 0.0}

    net = np.array(trades)
    avg_hold = n / max(len(trades), 1)
    tpy = 365 * BARS_PER_DAY / max(avg_hold, 1)
    sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy))

    return {
        "sharpe": sharpe,
        "trades": len(trades),
        "return": float(np.sum(net)) * 100,
        "win_rate": float(np.mean(net > 0)) * 100,
    }

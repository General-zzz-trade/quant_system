"""The strategy function shared by research and live trading.

``compute_targets(hourly, funding, spec)`` returns, for every closed 1h bar,
the target position as a fraction of sleeve equity (signed, |w| <= max_weight).
It uses only data available at that bar's close:

- daily layer (core): multi-lookback trend score, inverse-volatility sizing to
  a target annual volatility, optional long-only clip;
- funding filter: halve exposure on the crowded side when the 3-day mean
  funding is in the top / bottom 5% of its trailing year;
- 4h layer: halve exposure when 24h realized volatility exceeds twice the
  daily estimate (vol spike) and/or when the 24h trend disagrees (confirm);
- the 1h clock only decides WHEN to trade (rebalancing band, execution mode).

All features are finite rolling windows, so computing on a trailing window of
``spec.window_hours()`` bars gives exactly the value computed on full history.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading.domain.market import aggregate

SETTLEMENTS_PER_YEAR = 3 * 365


@dataclass(frozen=True)
class StrategySpec:
    lookbacks: tuple[int, ...] = (20, 60, 120)   # days
    signal: str = "sign"                          # sign | scaled
    long_only: bool = False
    vol_lookback: int = 30                        # days
    target_vol: float = 0.30                      # annualized, per sleeve
    max_weight: float = 0.9
    funding_filter: bool = False
    vol_spike_cut: bool = False
    confirm_4h: bool = False
    band: float = 0.10                            # min |target - current| weight change to trade
    execution: str = "maker"                      # maker | market
    ttl_bars: int = 1                             # maker order lifetime in 1h bars

    def __post_init__(self) -> None:
        if not self.lookbacks or any(int(x) < 2 for x in self.lookbacks):
            raise ValueError("lookbacks must be >= 2 days")
        if self.signal not in ("sign", "scaled") or self.execution not in ("maker", "market"):
            raise ValueError("invalid signal or execution")
        if not 0 < self.max_weight <= 1 or not 0 < self.target_vol < 5 or not 0 <= self.band < 1:
            raise ValueError("invalid sizing parameters")
        if self.vol_lookback < 5 or self.ttl_bars < 1:
            raise ValueError("invalid lookback or ttl")

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        out["lookbacks"] = list(self.lookbacks)
        return out

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> "StrategySpec":
        return cls(**{**data, "lookbacks": tuple(int(x) for x in data["lookbacks"])})

    def window_hours(self) -> int:
        """Hourly history needed for exact trailing-window evaluation."""
        days = max(self.lookbacks) + self.vol_lookback + 3
        hours = days * 24
        return hours

    def funding_window(self) -> int:
        """Funding settlements needed (trailing year + 3-day mean)."""
        return SETTLEMENTS_PER_YEAR + 12 if self.funding_filter else 12


def _asof(times: np.ndarray, source_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Latest source value with source_time <= time (NaN before the first)."""
    idx = np.searchsorted(source_times, times, side="right") - 1
    out = np.full(len(times), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def daily_weights(hourly: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    daily = aggregate(hourly, 24)
    close = daily["close"]
    logret = np.log(close).diff()
    vol_d = logret.rolling(spec.vol_lookback, min_periods=spec.vol_lookback).std()
    scores = []
    for lookback in spec.lookbacks:
        if spec.signal == "sign":
            scores.append(np.sign(close / close.shift(lookback) - 1.0))
        else:
            move = np.log(close / close.shift(lookback))
            scores.append(np.tanh(move / (vol_d * math.sqrt(lookback))))
    score = pd.concat(scores, axis=1).mean(axis=1, skipna=False)
    if spec.long_only:
        score = score.clip(lower=0.0)
    vol_annual = vol_d * math.sqrt(365.0)
    weight = (score * spec.target_vol / vol_annual).clip(-spec.max_weight, spec.max_weight)
    return pd.DataFrame({"available_at": daily["available_at"], "weight": weight, "vol_annual": vol_annual})


def four_hour_features(hourly: pd.DataFrame) -> pd.DataFrame:
    bars = aggregate(hourly, 4)
    logret = np.log(bars["close"]).diff()
    vol_24h = logret.rolling(6, min_periods=6).std() * math.sqrt(6 * 365.0)
    momentum = np.sign(bars["close"] / bars["close"].shift(6) - 1.0)
    return pd.DataFrame({"available_at": bars["available_at"], "vol_24h": vol_24h, "momentum": momentum})


def funding_bands(funding: pd.Series) -> pd.DataFrame:
    mean3 = funding.rolling(9, min_periods=9).mean()
    hi = mean3.rolling(SETTLEMENTS_PER_YEAR, min_periods=365).quantile(0.95)
    lo = mean3.rolling(SETTLEMENTS_PER_YEAR, min_periods=365).quantile(0.05)
    return pd.DataFrame({"available_at": funding.index.values, "mean3": mean3.values, "hi": hi.values, "lo": lo.values})


def compute_targets(hourly: pd.DataFrame, funding: pd.Series, spec: StrategySpec) -> pd.DataFrame:
    """Target weight for every hourly close; ``ready`` is False until warm."""
    times = hourly["available_at"].to_numpy(dtype=np.int64)
    daily = daily_weights(hourly, spec)
    w = _asof(times, daily["available_at"].to_numpy(dtype=np.int64), daily["weight"].to_numpy(dtype=float))
    vol_d = _asof(times, daily["available_at"].to_numpy(dtype=np.int64), daily["vol_annual"].to_numpy(dtype=float))
    ready = np.isfinite(w)
    if spec.funding_filter:
        bands = funding_bands(funding)
        # funding settled at T is known at T (settlement at the hour boundary)
        ft = bands["available_at"].to_numpy(dtype=np.int64)
        mean3 = _asof(times, ft, bands["mean3"].to_numpy(dtype=float))
        hi = _asof(times, ft, bands["hi"].to_numpy(dtype=float))
        lo = _asof(times, ft, bands["lo"].to_numpy(dtype=float))
        crowded_long = (w > 0) & (mean3 > hi)
        crowded_short = (w < 0) & (mean3 < lo)
        w = np.where(crowded_long | crowded_short, w * 0.5, w)
        ready &= np.isfinite(hi) & np.isfinite(lo)
    if spec.vol_spike_cut or spec.confirm_4h:
        f4 = four_hour_features(hourly)
        t4 = f4["available_at"].to_numpy(dtype=np.int64)
        if spec.vol_spike_cut:
            vol4 = _asof(times, t4, f4["vol_24h"].to_numpy(dtype=float))
            w = np.where(vol4 > 2.0 * vol_d, w * 0.5, w)
            ready &= np.isfinite(vol4)
        if spec.confirm_4h:
            mom = _asof(times, t4, f4["momentum"].to_numpy(dtype=float))
            w = np.where((w != 0) & (np.sign(mom) != np.sign(w)), w * 0.5, w)
            ready &= np.isfinite(mom)
    w = np.where(ready, w, 0.0)
    return pd.DataFrame({"available_at": times, "target": w, "ready": ready})


def latest_target(hourly: pd.DataFrame, funding: pd.Series, spec: StrategySpec) -> tuple[float, bool]:
    """Live entry point: target at the most recent closed bar (same function)."""
    out = compute_targets(hourly, funding, spec)
    return float(out["target"].iloc[-1]), bool(out["ready"].iloc[-1])

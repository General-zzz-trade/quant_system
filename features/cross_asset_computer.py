"""CrossAssetComputer — cross-asset features for multi-symbol alpha.

Maintains per-symbol return/funding state and computes inter-asset features
like rolling beta, relative strength, rolling correlation, and funding spread.

Independent of EnrichedFeatureComputer (needs multi-symbol state).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from features.enriched_computer import _EMA
from features.rolling import RollingWindow


CROSS_ASSET_FEATURE_NAMES: tuple[str, ...] = (
    "btc_ret_1", "btc_ret_3", "btc_ret_6",
    "rolling_beta_30", "rolling_beta_60",
    "relative_strength_20",
    "rolling_corr_30",
    "funding_diff", "funding_diff_ma8",
    "spread_zscore_20",
)

_BENCHMARK = "BTCUSDT"


@dataclass
class _AssetState:
    """Per-asset state for cross-asset computations."""
    close_history: Deque[float] = field(default_factory=lambda: deque(maxlen=65))
    _last_funding_rate: Optional[float] = None
    _bar_count: int = 0

    def push(self, close: float, *, funding_rate: Optional[float] = None) -> None:
        self.close_history.append(close)
        self._bar_count += 1
        if funding_rate is not None:
            self._last_funding_rate = funding_rate

    def ret(self, lag: int) -> Optional[float]:
        n = len(self.close_history)
        if n <= lag:
            return None
        base = self.close_history[-1 - lag]
        if base == 0:
            return None
        return (self.close_history[-1] - base) / base


@dataclass
class _PairState:
    """State for a (symbol, benchmark) pair."""
    # Deques for raw return access (beta/corr need raw values)
    sym_rets_30: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    bench_rets_30: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    sym_rets_60: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    bench_rets_60: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    # For relative strength (cumulative return)
    sym_cum_20: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    bench_cum_20: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    # For spread z-score
    spread_window_20: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    # Funding diff
    funding_diff_ema: _EMA = field(default_factory=lambda: _EMA(span=8))
    _last_funding_diff: Optional[float] = None
    _bar_count: int = 0

    def push(self, sym_ret: float, bench_ret: float, *,
             funding_diff: Optional[float] = None) -> None:
        # Compute current beta for spread before pushing new data
        beta = _beta_from_deques(self.sym_rets_30, self.bench_rets_30)

        self._bar_count += 1
        self.sym_rets_30.append(sym_ret)
        self.bench_rets_30.append(bench_ret)
        self.sym_rets_60.append(sym_ret)
        self.bench_rets_60.append(bench_ret)
        self.sym_cum_20.append(sym_ret)
        self.bench_cum_20.append(bench_ret)

        # Spread using beta computed from *before* this bar
        if beta is not None:
            spread = sym_ret - beta * bench_ret
            self.spread_window_20.push(spread)

        if funding_diff is not None:
            self._last_funding_diff = funding_diff
            self.funding_diff_ema.push(funding_diff)


def _beta_from_deques(sym: Deque[float], bench: Deque[float]) -> Optional[float]:
    n = len(sym)
    if n < sym.maxlen or n != len(bench):
        return None
    s_mean = sum(sym) / n
    b_mean = sum(bench) / n
    cov = sum((s - s_mean) * (b - b_mean) for s, b in zip(sym, bench)) / n
    var_b = sum((b - b_mean) ** 2 for b in bench) / n
    if var_b < 1e-20:
        return None
    return cov / var_b


def _pearson(x: Deque[float], y: Deque[float]) -> Optional[float]:
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx < 1e-12 or vy < 1e-12:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(vx * vy)


@dataclass
class CrossAssetComputer:
    """Cross-asset feature computer. Maintains multi-symbol state.

    Usage: call on_bar() for each symbol at each timestamp
    (benchmark first, then other symbols), then get_features().
    """

    _assets: Dict[str, _AssetState] = field(default_factory=dict, init=False)
    _pairs: Dict[str, _PairState] = field(default_factory=dict, init=False)

    def on_bar(self, symbol: str, *, close: float,
               funding_rate: Optional[float] = None) -> None:
        """Update asset state. Non-benchmark symbols also update pair state."""
        if symbol not in self._assets:
            self._assets[symbol] = _AssetState()
        self._assets[symbol].push(close, funding_rate=funding_rate)

        if symbol != _BENCHMARK:
            bench_state = self._assets.get(_BENCHMARK)
            if bench_state is None:
                return
            pair_key = f"{symbol}_{_BENCHMARK}"
            if pair_key not in self._pairs:
                self._pairs[pair_key] = _PairState()
            pair = self._pairs[pair_key]
            sym_ret = self._assets[symbol].ret(1)
            bench_ret = bench_state.ret(1)
            if sym_ret is not None and bench_ret is not None:
                f_diff = None
                sym_fr = self._assets[symbol]._last_funding_rate
                bench_fr = bench_state._last_funding_rate
                if sym_fr is not None and bench_fr is not None:
                    f_diff = sym_fr - bench_fr
                pair.push(sym_ret, bench_ret, funding_diff=f_diff)

    def get_features(self, symbol: str,
                     benchmark: str = "BTCUSDT") -> Dict[str, Optional[float]]:
        """Read cross-asset features for symbol vs benchmark."""
        feats: Dict[str, Optional[float]] = {n: None for n in CROSS_ASSET_FEATURE_NAMES}

        if symbol == benchmark:
            return feats

        bench_state = self._assets.get(benchmark)
        sym_state = self._assets.get(symbol)
        if sym_state is None or bench_state is None:
            return feats

        feats["btc_ret_1"] = bench_state.ret(1)
        feats["btc_ret_3"] = bench_state.ret(3)
        feats["btc_ret_6"] = bench_state.ret(6)

        pair_key = f"{symbol}_{benchmark}"
        pair = self._pairs.get(pair_key)
        if pair is None:
            return feats

        # Rolling beta
        feats["rolling_beta_30"] = _beta_from_deques(pair.sym_rets_30, pair.bench_rets_30)
        feats["rolling_beta_60"] = _beta_from_deques(pair.sym_rets_60, pair.bench_rets_60)

        # Relative strength (20-bar)
        if len(pair.sym_cum_20) >= 20:
            sym_cum = 1.0
            bench_cum = 1.0
            for sr, br in zip(pair.sym_cum_20, pair.bench_cum_20):
                sym_cum *= (1.0 + sr)
                bench_cum *= (1.0 + br)
            if bench_cum != 0:
                feats["relative_strength_20"] = sym_cum / bench_cum

        # Rolling correlation (30-bar)
        if len(pair.sym_rets_30) >= 30:
            feats["rolling_corr_30"] = _pearson(pair.sym_rets_30, pair.bench_rets_30)

        # Funding diff
        feats["funding_diff"] = pair._last_funding_diff
        if pair.funding_diff_ema.ready:
            feats["funding_diff_ma8"] = pair.funding_diff_ema.value

        # Spread z-score
        if pair.spread_window_20.full:
            mean_s = pair.spread_window_20.mean
            std_s = pair.spread_window_20.std
            beta30 = feats["rolling_beta_30"]
            sym_ret_1 = sym_state.ret(1)
            bench_ret_1 = bench_state.ret(1)
            if (std_s is not None and std_s > 1e-12 and beta30 is not None
                    and sym_ret_1 is not None and bench_ret_1 is not None):
                spread = sym_ret_1 - beta30 * bench_ret_1
                feats["spread_zscore_20"] = (spread - mean_s) / std_s

        return feats

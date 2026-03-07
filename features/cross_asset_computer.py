"""CrossAssetComputer — cross-asset features for multi-symbol alpha.

Delegates all computation to RustCrossAssetComputer.
"""
from __future__ import annotations

from typing import Dict, Optional

from _quant_hotpath import RustCrossAssetComputer

CROSS_ASSET_FEATURE_NAMES: tuple[str, ...] = (
    "btc_ret_1", "btc_ret_3", "btc_ret_6",
    "btc_ret_12", "btc_ret_24",
    "btc_rsi_14",
    "btc_macd_line",
    "btc_mean_reversion_20",
    "btc_atr_norm_14",
    "btc_bb_width_20",
    "rolling_beta_30", "rolling_beta_60",
    "relative_strength_20",
    "rolling_corr_30",
    "funding_diff", "funding_diff_ma8",
    "spread_zscore_20",
)

_BENCHMARK = "BTCUSDT"


class CrossAssetComputer:
    """Cross-asset feature computer. Thin wrapper over RustCrossAssetComputer."""

    def __init__(self) -> None:
        self._inner = RustCrossAssetComputer(_BENCHMARK)

    def on_bar(self, symbol: str, *, close: float,
               funding_rate: Optional[float] = None,
               high: Optional[float] = None,
               low: Optional[float] = None) -> None:
        self._inner.on_bar(symbol, close, funding_rate=funding_rate,
                           high=high, low=low)

    def get_features(self, symbol: str,
                     benchmark: str = "BTCUSDT") -> Dict[str, Optional[float]]:
        return dict(self._inner.get_features(symbol, benchmark=benchmark))

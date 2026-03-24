# features/enriched_computer.py
"""EnrichedFeatureComputer — 30+ incremental features for ML alpha models.

Same interface as LiveFeatureComputer but computes a much richer feature set
including RSI, MACD, Bollinger Bands, ATR, multi-horizon returns, volume
profile, trend indicators, time-of-day, and crypto-native features.

All computations are O(1) per bar (incremental/EMA-based).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Re-export trackers for backward compat (moved to enriched_trackers.py)
from features.enriched_trackers import _EMA, _RSITracker, _ATRTracker, _ADXTracker  # noqa: F401

from _quant_hotpath import VWAPWindow

# VWAPWindow — Rust-accelerated volume-weighted average price window.
# Used for VWAP deviation features (vwap_dev_20) in the enriched feature set.
VWAPWindowType = VWAPWindow

from features.enriched_trackers import (  # noqa: E402
    _build_multi_dominance_ratios,
)

logger = logging.getLogger(__name__)

from features.enriched_feature_names import (  # noqa: E402
    _MULTI_DOMINANCE_PAIRS,  # noqa: F401 — re-exported
    _MULTI_DOMINANCE_PREFIXES,  # noqa: F401 — re-exported
    _ALL_MULTI_DOMINANCE_FEATURES,  # noqa: F401 — re-exported
    ENRICHED_FEATURE_NAMES,  # noqa: F401 — re-exported
    _WARMUP_BARS,  # noqa: F401 — re-exported
)


# _SymbolState extracted to features/enriched_symbol_state.py
from features.enriched_symbol_state import _SymbolState  # noqa: F401, E402


@dataclass
class EnrichedFeatureComputer:
    """Enriched incremental feature computer producing 25+ features per bar.

    Drop-in replacement for LiveFeatureComputer with richer feature set.
    Same interface: on_bar() + get_features_dict().
    """

    _states: Dict[str, _SymbolState] = field(default_factory=dict, init=False)

    def on_bar(
        self,
        symbol: str,
        *,
        close: float,
        volume: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        open_: float = 0.0,
        hour: int = -1,
        dow: int = -1,
        funding_rate: Optional[float] = None,
        trades: float = 0.0,
        taker_buy_volume: float = 0.0,
        quote_volume: float = 0.0,
        taker_buy_quote_volume: float = 0.0,
        open_interest: Optional[float] = None,
        ls_ratio: Optional[float] = None,
        top_trader_ls_ratio: Optional[float] = None,
        eth_close: Optional[float] = None,
        spot_close: Optional[float] = None,
        fear_greed: Optional[float] = None,
        implied_vol: Optional[float] = None,
        put_call_ratio: Optional[float] = None,
        onchain_metrics: Optional[Dict[str, float]] = None,
        liquidation_metrics: Optional[Dict[str, float]] = None,
        mempool_metrics: Optional[Dict[str, float]] = None,
        macro_metrics: Optional[Dict[str, float]] = None,
        sentiment_metrics: Optional[Dict[str, float]] = None,
        btc_close: Optional[float] = None,
        reference_closes: Optional[Dict[str, float]] = None,
        dvol: Optional[float] = None,
        options_metrics: Optional[Dict[str, float]] = None,
        cross_market: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Optional[float]]:
        """Process a new bar and return computed features.

        btc_close: BTC price at same bar time. Required for V12 ALT cross-asset features.
        top_trader_ls_ratio: Top trader position L/S ratio (V13).
        eth_close: ETH price at same bar time. Required for V14 BTC dominance features.
        reference_closes: Optional cross-asset reference close map for V14b dominance pairs.
        dvol: Deribit DVOL index value at same bar time. Required for V19 IV features.
        options_metrics: Dict from OptionsFlowComputer.
        cross_market: Dict with keys like spy_ret_1d, qqq_ret_1d, vix_level etc (from Yahoo Finance daily).
        """
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()

        # Default open to close if not provided
        if open_ == 0.0:
            open_ = close
        if high == 0.0:
            high = close
        if low == 0.0:
            low = close

        state = self._states[symbol]
        multi_dom_ratios = _build_multi_dominance_ratios(symbol, close, reference_closes)
        state.push(close, volume, high, low, open_,
                   hour=hour, dow=dow, funding_rate=funding_rate,
                   trades=trades, taker_buy_volume=taker_buy_volume,
                   quote_volume=quote_volume,
                   taker_buy_quote_volume=taker_buy_quote_volume,
                   open_interest=open_interest, ls_ratio=ls_ratio,
                   top_trader_ls_ratio=top_trader_ls_ratio,
                   eth_close=eth_close,
                   spot_close=spot_close, fear_greed=fear_greed,
                   implied_vol=implied_vol, put_call_ratio=put_call_ratio,
                   onchain_metrics=onchain_metrics,
                   liquidation_metrics=liquidation_metrics,
                   mempool_metrics=mempool_metrics,
                   macro_metrics=macro_metrics,
                   sentiment_metrics=sentiment_metrics,
                   multi_dom_ratios=multi_dom_ratios,
                   dvol=dvol)
        feats = state.get_features(btc_close=btc_close)

        # --- V20: Options flow features (from OptionsFlowComputer) ---
        if options_metrics:
            for key in ("gamma_imbalance_zscore", "max_pain_distance",
                        "vega_net_zscore", "iv_term_slope", "pcr_zscore",
                        "iv_rv_premium", "dvol_zscore"):
                feats[key] = options_metrics.get(key)

        # --- V21: Cross-market features (from Yahoo Finance daily data) ---
        if cross_market:
            for key in ("spy_ret_1d", "qqq_ret_1d", "spy_ret_5d",
                        "vix_level", "tlt_ret_5d", "uso_ret_5d",
                        "coin_ret_1d", "spy_extreme"):
                feats[key] = cross_market.get(key)

        return feats

    def get_features_dict(self, symbol: str) -> Dict[str, Optional[float]]:
        """Get last computed features as a flat dict (for signal models)."""
        if symbol not in self._states:
            return {}
        return self._states[symbol].get_features()

    @property
    def symbols(self) -> List[str]:
        return list(self._states.keys())

    def reset(self, symbol: Optional[str] = None) -> None:
        if symbol is not None:
            self._states.pop(symbol, None)
        else:
            self._states.clear()



# _symbol_aliases, _resolve_multi_dominance_pairs, _build_multi_dominance_ratios,
# _lookup_reference_close, _window_zscore -> features.enriched_trackers

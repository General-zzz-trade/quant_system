# decision/regime_bridge.py
"""RegimeAwareDecisionModule — wraps a DecisionModule with regime gating.

NOTE: This module IS used in the production path. LiveRunner, BacktestRunner, and
LivePaperRunner all wrap inner decision modules with RegimeAwareDecisionModule to
apply regime-based gating before order generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from _quant_hotpath import RustRegimeBuffer
from regime.base import RegimeDetector, RegimeLabel
from regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from regime.param_router import RegimeParamRouter, RegimeParams
from decision.market_access import get_float_attr
from decision.regime_policy import RegimePolicy

logger = logging.getLogger(__name__)


def _make_buffer(maxlen: int = 100) -> RustRegimeBuffer:
    return RustRegimeBuffer(maxlen)


@dataclass
class RegimeAwareDecisionModule:
    """DecisionModule that gates an inner module with regime detection.

    Flow:
    1. Extract features from snapshot (or fallback to price buffers)
    2. Run regime detectors to get labels
    3. RegimePolicy decides allow/block
    4. Optionally route regime params via ParamRouter
    5. If allowed, delegate to inner DecisionModule.decide()
    """

    inner: Any  # DecisionModule (has .decide(snapshot))
    detectors: Sequence[RegimeDetector] = field(default_factory=lambda: [
        CompositeRegimeDetector(),
    ])
    policy: RegimePolicy = field(default_factory=RegimePolicy)
    param_router: RegimeParamRouter = field(default_factory=RegimeParamRouter)
    enable_param_routing: bool = False
    composite_regime_symbols: tuple[str, ...] = field(default_factory=tuple)

    ma_fast_window: int = 10
    ma_slow_window: int = 30
    vol_window: int = 20
    buffer_maxlen: int = 100

    _buffers: Dict[str, RustRegimeBuffer] = field(default_factory=dict, init=False)
    _last_labels: List[RegimeLabel] = field(default_factory=list, init=False)
    _current_params: Optional[RegimeParams] = field(default=None, init=False)
    _composite_symbols: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._composite_symbols = set(self.composite_regime_symbols)

    def _is_composite_symbol(self, symbol: str) -> bool:
        """Return True if this symbol is designated to use CompositeRegime+ParamRouter."""
        return symbol in self._composite_symbols

    def _should_route_params(self, symbol: str) -> bool:
        """Return True if param routing should apply for this symbol.

        Param routing is active only when:
        - enable_param_routing is True globally, OR
        - the symbol is explicitly in composite_regime_symbols.
        """
        if self.enable_param_routing:
            return True
        return symbol in self._composite_symbols

    def _extract_features_from_snapshot(self, snapshot: Any, sym: str) -> Optional[Dict[str, Any]]:
        """Try to extract features from snapshot for the given symbol.

        Features come from RustFeatureEngine via the pipeline and are stored
        in snapshot.features (a dict keyed by feature name).
        """
        features = getattr(snapshot, "features", None)
        if features is not None and isinstance(features, dict) and len(features) > 0:
            return dict(features)
        return None

    def _compute_features_from_buffer(self, sym: str) -> Dict[str, Any]:
        """Fallback: compute basic features from RustRegimeBuffer."""
        buf = self._buffers.get(sym)
        if buf is None:
            return {}
        features: Dict[str, Any] = {}
        vol = buf.rolling_vol(self.vol_window)
        if vol is not None:
            features["vol"] = vol
        ma_fast = buf.ma(self.ma_fast_window)
        ma_slow = buf.ma(self.ma_slow_window)
        if ma_fast is not None:
            features["ma_fast"] = ma_fast
        if ma_slow is not None:
            features["ma_slow"] = ma_slow
        return features

    def decide(self, snapshot: Any) -> Iterable[Any]:
        ts = getattr(snapshot, "ts", None)
        markets = getattr(snapshot, "markets", {})

        # Update price buffers from snapshot (always, for fallback)
        for sym, mkt in markets.items():
            price = get_float_attr(mkt, "close", "last_price")
            if price is None:
                continue
            if sym not in self._buffers:
                self._buffers[sym] = _make_buffer(self.buffer_maxlen)
            self._buffers[sym].push(price)

        # Compute features and detect regimes
        all_labels: List[RegimeLabel] = []
        for sym in markets:
            if ts is None:
                continue

            # Prefer snapshot features (from RustFeatureEngine), fallback to buffer
            features = self._extract_features_from_snapshot(snapshot, sym)
            if features is None:
                features = self._compute_features_from_buffer(sym)

            if not features:
                continue

            for det in self.detectors:
                label = det.detect(symbol=sym, ts=ts, features=features)
                if label is not None:
                    all_labels.append(label)

        self._last_labels = all_labels

        # Param routing — enabled globally OR for any composite symbol in this snapshot
        _any_composite_sym = bool(self._composite_symbols) and any(
            sym in self._composite_symbols for sym in markets
        )
        if (self.enable_param_routing or _any_composite_sym) and all_labels:
            self._route_params(all_labels)

        # Policy gate
        allowed, reason = self.policy.allow(all_labels)
        if not allowed:
            logger.info(
                "Regime gate blocked: %s (labels: %s)",
                reason, [(lb.name, lb.value) for lb in all_labels],
            )
            return ()

        # Delegate to inner module
        result: Iterable[Any] = self.inner.decide(snapshot)
        return result

    def _route_params(self, labels: List[RegimeLabel]) -> None:
        """Extract CompositeRegimeLabel from labels and route to params."""
        for label in labels:
            if label.name != "composite" or label.meta is None:
                continue
            composite = label.meta.get("composite")
            if not isinstance(composite, CompositeRegimeLabel):
                continue
            new_params = self.param_router.route(composite)
            if self._current_params != new_params:
                logger.info(
                    "Regime params updated: %s → deadzone=%.2f min_hold=%d "
                    "max_hold=%d scale=%.2f (regime=%s|%s)",
                    "initial" if self._current_params is None else "changed",
                    new_params.deadzone, new_params.min_hold,
                    new_params.max_hold, new_params.position_scale,
                    composite.trend, composite.vol,
                )
            self._current_params = new_params
            break

    @property
    def current_labels(self) -> List[RegimeLabel]:
        return list(self._last_labels)

    @property
    def current_regime_params(self) -> Optional[RegimeParams]:
        return self._current_params

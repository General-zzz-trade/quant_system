"""Dominance feature methods for FeatureComputeHook."""
from __future__ import annotations

import logging
from typing import Any, Dict

_log = logging.getLogger(__name__)


def base_symbol(symbol: str) -> str:
    """Strip timeframe suffix to get base symbol: 'BTCUSDT_4h' -> 'BTCUSDT'."""
    for suffix in ("_4h", "_15m", "_1d"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


class DominanceMixin:
    """Mixin providing V14 dominance feature computation for feature hooks.

    Expects the following instance attributes from the host class:
        _rust_engines: Dict[str, Any]
        _dom_engines: Dict[str, Any]

    Also uses the module-level _last_closes dict from feature_hook.
    """

    _rust_engines: Dict[str, Any]
    _dom_engines: Dict[str, Any]

    def _push_dominance(self, symbol: str, close_f: float,
                        features: Dict[str, Any]) -> None:
        """Compute V14 dominance features and merge into *features* dict in-place.

        Uses the module-level ``_last_closes`` cache to obtain the counterpart
        symbol's close (BTC needs ETH close and vice-versa).  Only called when
        both prices are available; never crashes the pipeline.

        For the legacy path, reuses the existing RustFeatureEngine in
        ``_rust_engines``.  For the unified-predictor path (no entry in
        ``_rust_engines``), a lightweight engine is lazily created in
        ``_dom_engines`` -- push_dominance state is independent from push_bar.
        """
        try:
            from engine.feature_hook import _last_closes, _RustFeatureEngine

            base = base_symbol(symbol)
            # Update module-level close cache for this base symbol
            _last_closes[base] = close_f

            btc_close = _last_closes.get("BTCUSDT")
            eth_close = _last_closes.get("ETHUSDT")
            if btc_close is None or eth_close is None:
                return  # counterpart not yet available -- skip silently

            # Prefer main engine; fall back to dedicated dominance engine
            engine = self._rust_engines.get(symbol)
            if engine is None:
                if symbol not in self._dom_engines:
                    self._dom_engines[symbol] = _RustFeatureEngine()
                engine = self._dom_engines[symbol]

            dom_feats = engine.push_dominance(btc_close, eth_close)
            if dom_feats:
                for k, v in dom_feats.items():
                    if v is not None:
                        features[k] = v
        except Exception:
            _log.debug("push_dominance failed for %s -- skipped", symbol, exc_info=True)

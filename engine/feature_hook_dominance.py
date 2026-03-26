"""Dominance feature methods for FeatureComputeHook."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# Alias mapping: Rust push_dominance() dict keys → model feature names
_DOM_ALIASES: Dict[str, str] = {
    "btc_dom_ratio_dev_20": "btc_dom_dev_20",
    "btc_dom_return_diff_24h": "btc_dom_ret_24",
}

# Multi-ratio dominance pairs: symbol → [(ref_symbol, feature_prefix), ...]
# Mirrors batch_features_extra._DOMINANCE_PAIRS
_DOMINANCE_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "BTCUSDT": [("SUIUSDT", "dom_vs_sui")],
    "ETHUSDT": [("SUIUSDT", "dom_vs_sui"), ("AXSUSDT", "dom_vs_axs")],
    "SUIUSDT": [("AXSUSDT", "dom_vs_axs")],
    "AXSUSDT": [("ETHUSDT", "dom_vs_eth")],
}

# CSV close-price cache: ref_symbol → latest close (loaded once, updated if live data arrives)
_ref_close_cache: Dict[str, Optional[float]] = {}
_ref_cache_loaded: bool = False


def _load_ref_close(ref_symbol: str) -> Optional[float]:
    """Load latest close price for a reference symbol from CSV.

    Returns None if file missing or empty.
    """
    path = Path(f"data_files/{ref_symbol}_1h.csv")
    if not path.exists():
        return None
    try:
        import csv
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            last_row = None
            # Read last row without loading entire file into memory
            for last_row in reader:
                pass
            if last_row is not None and "close" in last_row:
                return float(last_row["close"])
    except Exception:
        _log.debug("Failed to load ref close for %s", ref_symbol, exc_info=True)
    return None


def _ensure_ref_cache() -> None:
    """Load reference close prices from CSV on first call."""
    global _ref_cache_loaded
    if _ref_cache_loaded:
        return
    _ref_cache_loaded = True
    # Collect all unique reference symbols
    all_refs = set()
    for pairs in _DOMINANCE_PAIRS.values():
        for ref_sym, _ in pairs:
            all_refs.add(ref_sym)
    for ref_sym in all_refs:
        _ref_close_cache[ref_sym] = _load_ref_close(ref_sym)
        if _ref_close_cache[ref_sym] is not None:
            _log.info("Loaded ref close for %s: %.4f", ref_sym, _ref_close_cache[ref_sym])


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

    def _get_or_create_dom_engine(self, key: str) -> Any:
        """Get or create a dedicated RustFeatureEngine for dominance computation."""
        if key not in self._dom_engines:
            from engine.feature_hook import _RustFeatureEngine
            self._dom_engines[key] = _RustFeatureEngine()
        return self._dom_engines[key]

    def _push_dominance(self, symbol: str, close_f: float,
                        features: Dict[str, Any]) -> None:
        """Compute V14 dominance features and merge into *features* dict in-place.

        Computes:
        1. BTC/ETH dominance: btc_dom_dev_20, btc_dom_ret_24 (+ mom_10, diff_6h)
        2. Multi-ratio dominance: dom_vs_sui_dev_20, dom_vs_axs_dev_20, etc.

        Uses the module-level ``_last_closes`` cache to obtain the counterpart
        symbol's close (BTC needs ETH close and vice-versa).  Only called when
        both prices are available; never crashes the pipeline.

        For the legacy path, reuses the existing RustFeatureEngine in
        ``_rust_engines``.  For the unified-predictor path (no entry in
        ``_rust_engines``), a lightweight engine is lazily created in
        ``_dom_engines`` -- push_dominance state is independent from push_bar.
        """
        try:
            from engine.feature_hook import _last_closes

            base = base_symbol(symbol)
            # Update module-level close cache for this base symbol
            _last_closes[base] = close_f

            # ── 1. Original BTC/ETH dominance ──
            btc_close = _last_closes.get("BTCUSDT")
            eth_close = _last_closes.get("ETHUSDT")
            if btc_close is not None and eth_close is not None:
                # Prefer main engine; fall back to dedicated dominance engine
                engine = self._rust_engines.get(symbol)
                if engine is None:
                    engine = self._get_or_create_dom_engine(symbol)

                dom_feats = engine.push_dominance(btc_close, eth_close)
                if dom_feats:
                    for k, v in dom_feats.items():
                        if v is not None:
                            # Apply alias mapping (Rust names → model names)
                            model_key = _DOM_ALIASES.get(k, k)
                            features[model_key] = v

            # ── 2. Multi-ratio dominance (dom_vs_sui, dom_vs_axs, etc.) ──
            _ensure_ref_cache()
            pairs = _DOMINANCE_PAIRS.get(base, [])
            for ref_sym, prefix in pairs:
                try:
                    # Get reference close: prefer live close from _last_closes, fall back to CSV
                    ref_base = base_symbol(ref_sym)
                    ref_close = _last_closes.get(ref_base) or _ref_close_cache.get(ref_sym)
                    if ref_close is None or ref_close <= 0:
                        continue

                    # Use dedicated engine per (symbol, prefix) pair
                    dom_key = f"{symbol}__dom__{prefix}"
                    engine = self._get_or_create_dom_engine(dom_key)

                    # push_dominance(btc_close=symbol_close, eth_close=ref_close)
                    # This computes ratio = symbol_close / ref_close and derived features
                    dom_feats = engine.push_dominance(close_f, ref_close)
                    if dom_feats:
                        # Map: btc_dom_ratio_dev_20 → {prefix}_dev_20
                        dev_20 = dom_feats.get("btc_dom_ratio_dev_20")
                        if dev_20 is not None:
                            features[f"{prefix}_dev_20"] = dev_20
                        # Map: btc_dom_return_diff_24h → {prefix}_ret_24
                        ret_24 = dom_feats.get("btc_dom_return_diff_24h")
                        if ret_24 is not None:
                            features[f"{prefix}_ret_24"] = ret_24
                except Exception:
                    _log.debug("Multi-ratio dominance failed for %s/%s -- skipped",
                               symbol, prefix, exc_info=True)

        except Exception:
            _log.debug("push_dominance failed for %s -- skipped", symbol, exc_info=True)

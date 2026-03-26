"""FeatureComputeHook — computes features from market events for the pipeline."""
from __future__ import annotations

import logging
import math
import time as _time_mod
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from _quant_hotpath import RustFeatureEngine as _RustFeatureEngine
from engine.feature_hook_dominance import DominanceMixin
from engine.feature_hook_nan import NanTrackingMixin
from event.types import EventType as _EventType

logger = logging.getLogger(__name__)
_log = logger

# Module-level close price cache shared across hook instances for V14 dominance.
_last_closes: Dict[str, float] = {}


class FeatureComputeHook(NanTrackingMixin, DominanceMixin):
    """Bridges RustFeatureEngine into the engine pipeline.

    Computes features from market events, injects into PipelineInput.features,
    and optionally runs ML models via LiveInferenceBridge.
    """

    def __init__(self, computer: Any,
                 inference_bridge: Union[Any, Dict[str, Any], None] = None,
                 warmup_bars: int = 65,
                 funding_rate_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 cross_asset_computer: Any = None,
                 oi_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 ls_ratio_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 spot_close_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 fgi_source: Any = None,
                 implied_vol_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 put_call_ratio_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 onchain_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 liquidation_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 mempool_source: Any = None,
                 macro_source: Any = None,
                 sentiment_source: Any = None,
                 unified_predictor: Any = None,
                 microstructure_source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None] = None,
                 cross_market_source: Optional[Callable[[], Dict[str, float]]] = None) -> None:
        self._computer = computer
        self._inference = inference_bridge
        self._unified = unified_predictor
        self._warmup_bars = warmup_bars
        self._funding_rate_source = funding_rate_source
        self._cross_asset = cross_asset_computer
        self._oi_source = oi_source
        self._ls_ratio_source = ls_ratio_source
        self._spot_close_source = spot_close_source
        self._fgi_source = fgi_source
        self._implied_vol_source = implied_vol_source
        self._put_call_ratio_source = put_call_ratio_source
        self._onchain_source = onchain_source
        self._liquidation_source = liquidation_source
        self._mempool_source = mempool_source
        self._macro_source = macro_source
        self._sentiment_source = sentiment_source
        self._microstructure_source = microstructure_source
        self._cross_market_source = cross_market_source
        self._last_features: Dict[str, Dict[str, Any]] = {}
        self._bar_count: Dict[str, int] = {}
        self._rust_engines: Dict[str, Any] = {}
        self._dom_engines: Dict[str, Any] = {}   # dedicated engines for push_dominance (unified path)
        # External data cache for unified predictor path
        self._ext_cache: Dict[str, Dict[str, Any]] = {}
        self._ext_push_count: Dict[str, int] = {}
        # NaN rate tracking (Task A)
        self._nan_counts: Dict[str, Dict[str, int]] = {}  # symbol -> feature -> count
        self._nan_bar_counts: Dict[str, int] = {}  # symbol -> total bars tracked
        self._nan_recent: Dict[str, Dict[str, int]] = {}  # symbol -> feature -> count in last window
        self._nan_recent_bars: Dict[str, int] = {}  # symbol -> bars in recent window
        self._NAN_WARN_WINDOW = 100
        self._NAN_WARN_RATE = 0.05

        # Schema validation: warn if model requires features the computer doesn't provide
        if inference_bridge is not None and unified_predictor is None:
            self._validate_feature_schema(inference_bridge, computer)

    @staticmethod
    def _validate_feature_schema(inference_bridge: Any, computer: Any) -> None:
        required: set[str] = set()
        engine = getattr(inference_bridge, "_engine", None)
        models = getattr(engine, "_models", []) if engine else []
        for model in models:
            names = getattr(model, "feature_names", None)
            if names:
                required.update(names)
        if not required:
            return
        available = set(getattr(computer, "feature_names", []))
        if not available:
            return
        missing = required - available - {"close", "volume"}
        if missing:
            logger.warning(
                "Feature schema mismatch: model requires %d features but computer lacks: %s",
                len(required), sorted(missing),
            )

        # Validate model features against the centralized production catalog
        try:
            from features.feature_catalog import validate_model_features
            for model in models:
                names = getattr(model, "feature_names", None)
                if names:
                    model_name = getattr(model, "name", "") or ""
                    catalog_warnings = validate_model_features(names, model_name=model_name)
                    for w in catalog_warnings:
                        logger.warning(w)
        except Exception:
            pass  # catalog validation is advisory — never block

    @staticmethod
    def _resolve_source(source: Union[Callable[[], Any], Dict[str, Callable[[], Any]], None],
                        symbol: str) -> Optional[Callable[[], Any]]:
        """Resolve a data source: if dict, look up by symbol; if callable, return as-is."""
        if source is None:
            return None
        if isinstance(source, dict):
            return source.get(symbol)
        return source

    def _safe_call_source(self, source_fn: Callable[[], Any], source_name: str, symbol: str) -> Any:
        """Call a source callable safely. Returns None on any exception."""
        try:
            return source_fn()
        except Exception:
            _log.warning(
                "FeatureHook: %s source raised for %s, using NaN",
                source_name, symbol, exc_info=True,
            )
            return None

    def _resolve_unified(self, symbol: str) -> Optional[Any]:
        """Resolve unified predictor: if dict, look up by symbol; if single, return as-is."""
        if self._unified is None:
            return None
        if isinstance(self._unified, dict):
            return self._unified.get(symbol)
        return self._unified

    def _resolve_inference(self, symbol: str) -> Optional[Any]:
        """Resolve inference bridge: if dict, look up by symbol; if single, return as-is."""
        if self._inference is None:
            return None
        if isinstance(self._inference, dict):
            return self._inference.get(symbol)
        return self._inference

    @staticmethod
    def _to_float_or_nan(val: Any) -> float:
        """Convert value to float, returning NaN for None/missing. Preserves 0.0."""
        if val is None:
            return math.nan
        try:
            f = float(val)
            return f  # 0.0 is valid, not NaN
        except (TypeError, ValueError):
            return math.nan

    def _resolve_bar_sources(self, symbol: str, event: Any) -> Dict[str, Any]:
        """Resolve all external data sources into keyword args for push_bar."""
        NaN = math.nan
        _f = self._to_float_or_nan
        hour, dow = -1, -1
        ts = getattr(event, "ts", None)
        if isinstance(ts, datetime):
            hour, dow = ts.hour, ts.weekday()

        funding_rate = NaN
        _funding_src = self._resolve_source(self._funding_rate_source, symbol)
        if _funding_src is not None:
            rate = self._safe_call_source(_funding_src, "funding_rate", symbol)
            if rate is not None:
                funding_rate = rate

        trades = float(getattr(event, "trades", 0) or 0)
        taker_buy_volume = float(getattr(event, "taker_buy_volume", 0) or 0)
        quote_volume = float(getattr(event, "quote_volume", 0) or 0)
        taker_buy_quote_volume = float(getattr(event, "taker_buy_quote_volume", 0) or 0)

        open_interest = NaN
        ls_ratio = NaN
        _oi_src = self._resolve_source(self._oi_source, symbol)
        if _oi_src is not None:
            oi_val = self._safe_call_source(_oi_src, "open_interest", symbol)
            if oi_val is not None:
                open_interest = oi_val
        _ls_src = self._resolve_source(self._ls_ratio_source, symbol)
        if _ls_src is not None:
            ls_val = self._safe_call_source(_ls_src, "ls_ratio", symbol)
            if ls_val is not None:
                ls_ratio = ls_val

        spot_close = NaN
        _spot_src = self._resolve_source(self._spot_close_source, symbol)
        if _spot_src is not None:
            spot_val = self._safe_call_source(_spot_src, "spot_close", symbol)
            if spot_val is not None:
                spot_close = spot_val

        fear_greed = NaN
        if self._fgi_source is not None:
            fgi_val = self._safe_call_source(self._fgi_source, "fear_greed", symbol)
            if fgi_val is not None:
                fear_greed = fgi_val

        implied_vol = NaN
        put_call_ratio = NaN
        _iv_src = self._resolve_source(self._implied_vol_source, symbol)
        if _iv_src is not None:
            iv_val = self._safe_call_source(_iv_src, "implied_vol", symbol)
            if iv_val is not None:
                implied_vol = iv_val
        _pcr_src = self._resolve_source(self._put_call_ratio_source, symbol)
        if _pcr_src is not None:
            pcr_val = self._safe_call_source(_pcr_src, "put_call_ratio", symbol)
            if pcr_val is not None:
                put_call_ratio = pcr_val

        oc_flow_in = oc_flow_out = oc_supply = oc_addr = oc_tx = oc_hashrate = NaN
        _onchain_src = self._resolve_source(self._onchain_source, symbol)
        if _onchain_src is not None:
            oc = self._safe_call_source(_onchain_src, "onchain", symbol)
            if oc is not None:
                oc_flow_in = _f(oc.get("FlowInExUSD"))
                oc_flow_out = _f(oc.get("FlowOutExUSD"))
                oc_supply = _f(oc.get("SplyExNtv"))
                oc_addr = _f(oc.get("AdrActCnt"))
                oc_tx = _f(oc.get("TxTfrCnt"))
                oc_hashrate = _f(oc.get("HashRate"))

        liq_total_vol = liq_buy_vol = liq_sell_vol = liq_count = NaN
        _liq_src = self._resolve_source(self._liquidation_source, symbol)
        if _liq_src is not None:
            liq = self._safe_call_source(_liq_src, "liquidation", symbol)
            if liq is not None:
                liq_total_vol = _f(liq.get("liq_total_volume"))
                liq_buy_vol = _f(liq.get("liq_buy_volume"))
                liq_sell_vol = _f(liq.get("liq_sell_volume"))
                liq_count = _f(liq.get("liq_count"))

        mempool_fastest = mempool_economy = mempool_size = NaN
        if self._mempool_source is not None:
            mp = self._safe_call_source(self._mempool_source, "mempool", symbol)
            if mp is not None:
                mempool_fastest = _f(mp.get("fastest_fee"))
                mempool_economy = _f(mp.get("economy_fee"))
                mempool_size = _f(mp.get("mempool_size"))

        macro_dxy = macro_spx = macro_vix = NaN
        macro_day = -1
        if self._macro_source is not None:
            macro = self._safe_call_source(self._macro_source, "macro", symbol)
            if macro is not None:
                macro_dxy = _f(macro.get("dxy"))
                macro_spx = _f(macro.get("spx"))
                macro_vix = _f(macro.get("vix"))
                date_str = macro.get("date")
                if date_str:
                    try:
                        d = datetime.strptime(date_str, "%Y-%m-%d")
                        macro_day = int(d.timestamp() / 86400)
                    except (ValueError, TypeError):
                        pass

        social_volume = sentiment_score = NaN
        if self._sentiment_source is not None:
            sent = self._safe_call_source(self._sentiment_source, "sentiment", symbol)
            if sent is not None:
                social_volume = _f(sent.get("social_volume"))
                sentiment_score = _f(sent.get("sentiment_score"))

        return dict(hour=hour, dow=dow, funding_rate=funding_rate,
                    trades=trades, taker_buy_volume=taker_buy_volume,
                    quote_volume=quote_volume, taker_buy_quote_volume=taker_buy_quote_volume,
                    open_interest=open_interest, ls_ratio=ls_ratio,
                    spot_close=spot_close, fear_greed=fear_greed,
                    implied_vol=implied_vol, put_call_ratio=put_call_ratio,
                    oc_flow_in=oc_flow_in, oc_flow_out=oc_flow_out,
                    oc_supply=oc_supply, oc_addr=oc_addr, oc_tx=oc_tx, oc_hashrate=oc_hashrate,
                    liq_total_vol=liq_total_vol, liq_buy_vol=liq_buy_vol,
                    liq_sell_vol=liq_sell_vol, liq_count=liq_count,
                    mempool_fastest_fee=mempool_fastest, mempool_economy_fee=mempool_economy,
                    mempool_size=mempool_size, macro_dxy=macro_dxy, macro_spx=macro_spx,
                    macro_vix=macro_vix, macro_day=macro_day,
                    social_volume=social_volume, sentiment_score=sentiment_score)

    # ── Public API for Coordinator fast path ──

    def get_bar_count(self, symbol: str) -> int:
        """Return current bar count for a symbol."""
        return self._bar_count.get(symbol, 0)

    def increment_bar_count(self, symbol: str) -> int:
        """Increment and return the new bar count for a symbol."""
        count = self._bar_count.get(symbol, 0) + 1
        self._bar_count[symbol] = count
        return count

    @property
    def warmup_bars(self) -> int:
        """Return the warmup bar threshold."""
        return self._warmup_bars

    @property
    def cross_asset(self) -> Any:
        """Return the cross-asset computer (or None)."""
        return self._cross_asset

    def set_last_features(self, symbol: str, features: Dict[str, Any]) -> None:
        """Store the latest features dict for a symbol."""
        self._last_features[symbol] = features

    # NaN tracking methods (_track_nan, nan_report, reset_nan_stats) provided by NanTrackingMixin

    def push_external_data(self, symbol: str, event: Any, target: Any) -> None:
        """Resolve external data sources and push to target (tick processor).

        Full resolution every 5 bars; only per-bar fields updated otherwise.
        This encapsulates the _ext_cache / _ext_push_count / _resolve_bar_sources
        interaction so callers never touch internal state directly.
        """
        ext_count = self._ext_push_count.get(symbol, 0)
        if ext_count % 5 == 0:
            src = self._resolve_bar_sources(symbol, event)
            self._ext_cache[symbol] = src
            target.push_external_data(symbol, **src)
        else:
            cached = self._ext_cache.get(symbol)
            if cached is not None:
                trades = float(getattr(event, "trades", 0) or 0)
                taker_buy_volume = float(getattr(event, "taker_buy_volume", 0) or 0)
                quote_volume = float(getattr(event, "quote_volume", 0) or 0)
                taker_buy_quote_volume = float(getattr(event, "taker_buy_quote_volume", 0) or 0)
                ts = getattr(event, "ts", None)
                cached["hour"] = ts.hour if isinstance(ts, datetime) else -1
                cached["dow"] = ts.weekday() if isinstance(ts, datetime) else -1
                cached["trades"] = trades
                cached["taker_buy_volume"] = taker_buy_volume
                cached["quote_volume"] = quote_volume
                cached["taker_buy_quote_volume"] = taker_buy_quote_volume
                target.push_external_data(symbol, **cached)
            else:
                src = self._resolve_bar_sources(symbol, event)
                self._ext_cache[symbol] = src
                target.push_external_data(symbol, **src)
        self._ext_push_count[symbol] = ext_count + 1

    def _merge_microstructure(self, symbol: str, features: Dict[str, Any]) -> None:
        """Merge microstructure features (VPIN, OB imbalance, spread, depth ratio) into features dict.

        Reads from the optional microstructure_source. If unavailable or erroring,
        features are simply not added (graceful degradation). Never crashes the pipeline.
        """
        if self._microstructure_source is None:
            return
        try:
            _ms_src = self._resolve_source(self._microstructure_source, symbol)
            if _ms_src is None:
                return
            ms_data = self._safe_call_source(_ms_src, "microstructure", symbol)
            if ms_data is None:
                return
            # Merge known microstructure keys into features dict
            for key in ("vpin", "ob_imbalance", "spread_bps", "depth_ratio"):
                val = ms_data.get(key)
                if val is not None:
                    features[key] = float(val)
        except Exception:
            _log.debug("Microstructure merge failed for %s — skipped", symbol, exc_info=True)

    # _push_dominance provided by DominanceMixin

    def _rust_push(self, symbol: str, close_f: float, volume: float,
                   high: float, low: float, open_: float, event: Any) -> Dict[str, Any]:
        """Push bar to per-symbol RustFeatureEngine and return features dict."""
        if symbol not in self._rust_engines:
            self._rust_engines[symbol] = _RustFeatureEngine()

        engine = self._rust_engines[symbol]
        src = self._resolve_bar_sources(symbol, event)

        engine.push_bar(
            close=close_f, volume=volume, high=high, low=low, open=open_,
            **src,
        )

        # Push cross-market ETF data (SPY, TLT, USO, etc.)
        if self._cross_market_source is not None:
            cm = self._safe_call_source(self._cross_market_source, "cross_market", symbol)
            if cm is not None:
                engine.push_cross_market(**cm)

        features = engine.get_features()
        out = {k: v for k, v in features.items() if v is not None}

        # Alias mapping: Rust feature names → model config feature names
        # Rust engine uses generic names; models expect specific ETF names
        _ALIASES = {
            "tnx_change_5d": "treasury_10y_chg_5d",
            "etf_premium": "ethe_ret_1d",
            "ibit_flow_zscore": "gbtc_vol_zscore_14",
            "spy_vix_change": "spy_extreme",
        }
        for rust_name, model_name in _ALIASES.items():
            if rust_name in out and model_name not in out:
                out[model_name] = out[rust_name]

        # ETF 5d returns: Rust computes gold_ret_5d from xlf_close buffer.
        # Map it and compute missing ETF returns from cross_market source.
        if "gold_ret_5d" in out:
            for alias in ("xlf_ret_5d", "tlt_ret_5d", "uso_ret_5d"):
                if alias not in out:
                    out[alias] = out["gold_ret_5d"]  # approximate: same macro regime

        return out

    def on_event(self, event: Any) -> Optional[Mapping[str, Any]]:
        """Compute features if this is a market event. Returns features dict or None."""
        kind = getattr(event, "event_type", None)
        if kind is not None and kind is not _EventType.MARKET:
            # Fast path: direct enum comparison covers production events.
            # Fallback: substring check for test/custom event types like "market_data".
            kind_val = getattr(kind, "value", kind)
            if not isinstance(kind_val, str) or "market" not in kind_val.lower():
                sym = getattr(event, "symbol", None)
                if sym and sym in self._last_features:
                    return dict(self._last_features[sym])
                return None

        symbol = getattr(event, "symbol", None)
        close = getattr(event, "close", None)
        if symbol is None or close is None:
            return None

        try:
            close_f = float(close)
        except (TypeError, ValueError):
            return None

        volume = float(getattr(event, "volume", 0) or 0)
        high = float(getattr(event, "high", 0) or 0)
        low = float(getattr(event, "low", 0) or 0)
        open_ = float(getattr(event, "open", 0) or 0)

        count = self._bar_count.get(symbol, 0) + 1
        self._bar_count[symbol] = count
        if count < self._warmup_bars:
            if count == 1 or count % 10 == 0:
                logger.warning("Warmup %s: %d/%d bars (skipping inference)", symbol, count, self._warmup_bars)

        # ── Unified predictor path: single Rust call for features + predict + signal ──
        unified = self._resolve_unified(symbol)
        if unified is not None:
            features = self._unified_push(unified, symbol, close_f, volume, high, low, open_, event)
            features["close"] = close_f
            features["volume"] = volume

            if self._cross_asset is not None:
                funding_rate = features.get("funding_rate")
                self._cross_asset.on_bar(symbol, close=close_f, funding_rate=funding_rate,
                                         high=high, low=low)
                cross_feats = self._cross_asset.get_features(symbol)
                features.update(cross_feats)

            # V14 dominance features (BTC/ETH ratio deviation, momentum, return diff)
            self._push_dominance(symbol, close_f, features)

            # Microstructure features (VPIN, OB imbalance) — optional, live-only
            self._merge_microstructure(symbol, features)

            self._track_nan(symbol, features)
            self._last_features[symbol] = features
            return features

        # ── Legacy path: separate engine + bridge ──
        features = self._rust_push(symbol, close_f, volume, high, low, open_, event)
        features["close"] = close_f
        features["volume"] = volume

        if self._cross_asset is not None:
            funding_rate = features.get("funding_rate")
            self._cross_asset.on_bar(symbol, close=close_f, funding_rate=funding_rate,
                                     high=high, low=low)
            cross_feats = self._cross_asset.get_features(symbol)
            features.update(cross_feats)

        # V14 dominance features (BTC/ETH ratio deviation, momentum, return diff)
        self._push_dominance(symbol, close_f, features)

        bridge = self._resolve_inference(symbol)
        if bridge is not None and count >= self._warmup_bars:
            ts = getattr(event, "ts", None)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = None
            features = bridge.enrich(symbol, ts, features)

        # Microstructure features (VPIN, OB imbalance) — optional, live-only
        self._merge_microstructure(symbol, features)

        self._track_nan(symbol, features)
        self._last_features[symbol] = features
        return features

    def _unified_push(self, predictor: Any, symbol: str, close_f: float, volume: float,
                      high: float, low: float, open_: float, event: Any) -> Dict[str, Any]:
        """Push bar to unified predictor and return features dict with ml_score injected.

        Uses cached external data path via push_external_data(), then calls
        slim push_bar_predict() with only OHLCV.
        """
        self.push_external_data(symbol, event, predictor)

        # Compute hour_key for signal constraints
        ts = getattr(event, "ts", None)
        hour_key = int(ts.timestamp()) // 3600 if isinstance(ts, datetime) else int(_time_mod.time()) // 3600

        # Single Rust call: push bar + predict + get features (no None values)
        result: Tuple[Any, Dict[str, Any]] = predictor.push_bar_predict_features(
            symbol, close_f, volume, high, low, open_, hour_key,
        )
        prediction, features = result

        # Inject ML scores from unified prediction
        if self._bar_count.get(symbol, 0) >= self._warmup_bars:
            features["ml_score"] = prediction["ml_score"]
            if prediction.get("ml_short_score", 0.0) != 0.0:
                features["ml_short_score"] = prediction["ml_short_score"]

        return features

# engine/feature_hook.py
"""FeatureComputeHook — computes features from market events for the pipeline."""
from __future__ import annotations

import logging
import math
import time as _time_mod
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional, Union

logger = logging.getLogger(__name__)

from _quant_hotpath import RustFeatureEngine as _RustFeatureEngine
from event.types import EventType as _EventType


class FeatureComputeHook:
    """Bridges RustFeatureEngine into the engine pipeline.

    Called before pipeline.apply() to compute features from market events.
    Features are injected into PipelineInput.features and flow through
    to StateSnapshot.features for downstream ML signal consumption.

    Optionally integrates LiveInferenceBridge to run ML models and inject
    ml_score into the features dict.

    Uses per-symbol RustFeatureEngine instances for fast incremental
    feature computation.
    """

    def __init__(self, computer: Any,
                 inference_bridge: Union[Any, Dict[str, Any], None] = None,
                 warmup_bars: int = 65,
                 funding_rate_source: Union[Callable, Dict[str, Callable], None] = None,
                 cross_asset_computer: Any = None,
                 oi_source: Union[Callable, Dict[str, Callable], None] = None,
                 ls_ratio_source: Union[Callable, Dict[str, Callable], None] = None,
                 spot_close_source: Union[Callable, Dict[str, Callable], None] = None,
                 fgi_source: Any = None,
                 implied_vol_source: Union[Callable, Dict[str, Callable], None] = None,
                 put_call_ratio_source: Union[Callable, Dict[str, Callable], None] = None,
                 onchain_source: Union[Callable, Dict[str, Callable], None] = None,
                 liquidation_source: Union[Callable, Dict[str, Callable], None] = None,
                 mempool_source: Any = None,
                 macro_source: Any = None,
                 sentiment_source: Any = None,
                 unified_predictor: Any = None) -> None:
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
        self._last_features: Dict[str, Dict[str, Any]] = {}
        self._bar_count: Dict[str, int] = {}
        self._rust_engines: Dict[str, Any] = {}
        # External data cache for unified predictor path
        self._ext_cache: Dict[str, Dict[str, Any]] = {}
        self._ext_push_count: Dict[str, int] = {}

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

    @staticmethod
    def _resolve_source(source: Union[Callable, Dict[str, Callable], None],
                        symbol: str) -> Optional[Callable]:
        """Resolve a data source: if dict, look up by symbol; if callable, return as-is."""
        if source is None:
            return None
        if isinstance(source, dict):
            return source.get(symbol)
        return source

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

    def _resolve_bar_sources(self, symbol: str, event: Any) -> Dict[str, Any]:
        """Resolve all external data sources into keyword args for push_bar."""
        NaN = math.nan
        hour, dow = -1, -1
        ts = getattr(event, "ts", None)
        if isinstance(ts, datetime):
            hour, dow = ts.hour, ts.weekday()

        funding_rate = NaN
        _funding_src = self._resolve_source(self._funding_rate_source, symbol)
        if _funding_src is not None:
            rate = _funding_src()
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
            oi_val = _oi_src()
            if oi_val is not None:
                open_interest = oi_val
        _ls_src = self._resolve_source(self._ls_ratio_source, symbol)
        if _ls_src is not None:
            ls_val = _ls_src()
            if ls_val is not None:
                ls_ratio = ls_val

        spot_close = NaN
        _spot_src = self._resolve_source(self._spot_close_source, symbol)
        if _spot_src is not None:
            spot_val = _spot_src()
            if spot_val is not None:
                spot_close = spot_val

        fear_greed = NaN
        if self._fgi_source is not None:
            fgi_val = self._fgi_source()
            if fgi_val is not None:
                fear_greed = fgi_val

        implied_vol = NaN
        put_call_ratio = NaN
        _iv_src = self._resolve_source(self._implied_vol_source, symbol)
        if _iv_src is not None:
            iv_val = _iv_src()
            if iv_val is not None:
                implied_vol = iv_val
        _pcr_src = self._resolve_source(self._put_call_ratio_source, symbol)
        if _pcr_src is not None:
            pcr_val = _pcr_src()
            if pcr_val is not None:
                put_call_ratio = pcr_val

        oc_flow_in = oc_flow_out = oc_supply = oc_addr = oc_tx = oc_hashrate = NaN
        _onchain_src = self._resolve_source(self._onchain_source, symbol)
        if _onchain_src is not None:
            oc = _onchain_src()
            if oc is not None:
                oc_flow_in = oc.get("FlowInExUSD", NaN) or NaN
                oc_flow_out = oc.get("FlowOutExUSD", NaN) or NaN
                oc_supply = oc.get("SplyExNtv", NaN) or NaN
                oc_addr = oc.get("AdrActCnt", NaN) or NaN
                oc_tx = oc.get("TxTfrCnt", NaN) or NaN
                oc_hashrate = oc.get("HashRate", NaN) or NaN

        liq_total_vol = liq_buy_vol = liq_sell_vol = liq_count = NaN
        _liq_src = self._resolve_source(self._liquidation_source, symbol)
        if _liq_src is not None:
            liq = _liq_src()
            if liq is not None:
                liq_total_vol = liq.get("liq_total_volume", NaN) or NaN
                liq_buy_vol = liq.get("liq_buy_volume", NaN) or NaN
                liq_sell_vol = liq.get("liq_sell_volume", NaN) or NaN
                liq_count = liq.get("liq_count", NaN) or NaN

        mempool_fastest = mempool_economy = mempool_size = NaN
        if self._mempool_source is not None:
            mp = self._mempool_source()
            if mp is not None:
                mempool_fastest = mp.get("fastest_fee", NaN) or NaN
                mempool_economy = mp.get("economy_fee", NaN) or NaN
                mempool_size = mp.get("mempool_size", NaN) or NaN

        macro_dxy = macro_spx = macro_vix = NaN
        macro_day = -1
        if self._macro_source is not None:
            macro = self._macro_source()
            if macro is not None:
                macro_dxy = macro.get("dxy", NaN) or NaN
                macro_spx = macro.get("spx", NaN) or NaN
                macro_vix = macro.get("vix", NaN) or NaN
                date_str = macro.get("date")
                if date_str:
                    try:
                        d = datetime.strptime(date_str, "%Y-%m-%d")
                        macro_day = int(d.timestamp() / 86400)
                    except (ValueError, TypeError):
                        pass

        social_volume = sentiment_score = NaN
        if self._sentiment_source is not None:
            sent = self._sentiment_source()
            if sent is not None:
                social_volume = sent.get("social_volume", NaN) or NaN
                sentiment_score = sent.get("sentiment_score", NaN) or NaN

        return dict(
            hour=hour, dow=dow,
            funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume, taker_buy_quote_volume=taker_buy_quote_volume,
            open_interest=open_interest, ls_ratio=ls_ratio,
            spot_close=spot_close, fear_greed=fear_greed,
            implied_vol=implied_vol, put_call_ratio=put_call_ratio,
            oc_flow_in=oc_flow_in, oc_flow_out=oc_flow_out,
            oc_supply=oc_supply, oc_addr=oc_addr,
            oc_tx=oc_tx, oc_hashrate=oc_hashrate,
            liq_total_vol=liq_total_vol, liq_buy_vol=liq_buy_vol,
            liq_sell_vol=liq_sell_vol, liq_count=liq_count,
            mempool_fastest_fee=mempool_fastest, mempool_economy_fee=mempool_economy,
            mempool_size=mempool_size,
            macro_dxy=macro_dxy, macro_spx=macro_spx, macro_vix=macro_vix,
            macro_day=macro_day,
            social_volume=social_volume, sentiment_score=sentiment_score,
        )

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

        features = engine.get_features()
        return {k: v for k, v in features.items() if v is not None}

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

        bridge = self._resolve_inference(symbol)
        if bridge is not None and count >= self._warmup_bars:
            ts = getattr(event, "ts", None)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = None
            features = bridge.enrich(symbol, ts, features)

        self._last_features[symbol] = features
        return features

    def _unified_push(self, predictor: Any, symbol: str, close_f: float, volume: float,
                      high: float, low: float, open_: float, event: Any) -> Dict[str, Any]:
        """Push bar to unified predictor and return features dict with ml_score injected.

        Uses cached external data path: resolves sources once per change via
        push_external_data(), then calls slim push_bar_predict() with only OHLCV.
        """
        # Push external data: full resolve every 5 bars, event-fields only otherwise.
        # Most external sources (funding=8h, OI=5m, macro=1d) change far slower than bars.
        ext_count = self._ext_push_count.get(symbol, 0)
        if ext_count % 5 == 0:
            src = self._resolve_bar_sources(symbol, event)
            self._ext_cache[symbol] = src
            predictor.push_external_data(symbol, **src)
        else:
            # Update only per-bar fields (trades, taker_buy_volume, etc.) from event
            cached = self._ext_cache.get(symbol)
            if cached is not None:
                trades = float(getattr(event, "trades", 0) or 0)
                taker_buy_volume = float(getattr(event, "taker_buy_volume", 0) or 0)
                quote_volume = float(getattr(event, "quote_volume", 0) or 0)
                taker_buy_quote_volume = float(getattr(event, "taker_buy_quote_volume", 0) or 0)
                ts = getattr(event, "ts", None)
                hour = ts.hour if isinstance(ts, datetime) else -1
                dow = ts.weekday() if isinstance(ts, datetime) else -1
                cached["hour"] = hour
                cached["dow"] = dow
                cached["trades"] = trades
                cached["taker_buy_volume"] = taker_buy_volume
                cached["quote_volume"] = quote_volume
                cached["taker_buy_quote_volume"] = taker_buy_quote_volume
                predictor.push_external_data(symbol, **cached)
            else:
                src = self._resolve_bar_sources(symbol, event)
                self._ext_cache[symbol] = src
                predictor.push_external_data(symbol, **src)
        self._ext_push_count[symbol] = ext_count + 1

        # Compute hour_key for signal constraints
        ts = getattr(event, "ts", None)
        if isinstance(ts, datetime):
            hour_key = int(ts.timestamp()) // 3600
        else:
            hour_key = int(_time_mod.time()) // 3600

        # Single Rust call: push bar + predict + get features (no None values)
        prediction, features = predictor.push_bar_predict_features(
            symbol, close_f, volume, high, low, open_, hour_key,
        )

        # Inject ML scores from unified prediction
        if self._bar_count.get(symbol, 0) >= self._warmup_bars:
            features["ml_score"] = prediction["ml_score"]
            if prediction.get("ml_short_score", 0.0) != 0.0:
                features["ml_short_score"] = prediction["ml_short_score"]

        return features

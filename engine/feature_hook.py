# engine/feature_hook.py
"""FeatureComputeHook — computes features from market events for the pipeline."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

try:
    from _quant_hotpath import RustFeatureEngine as _RustFeatureEngine
    _HAS_RUST_FEATURES = True
except ImportError:
    _HAS_RUST_FEATURES = False


class FeatureComputeHook:
    """Bridges LiveFeatureComputer into the engine pipeline.

    Called before pipeline.apply() to compute features from market events.
    Features are injected into PipelineInput.features and flow through
    to StateSnapshot.features for downstream ML signal consumption.

    Optionally integrates LiveInferenceBridge to run ML models and inject
    ml_score into the features dict.

    When use_rust=True (default if available), uses per-symbol RustFeatureEngine
    instances for ~5-10x faster incremental feature computation.
    """

    def __init__(self, computer: Any, inference_bridge: Any = None,
                 warmup_bars: int = 65,
                 funding_rate_source: Any = None,
                 cross_asset_computer: Any = None,
                 oi_source: Any = None,
                 ls_ratio_source: Any = None,
                 spot_close_source: Any = None,
                 fgi_source: Any = None,
                 implied_vol_source: Any = None,
                 put_call_ratio_source: Any = None,
                 onchain_source: Any = None,
                 liquidation_source: Any = None,
                 mempool_source: Any = None,
                 macro_source: Any = None,
                 sentiment_source: Any = None,
                 use_rust: Optional[bool] = None) -> None:
        self._computer = computer
        self._inference = inference_bridge
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

        # Rust feature engine: per-symbol instances (opt-in via use_rust=True)
        self._use_rust = bool(use_rust) and _HAS_RUST_FEATURES
        self._rust_engines: Dict[str, Any] = {}

        if self._use_rust:
            logger.info("FeatureComputeHook: using RustFeatureEngine (fast path)")
        else:
            # Check once which extra params computer.on_bar accepts
            import inspect
            sig = inspect.signature(computer.on_bar)
            self._pass_open = "open_" in sig.parameters
            self._pass_hour = "hour" in sig.parameters
            self._pass_funding = "funding_rate" in sig.parameters
            self._pass_trades = "trades" in sig.parameters
            self._pass_oi = "open_interest" in sig.parameters
            self._pass_spot_close = "spot_close" in sig.parameters
            self._pass_fgi = "fear_greed" in sig.parameters
            self._pass_implied_vol = "implied_vol" in sig.parameters
            self._pass_onchain = "onchain_metrics" in sig.parameters
            self._pass_liquidation = "liquidation_metrics" in sig.parameters
            self._pass_mempool = "mempool_metrics" in sig.parameters
            self._pass_macro = "macro_metrics" in sig.parameters
            self._pass_sentiment = "sentiment_metrics" in sig.parameters

        # Schema validation: warn if model requires features the computer doesn't provide
        if inference_bridge is not None:
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

    def _rust_push(self, symbol: str, close_f: float, volume: float,
                   high: float, low: float, open_: float, event: Any) -> Dict[str, Any]:
        """Push bar to per-symbol RustFeatureEngine and return features dict."""
        if symbol not in self._rust_engines:
            self._rust_engines[symbol] = _RustFeatureEngine()

        engine = self._rust_engines[symbol]

        # Time features
        hour, dow = -1, -1
        ts = getattr(event, "ts", None)
        if isinstance(ts, datetime):
            hour, dow = ts.hour, ts.weekday()

        # Funding
        NaN = math.nan
        funding_rate = NaN
        if self._funding_rate_source is not None:
            rate = self._funding_rate_source()
            if rate is not None:
                funding_rate = rate

        # Microstructure
        trades = float(getattr(event, "trades", 0) or 0)
        taker_buy_volume = float(getattr(event, "taker_buy_volume", 0) or 0)
        quote_volume = float(getattr(event, "quote_volume", 0) or 0)
        taker_buy_quote_volume = float(getattr(event, "taker_buy_quote_volume", 0) or 0)

        # OI / LS ratio
        open_interest = NaN
        ls_ratio = NaN
        if self._oi_source is not None:
            oi_val = self._oi_source()
            if oi_val is not None:
                open_interest = oi_val
        if self._ls_ratio_source is not None:
            ls_val = self._ls_ratio_source()
            if ls_val is not None:
                ls_ratio = ls_val

        # Spot close
        spot_close = NaN
        if self._spot_close_source is not None:
            spot_val = self._spot_close_source()
            if spot_val is not None:
                spot_close = spot_val

        # Fear & Greed
        fear_greed = NaN
        if self._fgi_source is not None:
            fgi_val = self._fgi_source()
            if fgi_val is not None:
                fear_greed = fgi_val

        # Implied vol / put-call ratio
        implied_vol = NaN
        put_call_ratio = NaN
        if self._implied_vol_source is not None:
            iv_val = self._implied_vol_source()
            if iv_val is not None:
                implied_vol = iv_val
        if self._put_call_ratio_source is not None:
            pcr_val = self._put_call_ratio_source()
            if pcr_val is not None:
                put_call_ratio = pcr_val

        # On-chain (dict → flat args)
        oc_flow_in = oc_flow_out = oc_supply = oc_addr = oc_tx = oc_hashrate = NaN
        if self._onchain_source is not None:
            oc = self._onchain_source()
            if oc is not None:
                oc_flow_in = oc.get("FlowInExUSD", NaN) or NaN
                oc_flow_out = oc.get("FlowOutExUSD", NaN) or NaN
                oc_supply = oc.get("SplyExNtv", NaN) or NaN
                oc_addr = oc.get("AdrActCnt", NaN) or NaN
                oc_tx = oc.get("TxTfrCnt", NaN) or NaN
                oc_hashrate = oc.get("HashRate", NaN) or NaN

        # Liquidation (dict → flat args)
        liq_total_vol = liq_buy_vol = liq_sell_vol = liq_count = NaN
        if self._liquidation_source is not None:
            liq = self._liquidation_source()
            if liq is not None:
                liq_total_vol = liq.get("liq_total_volume", NaN) or NaN
                liq_buy_vol = liq.get("liq_buy_volume", NaN) or NaN
                liq_sell_vol = liq.get("liq_sell_volume", NaN) or NaN
                liq_count = liq.get("liq_count", NaN) or NaN

        # Mempool (dict → flat args)
        mempool_fastest = mempool_economy = mempool_size = NaN
        if self._mempool_source is not None:
            mp = self._mempool_source()
            if mp is not None:
                mempool_fastest = mp.get("fastest_fee", NaN) or NaN
                mempool_economy = mp.get("economy_fee", NaN) or NaN
                mempool_size = mp.get("mempool_size", NaN) or NaN

        # Macro (dict → flat args, date string → day integer)
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

        # Sentiment (dict → flat args)
        social_volume = sentiment_score = NaN
        if self._sentiment_source is not None:
            sent = self._sentiment_source()
            if sent is not None:
                social_volume = sent.get("social_volume", NaN) or NaN
                sentiment_score = sent.get("sentiment_score", NaN) or NaN

        engine.push_bar(
            close=close_f, volume=volume, high=high, low=low, open=open_,
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

        features = engine.get_features()
        # Replace None with NaN-absent (match Python computer behavior)
        return {k: v for k, v in features.items() if v is not None}

    def on_event(self, event: Any) -> Optional[Mapping[str, Any]]:
        """Compute features if this is a market event. Returns features dict or None."""
        kind = getattr(event, "event_type", None)
        if kind is not None:
            kind_val = getattr(kind, "value", kind)
            if "market" not in str(kind_val).lower():
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

        if self._use_rust:
            features = self._rust_push(symbol, close_f, volume, high, low, open_, event)
        else:
            bar_kwargs = {"close": close_f, "volume": volume, "high": high, "low": low}
            if self._pass_open:
                bar_kwargs["open_"] = open_

            if self._pass_hour:
                ts = getattr(event, "ts", None)
                if isinstance(ts, datetime):
                    bar_kwargs["hour"] = ts.hour
                    bar_kwargs["dow"] = ts.weekday()

            if self._pass_funding and self._funding_rate_source is not None:
                rate = self._funding_rate_source()
                if rate is not None:
                    bar_kwargs["funding_rate"] = rate

            if self._pass_trades:
                bar_kwargs["trades"] = float(getattr(event, "trades", 0) or 0)
                bar_kwargs["taker_buy_volume"] = float(getattr(event, "taker_buy_volume", 0) or 0)
                bar_kwargs["quote_volume"] = float(getattr(event, "quote_volume", 0) or 0)

            if self._pass_oi:
                if self._oi_source is not None:
                    oi_val = self._oi_source()
                    if oi_val is not None:
                        bar_kwargs["open_interest"] = oi_val
                if self._ls_ratio_source is not None:
                    ls_val = self._ls_ratio_source()
                    if ls_val is not None:
                        bar_kwargs["ls_ratio"] = ls_val

            if self._pass_spot_close and self._spot_close_source is not None:
                spot_val = self._spot_close_source()
                if spot_val is not None:
                    bar_kwargs["spot_close"] = spot_val

            if self._pass_fgi and self._fgi_source is not None:
                fgi_val = self._fgi_source()
                if fgi_val is not None:
                    bar_kwargs["fear_greed"] = fgi_val

            if self._pass_implied_vol:
                if self._implied_vol_source is not None:
                    iv_val = self._implied_vol_source()
                    if iv_val is not None:
                        bar_kwargs["implied_vol"] = iv_val
                if self._put_call_ratio_source is not None:
                    pcr_val = self._put_call_ratio_source()
                    if pcr_val is not None:
                        bar_kwargs["put_call_ratio"] = pcr_val

            if self._pass_onchain and self._onchain_source is not None:
                oc_val = self._onchain_source()
                if oc_val is not None:
                    bar_kwargs["onchain_metrics"] = oc_val

            if self._pass_liquidation and self._liquidation_source is not None:
                liq_val = self._liquidation_source()
                if liq_val is not None:
                    bar_kwargs["liquidation_metrics"] = liq_val

            if self._pass_mempool and self._mempool_source is not None:
                mp_val = self._mempool_source()
                if mp_val is not None:
                    bar_kwargs["mempool_metrics"] = mp_val

            if self._pass_macro and self._macro_source is not None:
                macro_val = self._macro_source()
                if macro_val is not None:
                    bar_kwargs["macro_metrics"] = macro_val

            if self._pass_sentiment and self._sentiment_source is not None:
                sent_val = self._sentiment_source()
                if sent_val is not None:
                    bar_kwargs["sentiment_metrics"] = sent_val

            self._computer.on_bar(symbol, **bar_kwargs)
            features = self._computer.get_features_dict(symbol)

        count = self._bar_count.get(symbol, 0) + 1
        self._bar_count[symbol] = count
        if count < self._warmup_bars:
            if count == 1 or count % 10 == 0:
                logger.warning("Warmup %s: %d/%d bars (skipping inference)", symbol, count, self._warmup_bars)

        features["close"] = close_f
        features["volume"] = volume

        if self._cross_asset is not None:
            funding_rate = features.get("funding_rate")
            self._cross_asset.on_bar(symbol, close=close_f, funding_rate=funding_rate,
                                     high=high, low=low)
            cross_feats = self._cross_asset.get_features(symbol)
            features.update(cross_feats)

        if self._inference is not None and self._bar_count.get(symbol, 0) >= self._warmup_bars:
            ts = getattr(event, "ts", None)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = None
            features = self._inference.enrich(symbol, ts, features)

        self._last_features[symbol] = features
        return features

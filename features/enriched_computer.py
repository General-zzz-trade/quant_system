# features/enriched_computer.py
"""EnrichedFeatureComputer — 30+ incremental features for ML alpha models.

Same interface as LiveFeatureComputer but computes a much richer feature set
including RSI, MACD, Bollinger Bands, ATR, multi-horizon returns, volume
profile, trend indicators, time-of-day, and crypto-native features.

All computations are O(1) per bar (incremental/EMA-based).
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from _quant_hotpath import RollingWindow, VWAPWindow

# VWAPWindow — Rust-accelerated volume-weighted average price window.
# Used for VWAP deviation features (vwap_dev_20) in the enriched feature set.
VWAPWindowType = VWAPWindow

from features.enriched_trackers import (  # noqa: E402
    _EMA,
    _RSITracker,
    _ATRTracker,
    _ADXTracker,
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


@dataclass
class _SymbolState:
    """Per-symbol state for enriched feature computation."""

    # Close history for multi-horizon returns
    close_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    open_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2))
    high_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2))
    low_history: Deque[float] = field(default_factory=lambda: deque(maxlen=2))

    # Moving averages
    ma_5: RollingWindow = field(default_factory=lambda: RollingWindow(5))
    ma_10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    ma_20: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    ma_30: RollingWindow = field(default_factory=lambda: RollingWindow(30))
    ma_50: RollingWindow = field(default_factory=lambda: RollingWindow(50))

    # Bollinger: 20-bar window
    bb_window: RollingWindow = field(default_factory=lambda: RollingWindow(20))

    # Return windows for volatility
    return_window_20: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    return_window_5: RollingWindow = field(default_factory=lambda: RollingWindow(5))

    # Volume windows
    vol_window_20: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    vol_window_5: RollingWindow = field(default_factory=lambda: RollingWindow(5))

    # RSI trackers
    rsi_14: _RSITracker = field(default_factory=lambda: _RSITracker(period=14))
    rsi_6: _RSITracker = field(default_factory=lambda: _RSITracker(period=6))

    # MACD EMAs
    ema_12: _EMA = field(default_factory=lambda: _EMA(span=12))
    ema_26: _EMA = field(default_factory=lambda: _EMA(span=26))
    macd_signal_ema: _EMA = field(default_factory=lambda: _EMA(span=9))

    # ATR tracker
    atr_14: _ATRTracker = field(default_factory=lambda: _ATRTracker(period=14))

    # ADX tracker (trend strength, needs 2*period = 28 bars warmup)
    adx_14: _ADXTracker = field(default_factory=lambda: _ADXTracker(period=14))

    # Funding rate EMA tracker (8 settlements = 64 hours)
    funding_ema: _EMA = field(default_factory=lambda: _EMA(span=8))

    # Microstructure: kline trade/taker fields
    trades_ema_20: _EMA = field(default_factory=lambda: _EMA(span=20))
    trades_ema_5: _EMA = field(default_factory=lambda: _EMA(span=5))
    taker_buy_ratio_ema_10: _EMA = field(default_factory=lambda: _EMA(span=10))
    avg_trade_size_ema_20: _EMA = field(default_factory=lambda: _EMA(span=20))
    volume_per_trade_ema_20: _EMA = field(default_factory=lambda: _EMA(span=20))

    # Funding deep features
    funding_window_24: RollingWindow = field(default_factory=lambda: RollingWindow(24))
    funding_history_8: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _funding_sign_count: int = 0
    _funding_last_sign: int = 0  # +1 or -1

    # OI features
    oi_change_ema_8: _EMA = field(default_factory=lambda: _EMA(span=8))
    _last_oi: Optional[float] = None
    _last_oi_change_pct: Optional[float] = None

    # LS Ratio features
    ls_ratio_window_24: RollingWindow = field(default_factory=lambda: RollingWindow(24))
    _last_ls_ratio: Optional[float] = None

    # V14: BTC Dominance (BTC/ETH ratio)
    _dom_ratio_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=75))
    _last_eth_close: Optional[float] = None
    _multi_dom_ratio_bufs: Dict[str, Deque[float]] = field(default_factory=dict)

    # V15: Return buffer for autocorrelation and skewness (24-bar window)
    _ret_buf_24: Deque[float] = field(default_factory=lambda: deque(maxlen=24))

    # V16: Orderbook proxy state
    _taker_imb_buf_6: Deque[float] = field(default_factory=lambda: deque(maxlen=6))
    _vol_buf_6: Deque[float] = field(default_factory=lambda: deque(maxlen=6))
    _vol_buf_24: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    # V16: Liquidation proxy state
    _liq_vol_buf_24: Deque[float] = field(default_factory=lambda: deque(maxlen=24))

    # V13: Enhanced OI/LS/Taker
    _oi_buf_12: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    _last_top_trader_ls: Optional[float] = None
    _last_taker_sell_volume: float = 0.0

    # V18: Long-window OI buffer for oi_change_24 and oi_change_96
    _oi_buf_97: Deque[float] = field(default_factory=lambda: deque(maxlen=97))

    # V19: DVOL (Deribit implied volatility) buffer — needs 721 for 720-bar mean
    _dvol_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=721))

    # V5: Order flow windows
    cvd_window_10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    cvd_window_20: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    taker_ratio_window_50: RollingWindow = field(default_factory=lambda: RollingWindow(50))

    # V5: Volatility microstructure
    vol_5_history: Deque[float] = field(default_factory=lambda: deque(maxlen=25))
    hl_log_sq_window: RollingWindow = field(default_factory=lambda: RollingWindow(20))

    # V5: Liquidation proxy
    leverage_proxy_ema: _EMA = field(default_factory=lambda: _EMA(span=20))
    _prev_oi_change_for_accel: Optional[float] = None

    # V7: Basis features (spot-futures)
    basis_window_24: RollingWindow = field(default_factory=lambda: RollingWindow(24))
    basis_ema_8: _EMA = field(default_factory=lambda: _EMA(span=8))
    _last_basis: Optional[float] = None

    # V7: Fear & Greed Index
    fgi_window_7: RollingWindow = field(default_factory=lambda: RollingWindow(7))
    fgi_window_14: RollingWindow = field(default_factory=lambda: RollingWindow(14))
    fgi_history_7d: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _last_fgi: Optional[float] = None

    # V8: VWAP deviation windows
    vwap_cv_window: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    vwap_v_window: RollingWindow = field(default_factory=lambda: RollingWindow(20))

    # V8: Adaptive vol regime
    vol_regime_ema: _EMA = field(default_factory=lambda: _EMA(span=5))
    vol_regime_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    # V9: Deribit IV features
    iv_window_24: RollingWindow = field(default_factory=lambda: RollingWindow(24))
    _last_implied_vol: Optional[float] = None
    _last_put_call_ratio: Optional[float] = None

    # V10: On-chain features (daily, from Coin Metrics)
    _onchain_netflow_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    _onchain_flowin_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    _onchain_flowout_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    _onchain_supply_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _onchain_addr_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    _onchain_tx_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    _onchain_hashrate_ema: _EMA = field(default_factory=lambda: _EMA(span=14))
    _last_onchain_supply: Optional[float] = None
    _last_onchain_hashrate: Optional[float] = None

    # V11: Liquidation features
    _liq_volume_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    _liq_imbalance_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=6))
    _last_liq_volume: Optional[float] = None
    _last_liq_imbalance: Optional[float] = None
    _last_liq_count: float = 0.0

    # V11: Mempool features (BTC-only)
    _mempool_fee_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    _mempool_size_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    _last_fee_urgency: Optional[float] = None

    # V11: Macro features (daily)
    _dxy_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    _spx_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _btc_close_buf_30: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _last_spx_close: Optional[float] = None
    _prev_spx_close: Optional[float] = None
    _last_vix: Optional[float] = None
    _vix_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=14))
    _last_macro_date: Optional[str] = None

    # V11: Social sentiment
    _social_vol_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    _last_sentiment_score: Optional[float] = None
    _last_social_volume: Optional[float] = None

    # V12: ALT cross-asset (BTC reference)
    _btc_ref_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    _btc_ref_vol_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=25))
    _alt_btc_ratio_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=25))

    # Acceleration: track recent momentum values
    _prev_momentum: Optional[float] = None
    _last_close: Optional[float] = None
    _last_volume: float = 0.0
    _last_hour: int = -1
    _last_dow: int = -1
    _last_funding_rate: Optional[float] = None
    _last_trades: float = 0.0
    _last_taker_buy_volume: float = 0.0
    _last_taker_buy_quote_volume: float = 0.0
    _last_quote_volume: float = 0.0
    _bar_count: int = 0

    def push(self, close: float, volume: float, high: float, low: float, open_: float,
             *, hour: int = -1, dow: int = -1, funding_rate: Optional[float] = None,
             trades: float = 0.0, taker_buy_volume: float = 0.0,
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
             multi_dom_ratios: Optional[Dict[str, float]] = None,
             dvol: Optional[float] = None) -> None:
        self._last_hour = hour
        self._last_dow = dow
        if funding_rate is not None:
            self._last_funding_rate = funding_rate
            self.funding_ema.push(funding_rate)
            self.funding_window_24.push(funding_rate)
            self.funding_history_8.append(funding_rate)
            # Track sign persistence
            sign = 1 if funding_rate > 0 else (-1 if funding_rate < 0 else 0)
            if sign != 0:
                if sign == self._funding_last_sign:
                    self._funding_sign_count += 1
                else:
                    self._funding_sign_count = 1
                    self._funding_last_sign = sign

        # OI state
        if open_interest is not None:
            if self._last_oi is not None and self._last_oi > 0:
                change = (open_interest - self._last_oi) / self._last_oi
                # V5: track acceleration before overwriting
                self._prev_oi_change_for_accel = self._last_oi_change_pct
                self._last_oi_change_pct = change
                self.oi_change_ema_8.push(change)
            self._last_oi = open_interest
            # V13: OI history buffer for multi-bar change
            self._oi_buf_12.append(open_interest)
            # V18: Long-window OI buffer for oi_change_24 and oi_change_96
            self._oi_buf_97.append(open_interest)

            # V5: leverage proxy = OI / (close * volume), normalized by EMA
            if close > 0 and volume > 0:
                raw_lev = open_interest / (close * volume)
                self.leverage_proxy_ema.push(raw_lev)

        # V13: Top trader LS ratio
        if top_trader_ls_ratio is not None:
            self._last_top_trader_ls = top_trader_ls_ratio

        # V13: Taker sell volume (for buy/sell ratio)
        taker_sell = volume - taker_buy_volume if volume > 0 and taker_buy_volume > 0 else 0.0
        self._last_taker_sell_volume = taker_sell

        # V14: BTC Dominance (BTC/ETH ratio)
        if eth_close is not None and eth_close > 0:
            self._last_eth_close = eth_close
            dom_ratio = close / eth_close
            self._dom_ratio_buf.append(dom_ratio)
        if multi_dom_ratios:
            for prefix, ratio in multi_dom_ratios.items():
                if ratio <= 0:
                    continue
                self._multi_dom_ratio_bufs.setdefault(prefix, deque(maxlen=25)).append(ratio)

        # V19: DVOL buffer
        if dvol is not None and not math.isnan(dvol):
            self._dvol_buf.append(dvol)

        # LS Ratio state
        if ls_ratio is not None:
            self._last_ls_ratio = ls_ratio
            self.ls_ratio_window_24.push(ls_ratio)

        # V7: Basis (spot-futures)
        if spot_close is not None and close > 0 and spot_close > 0:
            basis = (close - spot_close) / spot_close
            self._last_basis = basis
            self.basis_window_24.push(basis)
            self.basis_ema_8.push(basis)

        # V7: Fear & Greed Index (daily — only push to window when value changes)
        if fear_greed is not None:
            if self._last_fgi is None or abs(fear_greed - self._last_fgi) > 0.01:
                self.fgi_window_7.push(fear_greed)
                self.fgi_window_14.push(fear_greed)
                self.fgi_history_7d.append(fear_greed)
            self._last_fgi = fear_greed

        # V9: Deribit IV
        if implied_vol is not None:
            self._last_implied_vol = implied_vol
            self.iv_window_24.push(implied_vol)
        if put_call_ratio is not None:
            self._last_put_call_ratio = put_call_ratio

        # V10: On-chain metrics (daily)
        if onchain_metrics is not None:
            flow_in = onchain_metrics.get("FlowInExUSD")
            flow_out = onchain_metrics.get("FlowOutExUSD")
            if flow_in is not None:
                self._onchain_flowin_buf.append(flow_in)
            if flow_out is not None:
                self._onchain_flowout_buf.append(flow_out)
            if flow_in is not None and flow_out is not None:
                self._onchain_netflow_buf.append(flow_in - flow_out)

            supply = onchain_metrics.get("SplyExNtv")
            if supply is not None:
                self._onchain_supply_buf.append(supply)
                self._last_onchain_supply = supply

            addr = onchain_metrics.get("AdrActCnt")
            if addr is not None:
                self._onchain_addr_buf.append(addr)

            tx = onchain_metrics.get("TxTfrCnt")
            if tx is not None:
                self._onchain_tx_buf.append(tx)

            hr = onchain_metrics.get("HashRate")
            if hr is not None:
                self._onchain_hashrate_ema.push(hr)
                self._last_onchain_hashrate = hr

        # V11: Liquidation metrics
        if liquidation_metrics is not None:
            total = liquidation_metrics.get("liq_total_volume", 0.0)
            buy = liquidation_metrics.get("liq_buy_volume", 0.0)
            sell = liquidation_metrics.get("liq_sell_volume", 0.0)
            self._liq_volume_buf.append(total)
            self._last_liq_volume = total
            self._last_liq_count = liquidation_metrics.get("liq_count", 0.0)
            if total > 0:
                imb = (buy - sell) / total
            else:
                imb = 0.0
            self._liq_imbalance_buf.append(imb)
            self._last_liq_imbalance = imb

        # V11: Mempool metrics
        if mempool_metrics is not None:
            fee = mempool_metrics.get("fastest_fee")
            if fee is not None:
                self._mempool_fee_buf.append(fee)
            size = mempool_metrics.get("mempool_size")
            if size is not None:
                self._mempool_size_buf.append(size)
            eco = mempool_metrics.get("economy_fee")
            if fee is not None and eco is not None and eco > 0:
                self._last_fee_urgency = fee / eco

        # V11: Macro metrics (daily — only push when date changes)
        if macro_metrics is not None:
            date_str_raw = macro_metrics.get("date")
            date_str: Optional[str] = str(date_str_raw) if date_str_raw is not None else None
            if date_str is None or date_str != self._last_macro_date:
                self._last_macro_date = date_str
                dxy = macro_metrics.get("dxy")
                if dxy is not None:
                    self._dxy_buf.append(dxy)
                spx = macro_metrics.get("spx")
                if spx is not None:
                    self._prev_spx_close = self._last_spx_close
                    self._last_spx_close = spx
                    self._spx_buf.append(spx)
                vix = macro_metrics.get("vix")
                if vix is not None:
                    self._last_vix = vix
                    self._vix_buf.append(vix)
            # Always track BTC close for SPX-BTC correlation
            self._btc_close_buf_30.append(close)

        # V11: Social sentiment metrics
        if sentiment_metrics is not None:
            sv = sentiment_metrics.get("social_volume")
            if sv is not None:
                self._social_vol_buf.append(sv)
                self._last_social_volume = sv
            ss = sentiment_metrics.get("sentiment_score")
            if ss is not None:
                self._last_sentiment_score = ss

        # Microstructure state
        self._last_trades = trades
        self._last_taker_buy_volume = taker_buy_volume
        self._last_taker_buy_quote_volume = taker_buy_quote_volume
        self._last_quote_volume = quote_volume

        # V8: VWAP windows
        if volume > 0:
            self.vwap_cv_window.push(close * volume)
            self.vwap_v_window.push(volume)
        if trades > 0:
            self.trades_ema_20.push(trades)
            self.trades_ema_5.push(trades)
            # Taker buy ratio
            tbr = taker_buy_volume / volume if volume > 0 else 0.5
            self.taker_buy_ratio_ema_10.push(tbr)
            # V5: Order flow — push imbalance into CVD windows, tbr into zscore window
            imbalance = 2.0 * tbr - 1.0
            self.cvd_window_10.push(imbalance)
            self.cvd_window_20.push(imbalance)
            self.taker_ratio_window_50.push(tbr)
            # Average trade size
            ats = quote_volume / trades
            self.avg_trade_size_ema_20.push(ats)
            # Volume per trade
            vpt = volume / trades
            self.volume_per_trade_ema_20.push(vpt)

        self._bar_count += 1
        self.close_history.append(close)
        self.open_history.append(open_)
        self.high_history.append(high)
        self.low_history.append(low)

        # MAs
        self.ma_5.push(close)
        self.ma_10.push(close)
        self.ma_20.push(close)
        self.ma_30.push(close)
        self.ma_50.push(close)
        self.bb_window.push(close)

        # Returns
        if self._last_close is not None and self._last_close != 0:
            ret = (close - self._last_close) / self._last_close
            self.return_window_20.push(ret)
            self.return_window_5.push(ret)
            self._ret_buf_24.append(ret)  # V15: for autocorr/skewness
        self._last_close = close

        # V16: Orderbook proxy + liquidation buffers
        self._vol_buf_6.append(volume)
        self._vol_buf_24.append(volume)
        if volume > 0:
            _tbv = taker_buy_volume or 0.0
            _tsv = volume - _tbv
            _total = _tbv + _tsv
            _imb = (_tbv - _tsv) / _total if _total > 0 else 0
            self._taker_imb_buf_6.append(_imb)
        # Liquidation proxy: |OI drop| × volume spike
        if (open_interest is not None and self._last_oi is not None
                and self._last_oi > 0 and volume > 0):
            oi_change = abs(open_interest - self._last_oi) / self._last_oi
            vol_ma = sum(self._vol_buf_24) / max(len(self._vol_buf_24), 1)
            vol_spike = volume / vol_ma if vol_ma > 0 else 1.0
            liq_proxy = oi_change * vol_spike * volume
            self._liq_vol_buf_24.append(liq_proxy)
        elif len(self._liq_vol_buf_24) > 0:
            self._liq_vol_buf_24.append(0.0)  # no data this bar

        # V5: vol_5 history for vol_of_vol and rv_acceleration
        if self.return_window_5.full:
            self.vol_5_history.append(self.return_window_5.std)

        # V8: Adaptive vol regime — track EMA and history of vol_regime
        if self.return_window_5.full and self.return_window_20.full:
            v5 = self.return_window_5.std
            v20 = self.return_window_20.std
            if v20 is not None and v20 > 1e-12:
                vr = v5 / v20
                self.vol_regime_ema.push(vr)
                self.vol_regime_history.append(vr)

        # V5: Parkinson volatility — ln(H/L)^2
        if high > 0 and low > 0 and high >= low:
            hl_ratio = high / low
            if hl_ratio > 0:
                ln_hl = math.log(hl_ratio)
                self.hl_log_sq_window.push(ln_hl * ln_hl)

        # Volume
        self._last_volume = volume
        self.vol_window_20.push(volume)
        self.vol_window_5.push(volume)

        # RSI
        self.rsi_14.push(close)
        self.rsi_6.push(close)

        # MACD
        self.ema_12.push(close)
        self.ema_26.push(close)
        if self.ema_12.ready and self.ema_26.ready:
            macd_val = self.ema_12.value - self.ema_26.value  # type: ignore[operator]
            self.macd_signal_ema.push(macd_val)

        # ATR
        self.atr_14.push(high, low, close)

        # ADX
        self.adx_14.push(high, low, close)

    def get_features(self, btc_close: float | None = None) -> Dict[str, Optional[float]]:
        """Compute all features from current state.

        Delegates to enriched_feature_getters.compute_base_features().
        """
        from features.enriched_feature_getters import compute_base_features
        return compute_base_features(self, btc_close)


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

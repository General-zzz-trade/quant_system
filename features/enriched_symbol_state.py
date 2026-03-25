"""_SymbolState — per-symbol state for enriched feature computation.

Extracted from enriched_computer.py to keep both files under 500 lines.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from _quant_hotpath import RollingWindow, VWAPWindow

from features.enriched_trackers import (
    _EMA,
    _RSITracker,
    _ATRTracker,
    _ADXTracker,
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

    # V8: VWAP deviation window (Rust VWAPWindow — single push(price, volume))
    vwap_window: Any = field(default_factory=lambda: VWAPWindow(20))

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
        """Push a new bar of data and update all trackers.

        Delegates to enriched_symbol_state_push module for the actual logic.
        """
        from features.enriched_symbol_state_push import push_bar
        push_bar(self, close, volume, high, low, open_,
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

    def get_features(self, btc_close: float | None = None) -> Dict[str, Optional[float]]:
        """Compute all features from current state.

        Delegates to enriched_feature_getters.compute_base_features().
        """
        from features.enriched_feature_getters import compute_base_features
        return compute_base_features(self, btc_close)

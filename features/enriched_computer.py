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
from math import sqrt
from typing import Deque, Dict, List, Optional

from features.rolling import RollingWindow

logger = logging.getLogger(__name__)

# Feature names produced by this computer — used by training scripts
ENRICHED_FEATURE_NAMES: tuple[str, ...] = (
    # Returns at multiple horizons
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    # Moving average crossovers
    "ma_cross_10_30", "ma_cross_5_20", "close_vs_ma20", "close_vs_ma50",
    # RSI
    "rsi_14", "rsi_6",
    # MACD
    "macd_line", "macd_signal", "macd_hist",
    # Bollinger Bands
    "bb_width_20", "bb_pctb_20",
    # ATR
    "atr_norm_14",
    # ADX (trend strength)
    "adx_14",
    # Volatility
    "vol_20", "vol_5",
    # Volume
    "vol_ratio_20", "vol_ma_ratio_5_20",
    # Candle structure
    "body_ratio", "upper_shadow", "lower_shadow",
    # Trend
    "mean_reversion_20", "price_acceleration",
    # --- P0 Crypto-native features ---
    # Time-of-day (cyclical encoding)
    "hour_sin", "hour_cos",
    # Day-of-week (cyclical encoding)
    "dow_sin", "dow_cos",
    # Volatility regime
    "vol_regime",  # vol_5 / vol_20 ratio (>1 = expanding, <1 = contracting)
    # Funding rate features (set externally)
    "funding_rate", "funding_ma8",  # current rate + 8-period MA (= 64h)
    # --- Kline microstructure features ---
    "trade_intensity",        # trades / EMA(trades, 20) — activity anomaly
    "taker_buy_ratio",        # taker_buy_vol / volume — active buy ratio (0~1)
    "taker_buy_ratio_ma10",   # EMA(taker_buy_ratio, 10) — smoothed buy pressure
    "taker_imbalance",        # 2 * taker_buy_ratio - 1 — buy/sell imbalance (-1~1)
    "avg_trade_size",         # quote_vol / trades — average trade size
    "avg_trade_size_ratio",   # avg_trade_size / EMA(avg_trade_size, 20) — whale detection
    "volume_per_trade",       # volume / trades (normalized by EMA) — per-trade volume
    "trade_count_regime",     # EMA(trades, 5) / EMA(trades, 20) — activity expansion
    # --- Funding deep features ---
    "funding_zscore_24",      # (rate - mean_24) / std_24 — extreme funding detection
    "funding_momentum",       # rate - funding_ma8 — funding change velocity
    "funding_extreme",        # |zscore| > 2 flag — crowded trade signal
    "funding_cumulative_8",   # sum of last 8 funding settlements — holding cost
    "funding_sign_persist",   # consecutive same-sign funding count — bias persistence
    # --- Open Interest features ---
    "oi_change_pct",          # OI change rate
    "oi_change_ma8",          # OI change rate EMA(8)
    "oi_close_divergence",    # price direction vs OI direction divergence
    # --- Long/Short Ratio features ---
    "ls_ratio",               # long/short account ratio
    "ls_ratio_zscore_24",     # 24-bar z-score of LS ratio
    "ls_extreme",             # extreme long/short bias flag
    # --- V5: Order Flow features ---
    "cvd_10",                 # cumulative volume delta 10 bars
    "cvd_20",                 # cumulative volume delta 20 bars
    "cvd_price_divergence",   # CVD vs price direction divergence flag
    "aggressive_flow_zscore", # z-score of taker buy ratio over 50 bars
    # --- V5: Volatility microstructure ---
    "vol_of_vol",             # std of vol_5 over 20 bars
    "range_vs_rv",            # (H-L)/C / vol_5 — intrabar range vs realized vol
    "parkinson_vol",          # Parkinson volatility estimator (20 bars)
    "rv_acceleration",        # vol_5[t] - vol_5[t-5]
    # --- V5: Liquidation proxy ---
    "oi_acceleration",        # oi_change_pct acceleration
    "leverage_proxy",         # OI/(close*volume) normalized by EMA(20)
    "oi_vol_divergence",      # OI up + volume down divergence flag
    "oi_liquidation_flag",    # large OI drop + volume spike flag
    # --- V5: Funding carry ---
    "funding_annualized",     # funding_rate * 3 * 365
    "funding_vs_vol",         # funding_rate / vol_20
    # --- V7: Spot-futures basis ---
    "basis",                  # (futures_close - spot_close) / spot_close
    "basis_zscore_24",        # z-score of basis over 24 bars
    "basis_momentum",         # basis - EMA(basis, 8)
    "basis_extreme",          # |zscore| > 2 flag
    # --- V7: Fear & Greed Index ---
    "fgi_normalized",         # FGI / 100 - 0.5 (range [-0.5, 0.5])
    "fgi_zscore_7",           # (FGI - mean_7d) / std_7d
    "fgi_extreme",            # FGI < 25 → -1, FGI > 75 → 1, else 0
    # --- V8: Alpha Rebuild V3 features ---
    "taker_bq_ratio",         # taker_buy_quote_volume / quote_volume — USD-weighted buy pressure
    "vwap_dev_20",            # (close - VWAP_20) / close — VWAP deviation (mean reversion)
    "volume_momentum_10",     # ret_10 × clip(volume/SMA_vol_20, 3.0) — volume-confirmed momentum
    "mom_vol_divergence",     # sign(ret_1)==sign(vol_ratio-1) ? +1 : -1 — exhaustion detector
    "basis_carry_adj",        # basis + funding_rate × 3 — carry-adjusted basis
    "vol_regime_adaptive",    # EMA(vol_regime,5) vs 30-bar median — regime persistence
    # --- V9: Cross-factor interaction features ---
    "liquidation_cascade_score",  # |oi_change_pct| * vol_ma_ratio_5_20 — liquidation pressure
    "funding_term_slope",         # (funding_rate - funding_ma8) / max(|funding_ma8|, 1e-6) — funding curve slope
    "cross_tf_regime_sync",       # sign(close_vs_ma20) == sign(tf4h_close_vs_ma20) — multi-TF alignment
    # --- V9: Deribit IV features (set externally) ---
    "implied_vol_zscore_24",      # IV z-score over 24 bars
    "iv_rv_spread",               # implied vol - realized vol (vol_20)
    "put_call_ratio",             # Deribit options put/call OI ratio
    # --- V10: On-chain features (Coin Metrics, daily) ---
    "exchange_netflow_zscore",    # zscore_7d(exchange inflow - outflow)
    "exchange_supply_change",     # daily pct change of exchange-held supply
    "exchange_supply_zscore_30",  # zscore_30d(SplyExNtv)
    "active_addr_zscore_14",      # zscore_14d(AdrActCnt)
    "tx_count_zscore_14",         # zscore_14d(TxTfrCnt)
    "hashrate_momentum",          # (HashRate - EMA14) / EMA14
    # --- V11: Liquidation features (Binance force orders) ---
    "liquidation_volume_zscore_24",   # z-score of hourly liquidation volume over 24 bars
    "liquidation_imbalance",          # (buy_liq - sell_liq) / total_liq — directional
    "liquidation_volume_ratio",       # liq_volume / trade_volume — leverage flush intensity
    "liquidation_cluster_flag",       # short-window cluster detection flag
    # --- V11: Mempool features (BTC-only, mempool.space) ---
    "mempool_fee_zscore_24",          # z-score of recommended fee over 24 bars
    "mempool_size_zscore_24",         # z-score of mempool size over 24 bars
    "fee_urgency_ratio",             # fastest_fee / economy_fee
    # --- V11: Macro features (DXY/SPX/VIX, daily) ---
    "dxy_change_5d",                  # 5-day DXY return
    "spx_btc_corr_30d",              # 30-day SPX-BTC correlation
    "spx_overnight_ret",              # SPX overnight return
    "vix_zscore_14",                  # 14-day VIX z-score
    # --- V11: Social sentiment ---
    "social_volume_zscore_24",        # z-score of social volume over 24 bars
    "social_sentiment_score",         # normalized sentiment score
    "social_volume_price_div",        # social volume vs price direction divergence
    # --- V12: ALT coin cross-asset features ---
    "btc_relative_strength_24",       # ALT ret_24 - BTC ret_24 — outperformance signal
    "btc_relative_strength_6",        # ALT ret_6 - BTC ret_6 — short-term alpha vs BTC
    "btc_ratio_ma20_dev",             # (ALT/BTC ratio) / MA20(ratio) - 1 — pair mean reversion
    "btc_dom_momentum",               # BTC 24h ret - ALT 24h ret — dominance shift velocity
    "btc_lead_ret_1",                 # BTC ret_1 (lagged signal for ALT)
    "btc_vol_ratio",                  # ALT vol_20 / BTC vol_20 — relative volatility regime
    # --- V14: BTC Dominance features (BTC/ETH ratio for BTC alpha) ---
    "btc_dom_dev_20",                 # BTC/ETH ratio deviation from MA(20) — capital flow signal
    "btc_dom_dev_50",                 # BTC/ETH ratio deviation from MA(50) — trend signal
    "btc_dom_ret_24",                 # BTC/ETH ratio 24-bar return — short-term dominance shift
    "btc_dom_ret_72",                 # BTC/ETH ratio 72-bar return — medium-term dominance shift
    # --- V14b: Multi-ratio dominance features (symbol vs reference pairs) ---
    "dom_vs_sui_dev_20",              # ratio vs SUI deviation from MA(20)
    "dom_vs_sui_ret_24",             # ratio vs SUI 24-bar return
    "dom_vs_axs_dev_20",             # ratio vs AXS deviation from MA(20)
    "dom_vs_axs_ret_24",             # ratio vs AXS 24-bar return
    "dom_vs_eth_dev_20",             # ratio vs ETH deviation from MA(20)
    "dom_vs_eth_ret_24",             # ratio vs ETH 24-bar return
    # --- V13: Enhanced OI/LS/Taker features (IC-validated 2026-03) ---
    "oi_pct_4h",                      # 4-bar OI change rate — short-term position buildup
    "ls_deviation",                   # ls_ratio - 1.0 — directional bias magnitude
    "taker_buy_sell_ratio",           # taker_buy_vol / taker_sell_vol — raw buy/sell pressure
    "top_retail_divergence",          # top_trader_ls - global_ls — smart vs dumb money divergence
    "oi_price_divergence_12",         # 12-bar OI change - 12-bar price change — position/price mismatch
)

_WARMUP_BARS = 65  # bars needed before all features are valid (funding_ma8 needs 64h)


@dataclass
class _EMA:
    """Incremental EMA tracker."""
    span: int
    _alpha: float = 0.0
    _value: float = 0.0
    _n: int = 0

    def __post_init__(self) -> None:
        self._alpha = 2.0 / (self.span + 1.0)

    def push(self, x: float) -> None:
        if self._n == 0:
            self._value = x
        else:
            self._value = self._alpha * x + (1.0 - self._alpha) * self._value
        self._n += 1

    @property
    def value(self) -> Optional[float]:
        return self._value if self._n > 0 else None

    @property
    def ready(self) -> bool:
        return self._n >= self.span


@dataclass
class _RSITracker:
    """Incremental RSI using Wilder's smoothing."""
    period: int
    _avg_gain: float = 0.0
    _avg_loss: float = 0.0
    _n: int = 0
    _prev_close: Optional[float] = None
    _init_gains: float = 0.0
    _init_losses: float = 0.0

    def push(self, close: float) -> None:
        if self._prev_close is None:
            self._prev_close = close
            return

        change = close - self._prev_close
        self._prev_close = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._n += 1

        if self._n <= self.period:
            self._init_gains += gain
            self._init_losses += loss
            if self._n == self.period:
                self._avg_gain = self._init_gains / self.period
                self._avg_loss = self._init_losses / self.period
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

    @property
    def value(self) -> Optional[float]:
        if self._n < self.period:
            return None
        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class _ATRTracker:
    """Incremental ATR using Wilder's smoothing."""
    period: int
    _atr: float = 0.0
    _n: int = 0
    _prev_close: Optional[float] = None
    _init_sum: float = 0.0

    def push(self, high: float, low: float, close: float) -> None:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        self._prev_close = close
        self._n += 1

        if self._n <= self.period:
            self._init_sum += tr
            if self._n == self.period:
                self._atr = self._init_sum / self.period
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

    @property
    def value(self) -> Optional[float]:
        return self._atr if self._n >= self.period else None


@dataclass
class _ADXTracker:
    """Incremental ADX(14) using Wilder's smoothing.

    Requires 2*period bars of warmup before producing a value.
    Steps:
      1. True Range, +DM, -DM each bar
      2. Wilder-smooth TR, +DM, -DM over `period` bars
      3. +DI = 100 * smooth_+DM / smooth_TR, -DI analogous
      4. DX = 100 * |+DI - -DI| / (+DI + -DI)
      5. ADX = Wilder-smooth DX over `period` bars
    """
    period: int = 14
    _n: int = 0
    _prev_high: Optional[float] = None
    _prev_low: Optional[float] = None
    _prev_close: Optional[float] = None
    # Wilder-smoothed values
    _smooth_tr: float = 0.0
    _smooth_plus_dm: float = 0.0
    _smooth_minus_dm: float = 0.0
    # ADX Wilder smoothing
    _adx: float = 0.0
    _dx_init_sum: float = 0.0
    _dx_count: int = 0
    _adx_initialized: bool = False

    def push(self, high: float, low: float, close: float) -> None:
        if self._prev_close is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return

        self._n += 1
        prev_high = self._prev_high or high
        prev_low = self._prev_low or low
        prev_close = self._prev_close

        # True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        p = self.period
        if self._n <= p:
            # Accumulate initial sums
            self._smooth_tr += tr
            self._smooth_plus_dm += plus_dm
            self._smooth_minus_dm += minus_dm
            if self._n == p:
                # First DX after p bars
                self._compute_dx_and_accumulate()
        else:
            # Wilder smoothing: new = prev - prev/period + current
            self._smooth_tr = self._smooth_tr - self._smooth_tr / p + tr
            self._smooth_plus_dm = self._smooth_plus_dm - self._smooth_plus_dm / p + plus_dm
            self._smooth_minus_dm = self._smooth_minus_dm - self._smooth_minus_dm / p + minus_dm
            self._compute_dx_and_accumulate()

    def _compute_dx_and_accumulate(self) -> None:
        if self._smooth_tr == 0:
            # No price movement → no directional movement → DX = 0
            dx = 0.0
            p = self.period
            if not self._adx_initialized:
                self._dx_init_sum += dx
                self._dx_count += 1
                if self._dx_count >= p:
                    self._adx = self._dx_init_sum / p
                    self._adx_initialized = True
            else:
                self._adx = (self._adx * (p - 1) + dx) / p
            return
        plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
        minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0.0
        else:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum

        p = self.period
        if not self._adx_initialized:
            self._dx_init_sum += dx
            self._dx_count += 1
            if self._dx_count >= p:
                self._adx = self._dx_init_sum / p
                self._adx_initialized = True
        else:
            self._adx = (self._adx * (p - 1) + dx) / p

    @property
    def value(self) -> Optional[float]:
        return self._adx if self._adx_initialized else None


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

    # V13: Enhanced OI/LS/Taker
    _oi_buf_12: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    _last_top_trader_ls: Optional[float] = None
    _last_taker_sell_volume: float = 0.0

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
    _onchain_netflow_buf: Deque[float] = field(default_factory=lambda: deque(maxlen=7))
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
             sentiment_metrics: Optional[Dict[str, float]] = None) -> None:
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
        self._last_close = close

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
        """Compute all features from current state."""
        feats: Dict[str, Optional[float]] = {}
        close = self._last_close

        # --- Multi-horizon returns ---
        hist = self.close_history
        n = len(hist)
        for horizon, name in [(1, "ret_1"), (3, "ret_3"), (6, "ret_6"),
                              (12, "ret_12"), (24, "ret_24")]:
            if n > horizon and hist[-1 - horizon] != 0:
                feats[name] = (hist[-1] - hist[-1 - horizon]) / hist[-1 - horizon]
            else:
                feats[name] = None

        # --- MA crossovers ---
        ma10 = self.ma_10.mean if self.ma_10.full else None
        ma30 = self.ma_30.mean if self.ma_30.full else None
        ma5 = self.ma_5.mean if self.ma_5.full else None
        ma20 = self.ma_20.mean if self.ma_20.full else None
        ma50 = self.ma_50.mean if self.ma_50.full else None

        if ma10 is not None and ma30 is not None and ma30 != 0:
            feats["ma_cross_10_30"] = ma10 / ma30 - 1.0
        else:
            feats["ma_cross_10_30"] = None

        if ma5 is not None and ma20 is not None and ma20 != 0:
            feats["ma_cross_5_20"] = ma5 / ma20 - 1.0
        else:
            feats["ma_cross_5_20"] = None

        if close is not None and ma20 is not None and ma20 != 0:
            feats["close_vs_ma20"] = close / ma20 - 1.0
        else:
            feats["close_vs_ma20"] = None

        if close is not None and ma50 is not None and ma50 != 0:
            feats["close_vs_ma50"] = close / ma50 - 1.0
        else:
            feats["close_vs_ma50"] = None

        # --- RSI ---
        feats["rsi_14"] = self.rsi_14.value
        feats["rsi_6"] = self.rsi_6.value
        # Normalize RSI to [-1, 1] range for ML
        if feats["rsi_14"] is not None:
            feats["rsi_14"] = (feats["rsi_14"] - 50.0) / 50.0
        if feats["rsi_6"] is not None:
            feats["rsi_6"] = (feats["rsi_6"] - 50.0) / 50.0

        # --- MACD ---
        if self.ema_12.ready and self.ema_26.ready:
            macd_line = self.ema_12.value - self.ema_26.value  # type: ignore[operator]
            # Normalize by close
            if close and close != 0:
                feats["macd_line"] = macd_line / close
            else:
                feats["macd_line"] = None

            if self.macd_signal_ema.ready:
                sig = self.macd_signal_ema.value
                if close and close != 0:
                    feats["macd_signal"] = sig / close  # type: ignore[operator]
                    feats["macd_hist"] = (macd_line - sig) / close  # type: ignore[operator]
                else:
                    feats["macd_signal"] = None
                    feats["macd_hist"] = None
            else:
                feats["macd_signal"] = None
                feats["macd_hist"] = None
        else:
            feats["macd_line"] = None
            feats["macd_signal"] = None
            feats["macd_hist"] = None

        # --- Bollinger Bands ---
        if self.bb_window.full:
            bb_mid = self.bb_window.mean
            bb_std = self.bb_window.std
            if bb_mid and bb_std and bb_mid != 0:
                upper = bb_mid + 2.0 * bb_std
                lower = bb_mid - 2.0 * bb_std
                feats["bb_width_20"] = (upper - lower) / bb_mid
                band_range = upper - lower
                if band_range != 0 and close is not None:
                    feats["bb_pctb_20"] = (close - lower) / band_range
                else:
                    feats["bb_pctb_20"] = None
            else:
                feats["bb_width_20"] = None
                feats["bb_pctb_20"] = None
        else:
            feats["bb_width_20"] = None
            feats["bb_pctb_20"] = None

        # --- ATR (normalized by close) ---
        atr_val = self.atr_14.value
        if atr_val is not None and close and close != 0:
            feats["atr_norm_14"] = atr_val / close
        else:
            feats["atr_norm_14"] = None

        # --- ADX (trend strength, 0-100) ---
        feats["adx_14"] = self.adx_14.value

        # --- Volatility ---
        feats["vol_20"] = self.return_window_20.std if self.return_window_20.full else None
        feats["vol_5"] = self.return_window_5.std if self.return_window_5.full else None

        # --- Volume features ---
        vol_ma20 = self.vol_window_20.mean if self.vol_window_20.full else None
        vol_ma5 = self.vol_window_5.mean if self.vol_window_5.full else None
        if vol_ma20 and vol_ma20 != 0 and self.vol_window_20.n > 0:
            feats["vol_ratio_20"] = self._last_volume / vol_ma20
        else:
            feats["vol_ratio_20"] = None

        if vol_ma5 is not None and vol_ma20 is not None and vol_ma20 != 0:
            feats["vol_ma_ratio_5_20"] = vol_ma5 / vol_ma20
        else:
            feats["vol_ma_ratio_5_20"] = None

        # --- Candle structure ---
        if n > 0 and len(self.open_history) > 0 and len(self.high_history) > 0 and len(self.low_history) > 0:
            o = self.open_history[-1]
            h = self.high_history[-1]
            l = self.low_history[-1]  # noqa: E741
            c = hist[-1]
            hl_range = h - l
            if hl_range > 0:
                feats["body_ratio"] = (c - o) / hl_range
                feats["upper_shadow"] = (h - max(o, c)) / hl_range
                feats["lower_shadow"] = (min(o, c) - l) / hl_range
            else:
                feats["body_ratio"] = None
                feats["upper_shadow"] = None
                feats["lower_shadow"] = None
        else:
            feats["body_ratio"] = None
            feats["upper_shadow"] = None
            feats["lower_shadow"] = None

        # --- Mean reversion (z-score) ---
        if self.bb_window.full and close is not None:
            bb_mid = self.bb_window.mean
            bb_std = self.bb_window.std
            if bb_mid is not None and bb_std is not None and bb_std != 0:
                feats["mean_reversion_20"] = (close - bb_mid) / bb_std
            else:
                feats["mean_reversion_20"] = None
        else:
            feats["mean_reversion_20"] = None

        # --- Price acceleration (change in momentum) ---
        current_momentum = feats.get("ma_cross_10_30")
        if current_momentum is not None and self._prev_momentum is not None:
            feats["price_acceleration"] = current_momentum - self._prev_momentum
        else:
            feats["price_acceleration"] = None
        self._prev_momentum = current_momentum

        # --- Time-of-day (cyclical encoding) ---
        if self._last_hour >= 0:
            feats["hour_sin"] = math.sin(2 * math.pi * self._last_hour / 24.0)
            feats["hour_cos"] = math.cos(2 * math.pi * self._last_hour / 24.0)
        else:
            feats["hour_sin"] = None
            feats["hour_cos"] = None

        # --- Day-of-week (cyclical encoding) ---
        if self._last_dow >= 0:
            feats["dow_sin"] = math.sin(2 * math.pi * self._last_dow / 7.0)
            feats["dow_cos"] = math.cos(2 * math.pi * self._last_dow / 7.0)
        else:
            feats["dow_sin"] = None
            feats["dow_cos"] = None

        # --- Volatility regime ---
        vol5 = feats.get("vol_5")
        vol20 = feats.get("vol_20")
        if vol5 is not None and vol20 is not None and vol20 != 0:
            feats["vol_regime"] = vol5 / vol20
        else:
            feats["vol_regime"] = None

        # --- Funding rate features ---
        feats["funding_rate"] = self._last_funding_rate
        feats["funding_ma8"] = self.funding_ema.value if self.funding_ema.ready else None

        # --- Kline microstructure features ---
        trades = self._last_trades
        if trades > 0 and self.trades_ema_20.ready:
            ema_trades_20 = self.trades_ema_20.value
            feats["trade_intensity"] = trades / ema_trades_20 if ema_trades_20 and ema_trades_20 > 0 else None
        else:
            feats["trade_intensity"] = None

        volume = self._last_volume
        tbr: Optional[float] = None
        if trades > 0 and volume > 0:
            tbr = self._last_taker_buy_volume / volume
            feats["taker_buy_ratio"] = tbr
        else:
            feats["taker_buy_ratio"] = None

        feats["taker_buy_ratio_ma10"] = self.taker_buy_ratio_ema_10.value if self.taker_buy_ratio_ema_10.ready else None

        if tbr is not None:
            feats["taker_imbalance"] = 2.0 * tbr - 1.0
        else:
            feats["taker_imbalance"] = None

        if trades > 0:
            ats = self._last_quote_volume / trades
            feats["avg_trade_size"] = ats
            ats_ema = self.avg_trade_size_ema_20.value
            if self.avg_trade_size_ema_20.ready and ats_ema and ats_ema > 0:
                feats["avg_trade_size_ratio"] = ats / ats_ema
            else:
                feats["avg_trade_size_ratio"] = None

            vpt = volume / trades
            vpt_ema = self.volume_per_trade_ema_20.value
            if self.volume_per_trade_ema_20.ready and vpt_ema and vpt_ema > 0:
                feats["volume_per_trade"] = vpt / vpt_ema
            else:
                feats["volume_per_trade"] = None
        else:
            feats["avg_trade_size"] = None
            feats["avg_trade_size_ratio"] = None
            feats["volume_per_trade"] = None

        if self.trades_ema_5.ready and self.trades_ema_20.ready:
            ema5 = self.trades_ema_5.value
            ema20 = self.trades_ema_20.value
            if ema5 is not None and ema20 is not None and ema20 > 0:
                feats["trade_count_regime"] = ema5 / ema20
            else:
                feats["trade_count_regime"] = None
        else:
            feats["trade_count_regime"] = None

        # --- Funding deep features ---
        if self.funding_window_24.full:
            f_mean = self.funding_window_24.mean
            f_std = self.funding_window_24.std
            if f_std is not None and f_std > 1e-12 and self._last_funding_rate is not None:
                zscore = (self._last_funding_rate - f_mean) / f_std
                feats["funding_zscore_24"] = zscore
                feats["funding_extreme"] = 1.0 if abs(zscore) > 2.0 else 0.0
            else:
                feats["funding_zscore_24"] = None
                feats["funding_extreme"] = None
        else:
            feats["funding_zscore_24"] = None
            feats["funding_extreme"] = None

        fr_ma8 = feats.get("funding_ma8")
        if self._last_funding_rate is not None and fr_ma8 is not None:
            feats["funding_momentum"] = self._last_funding_rate - fr_ma8
        else:
            feats["funding_momentum"] = None

        if len(self.funding_history_8) == 8:
            feats["funding_cumulative_8"] = sum(self.funding_history_8)
        else:
            feats["funding_cumulative_8"] = None

        feats["funding_sign_persist"] = float(self._funding_sign_count) if self._funding_sign_count > 0 else None

        # --- OI features ---
        feats["oi_change_pct"] = self._last_oi_change_pct
        feats["oi_change_ma8"] = self.oi_change_ema_8.value if self.oi_change_ema_8.ready else None

        # OI-price divergence: price up but OI down (or vice versa)
        ret1 = feats.get("ret_1")
        if ret1 is not None and self._last_oi_change_pct is not None:
            # Divergence: opposite signs → positive value; same signs → negative
            price_sign = 1.0 if ret1 > 0 else (-1.0 if ret1 < 0 else 0.0)
            oi_sign = 1.0 if self._last_oi_change_pct > 0 else (-1.0 if self._last_oi_change_pct < 0 else 0.0)
            feats["oi_close_divergence"] = -price_sign * oi_sign
        else:
            feats["oi_close_divergence"] = None

        # --- LS Ratio features ---
        feats["ls_ratio"] = self._last_ls_ratio
        if self.ls_ratio_window_24.full and self._last_ls_ratio is not None:
            ls_mean = self.ls_ratio_window_24.mean
            ls_std = self.ls_ratio_window_24.std
            if ls_std is not None and ls_std > 1e-12:
                zscore = (self._last_ls_ratio - ls_mean) / ls_std
                feats["ls_ratio_zscore_24"] = zscore
                feats["ls_extreme"] = 1.0 if abs(zscore) > 2.0 else 0.0
            else:
                feats["ls_ratio_zscore_24"] = None
                feats["ls_extreme"] = None
        else:
            feats["ls_ratio_zscore_24"] = None
            feats["ls_extreme"] = None

        # --- V14: BTC Dominance features ---
        buf = self._dom_ratio_buf
        if len(buf) >= 21:
            cur = buf[-1]
            ma20 = sum(list(buf)[-20:]) / 20
            feats["btc_dom_dev_20"] = cur / ma20 - 1 if ma20 > 0 else None
        else:
            feats["btc_dom_dev_20"] = None

        if len(buf) >= 51:
            cur = buf[-1]
            ma50 = sum(list(buf)[-50:]) / 50
            feats["btc_dom_dev_50"] = cur / ma50 - 1 if ma50 > 0 else None
        else:
            feats["btc_dom_dev_50"] = None

        if len(buf) >= 25:
            feats["btc_dom_ret_24"] = buf[-1] / buf[-25] - 1 if buf[-25] > 0 else None
        else:
            feats["btc_dom_ret_24"] = None

        if len(buf) >= 73:
            feats["btc_dom_ret_72"] = buf[-1] / buf[-73] - 1 if buf[-73] > 0 else None
        else:
            feats["btc_dom_ret_72"] = None

        # --- V13: Enhanced OI/LS/Taker features ---
        # oi_pct_4h: 4-bar OI change rate
        if len(self._oi_buf_12) >= 5 and self._oi_buf_12[-5] > 0:
            feats["oi_pct_4h"] = (self._oi_buf_12[-1] - self._oi_buf_12[-5]) / self._oi_buf_12[-5]
        else:
            feats["oi_pct_4h"] = None

        # ls_deviation: ls_ratio - 1.0
        if self._last_ls_ratio is not None:
            feats["ls_deviation"] = self._last_ls_ratio - 1.0
        else:
            feats["ls_deviation"] = None

        # taker_buy_sell_ratio: buy_vol / sell_vol
        if self._last_taker_buy_volume > 0 and self._last_taker_sell_volume > 0:
            feats["taker_buy_sell_ratio"] = self._last_taker_buy_volume / self._last_taker_sell_volume
        else:
            feats["taker_buy_sell_ratio"] = None

        # top_retail_divergence: top_trader_ls - global_ls
        if self._last_top_trader_ls is not None and self._last_ls_ratio is not None:
            feats["top_retail_divergence"] = self._last_top_trader_ls - self._last_ls_ratio
        else:
            feats["top_retail_divergence"] = None

        # oi_price_divergence_12: 12-bar OI change - 12-bar price change
        if len(self._oi_buf_12) >= 12 and self._oi_buf_12[0] > 0:
            oi_change_12 = (self._oi_buf_12[-1] - self._oi_buf_12[0]) / self._oi_buf_12[0]
            ret_12 = feats.get("ret_12")
            if ret_12 is not None:
                feats["oi_price_divergence_12"] = oi_change_12 - ret_12
            else:
                feats["oi_price_divergence_12"] = None
        else:
            feats["oi_price_divergence_12"] = None

        # --- V5: Order Flow features ---
        if self.cvd_window_10.full:
            feats["cvd_10"] = self.cvd_window_10.mean * self.cvd_window_10.n
        else:
            feats["cvd_10"] = None

        if self.cvd_window_20.full:
            cvd_20_val = self.cvd_window_20.mean * self.cvd_window_20.n
            feats["cvd_20"] = cvd_20_val
            # CVD-price divergence: sign(cvd_20) != sign(ret_20)
            if n > 20 and hist[-21] != 0:
                ret_20 = (hist[-1] - hist[-21]) / hist[-21]
                cvd_sign = 1.0 if cvd_20_val > 0 else (-1.0 if cvd_20_val < 0 else 0.0)
                ret_sign = 1.0 if ret_20 > 0 else (-1.0 if ret_20 < 0 else 0.0)
                feats["cvd_price_divergence"] = 1.0 if cvd_sign != 0 and cvd_sign != ret_sign else 0.0
            else:
                feats["cvd_price_divergence"] = None
        else:
            feats["cvd_20"] = None
            feats["cvd_price_divergence"] = None

        if self.taker_ratio_window_50.full:
            tr_mean = self.taker_ratio_window_50.mean
            tr_std = self.taker_ratio_window_50.std
            if tr_std is not None and tr_std > 1e-12 and tbr is not None:
                feats["aggressive_flow_zscore"] = (tbr - tr_mean) / tr_std
            else:
                feats["aggressive_flow_zscore"] = None
        else:
            feats["aggressive_flow_zscore"] = None

        # --- V5: Volatility microstructure ---
        vol5_val = feats.get("vol_5")
        vol20_val = feats.get("vol_20")

        if len(self.vol_5_history) >= 20:
            recent = list(self.vol_5_history)[-20:]
            mean_v = sum(recent) / len(recent)
            var_v = sum((x - mean_v) ** 2 for x in recent) / len(recent)
            feats["vol_of_vol"] = sqrt(var_v)
        else:
            feats["vol_of_vol"] = None

        if n > 0 and close and close != 0 and vol5_val is not None and vol5_val > 1e-12:
            h = self.high_history[-1] if len(self.high_history) > 0 else close
            l = self.low_history[-1] if len(self.low_history) > 0 else close  # noqa: E741
            feats["range_vs_rv"] = ((h - l) / close) / vol5_val
        else:
            feats["range_vs_rv"] = None

        if self.hl_log_sq_window.full:
            mean_sq = self.hl_log_sq_window.mean
            if mean_sq is not None and mean_sq >= 0:
                feats["parkinson_vol"] = sqrt(mean_sq / (4.0 * math.log(2)))
            else:
                feats["parkinson_vol"] = None
        else:
            feats["parkinson_vol"] = None

        if len(self.vol_5_history) >= 6:
            feats["rv_acceleration"] = self.vol_5_history[-1] - self.vol_5_history[-6]
        else:
            feats["rv_acceleration"] = None

        # --- V5: Liquidation proxy ---
        if self._last_oi_change_pct is not None and self._prev_oi_change_for_accel is not None:
            feats["oi_acceleration"] = self._last_oi_change_pct - self._prev_oi_change_for_accel
        else:
            feats["oi_acceleration"] = None

        if self._last_oi is not None and close and close > 0 and self._last_volume > 0:
            raw_lev = self._last_oi / (close * self._last_volume)
            lev_ema = self.leverage_proxy_ema.value
            if self.leverage_proxy_ema.ready and lev_ema and lev_ema > 0:
                feats["leverage_proxy"] = raw_lev / lev_ema
            else:
                feats["leverage_proxy"] = None
        else:
            feats["leverage_proxy"] = None

        # OI up + volume down divergence
        oi_chg = self._last_oi_change_pct
        vol_r = feats.get("vol_ratio_20")
        if oi_chg is not None and vol_r is not None:
            feats["oi_vol_divergence"] = 1.0 if oi_chg > 0 and vol_r < 1.0 else 0.0
        else:
            feats["oi_vol_divergence"] = None

        # Large OI drop + volume spike → liquidation
        if oi_chg is not None and vol_r is not None:
            feats["oi_liquidation_flag"] = 1.0 if oi_chg < -0.05 and vol_r > 2.0 else 0.0
        else:
            feats["oi_liquidation_flag"] = None

        # --- V5: Funding carry ---
        if self._last_funding_rate is not None:
            feats["funding_annualized"] = self._last_funding_rate * 3.0 * 365.0
        else:
            feats["funding_annualized"] = None

        if self._last_funding_rate is not None and vol20_val is not None and vol20_val > 1e-12:
            feats["funding_vs_vol"] = self._last_funding_rate / vol20_val
        else:
            feats["funding_vs_vol"] = None

        # --- V7: Spot-futures basis ---
        feats["basis"] = self._last_basis
        if self.basis_window_24.full and self._last_basis is not None:
            b_mean = self.basis_window_24.mean
            b_std = self.basis_window_24.std
            if b_std is not None and b_std > 1e-12:
                zscore = (self._last_basis - b_mean) / b_std
                feats["basis_zscore_24"] = zscore
                feats["basis_extreme"] = 1.0 if zscore > 2.0 else (-1.0 if zscore < -2.0 else 0.0)
            else:
                feats["basis_zscore_24"] = None
                feats["basis_extreme"] = None
        else:
            feats["basis_zscore_24"] = None
            feats["basis_extreme"] = None

        basis_ema_val = self.basis_ema_8.value if self.basis_ema_8.ready else None
        if self._last_basis is not None and basis_ema_val is not None:
            feats["basis_momentum"] = self._last_basis - basis_ema_val
        else:
            feats["basis_momentum"] = None

        # --- V7: Fear & Greed Index ---
        if self._last_fgi is not None:
            feats["fgi_normalized"] = self._last_fgi / 100.0 - 0.5
            feats["fgi_extreme"] = (
                -1.0 if self._last_fgi < 25 else (1.0 if self._last_fgi > 75 else 0.0)
            )
        else:
            feats["fgi_normalized"] = None
            feats["fgi_extreme"] = None

        if self.fgi_window_7.full and self._last_fgi is not None:
            fgi_mean = self.fgi_window_7.mean
            fgi_std = self.fgi_window_7.std
            if fgi_std is not None and fgi_std > 1e-12:
                feats["fgi_zscore_7"] = (self._last_fgi - fgi_mean) / fgi_std
            else:
                feats["fgi_zscore_7"] = None
        else:
            feats["fgi_zscore_7"] = None

        # --- V8: Alpha Rebuild V3 features ---

        # taker_bq_ratio: USD-weighted buy pressure
        tbqv = self._last_taker_buy_quote_volume
        qv = self._last_quote_volume
        if tbqv > 0 and qv > 0:
            feats["taker_bq_ratio"] = tbqv / qv
        else:
            feats["taker_bq_ratio"] = None

        # vwap_dev_20: VWAP deviation (mean reversion signal)
        if self.vwap_cv_window.full and self.vwap_v_window.full and close and close > 0:
            sum_cv = self.vwap_cv_window.mean * self.vwap_cv_window.n
            sum_v = self.vwap_v_window.mean * self.vwap_v_window.n
            if sum_v > 0:
                vwap = sum_cv / sum_v
                feats["vwap_dev_20"] = (close - vwap) / close
            else:
                feats["vwap_dev_20"] = None
        else:
            feats["vwap_dev_20"] = None

        # volume_momentum_10: ret_10 × clip(volume/SMA_vol_20, 3.0)
        ret_10 = feats.get("ret_12")  # closest available; use close_history directly
        if n > 10 and hist[-11] != 0:
            ret_10 = (hist[-1] - hist[-11]) / hist[-11]
        else:
            ret_10 = None
        vol_r_20 = feats.get("vol_ratio_20")
        if ret_10 is not None and vol_r_20 is not None:
            feats["volume_momentum_10"] = ret_10 * min(vol_r_20, 3.0)
        else:
            feats["volume_momentum_10"] = None

        # mom_vol_divergence: price direction vs volume direction agreement
        ret1 = feats.get("ret_1")
        vol_r = feats.get("vol_ratio_20")
        if ret1 is not None and vol_r is not None:
            price_up = ret1 > 0
            vol_up = vol_r > 1.0
            feats["mom_vol_divergence"] = 1.0 if price_up == vol_up else -1.0
        else:
            feats["mom_vol_divergence"] = None

        # basis_carry_adj: basis + funding_rate × 3
        if self._last_basis is not None and self._last_funding_rate is not None:
            feats["basis_carry_adj"] = self._last_basis + self._last_funding_rate * 3.0
        else:
            feats["basis_carry_adj"] = None

        # vol_regime_adaptive: EMA(vol_regime,5) vs 30-bar median
        if self.vol_regime_ema.ready and len(self.vol_regime_history) >= 30:
            ema_val = self.vol_regime_ema.value
            if ema_val is None:
                feats["vol_regime_adaptive"] = None
            else:
                sorted_hist = sorted(self.vol_regime_history)
                median_val = sorted_hist[len(sorted_hist) // 2]
                if ema_val > median_val * 1.05:
                    feats["vol_regime_adaptive"] = 1.0
                elif ema_val < median_val * 0.95:
                    feats["vol_regime_adaptive"] = -1.0
                else:
                    feats["vol_regime_adaptive"] = 0.0
        else:
            feats["vol_regime_adaptive"] = None

        # --- V9: Cross-factor interaction features ---
        # liquidation_cascade_score: |oi_change_pct| * vol_ma_ratio_5_20
        oi_pct = feats.get("oi_change_pct")
        vmr = feats.get("vol_ma_ratio_5_20")
        if oi_pct is not None and vmr is not None:
            feats["liquidation_cascade_score"] = abs(oi_pct) * vmr
        else:
            feats["liquidation_cascade_score"] = None

        # funding_term_slope: normalized (funding_rate - funding_ma8)
        fr = feats.get("funding_rate")
        fma8 = feats.get("funding_ma8")
        if fr is not None and fma8 is not None:
            denom = max(abs(fma8), 1e-6)
            feats["funding_term_slope"] = (fr - fma8) / denom
        else:
            feats["funding_term_slope"] = None

        # cross_tf_regime_sync: sign(close_vs_ma20) == sign(tf4h_close_vs_ma20)
        # tf4h_close_vs_ma20 is computed externally (multi-timeframe aggregator), not available here
        feats["cross_tf_regime_sync"] = None

        # --- V9: Deribit IV features ---
        # implied_vol_zscore_24
        if self.iv_window_24.full and self._last_implied_vol is not None:
            iv_mean = self.iv_window_24.mean
            iv_std = self.iv_window_24.std
            if iv_std is not None and iv_std > 1e-8:
                feats["implied_vol_zscore_24"] = (self._last_implied_vol - iv_mean) / iv_std
            else:
                feats["implied_vol_zscore_24"] = None
        else:
            feats["implied_vol_zscore_24"] = None

        # iv_rv_spread: implied vol - realized vol (vol_20)
        vol20 = feats.get("vol_20")
        if self._last_implied_vol is not None and vol20 is not None:
            feats["iv_rv_spread"] = self._last_implied_vol - vol20
        else:
            feats["iv_rv_spread"] = None

        # put_call_ratio (passthrough from external feed)
        feats["put_call_ratio"] = self._last_put_call_ratio

        # --- V10: On-chain features ---
        # exchange_netflow_zscore: zscore_7d(inflow - outflow)
        if len(self._onchain_netflow_buf) >= 7:
            vals = list(self._onchain_netflow_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["exchange_netflow_zscore"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["exchange_netflow_zscore"] = None

        # exchange_supply_change: (today - yesterday) / yesterday
        if len(self._onchain_supply_buf) >= 2:
            prev = self._onchain_supply_buf[-2]
            curr = self._onchain_supply_buf[-1]
            feats["exchange_supply_change"] = (curr - prev) / prev if prev > 1e-8 else 0.0
        else:
            feats["exchange_supply_change"] = None

        # exchange_supply_zscore_30: zscore_30d(SplyExNtv)
        if len(self._onchain_supply_buf) >= 30:
            vals = list(self._onchain_supply_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["exchange_supply_zscore_30"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["exchange_supply_zscore_30"] = None

        # active_addr_zscore_14: zscore_14d(AdrActCnt)
        if len(self._onchain_addr_buf) >= 14:
            vals = list(self._onchain_addr_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["active_addr_zscore_14"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["active_addr_zscore_14"] = None

        # tx_count_zscore_14: zscore_14d(TxTfrCnt)
        if len(self._onchain_tx_buf) >= 14:
            vals = list(self._onchain_tx_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["tx_count_zscore_14"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["tx_count_zscore_14"] = None

        # hashrate_momentum: (HashRate - EMA14) / EMA14
        if self._onchain_hashrate_ema.ready and self._last_onchain_hashrate is not None:
            ema_val = self._onchain_hashrate_ema.value
            if ema_val is not None and abs(ema_val) > 1e-8:
                feats["hashrate_momentum"] = (self._last_onchain_hashrate - ema_val) / ema_val
            else:
                feats["hashrate_momentum"] = None
        else:
            feats["hashrate_momentum"] = None

        # --- V11: Liquidation features ---
        # liquidation_volume_zscore_24
        if len(self._liq_volume_buf) >= 24:
            vals = list(self._liq_volume_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["liquidation_volume_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["liquidation_volume_zscore_24"] = None

        # liquidation_imbalance
        feats["liquidation_imbalance"] = self._last_liq_imbalance

        # liquidation_volume_ratio: liq_vol / quote_vol
        if self._last_liq_volume is not None and self._last_quote_volume > 0:
            feats["liquidation_volume_ratio"] = self._last_liq_volume / self._last_quote_volume
        else:
            feats["liquidation_volume_ratio"] = None

        # liquidation_cluster_flag: >3 std liq volume in short window (6 bars)
        if len(self._liq_imbalance_buf) >= 6 and len(self._liq_volume_buf) >= 6:
            recent = list(self._liq_volume_buf)[-6:]
            mean = sum(recent) / len(recent)
            var = sum((v - mean) ** 2 for v in recent) / len(recent)
            std = sqrt(var) if var > 0 else 0.0
            if std > 1e-8 and recent[-1] > mean + 3.0 * std:
                feats["liquidation_cluster_flag"] = 1.0
            else:
                feats["liquidation_cluster_flag"] = 0.0
        else:
            feats["liquidation_cluster_flag"] = None

        # --- V11: Mempool features ---
        if len(self._mempool_fee_buf) >= 24:
            vals = list(self._mempool_fee_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["mempool_fee_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["mempool_fee_zscore_24"] = None

        if len(self._mempool_size_buf) >= 24:
            vals = list(self._mempool_size_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["mempool_size_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["mempool_size_zscore_24"] = None

        feats["fee_urgency_ratio"] = self._last_fee_urgency

        # --- V11: Macro features ---
        # dxy_change_5d
        if len(self._dxy_buf) >= 6:
            feats["dxy_change_5d"] = (
                (self._dxy_buf[-1] - self._dxy_buf[-6]) / self._dxy_buf[-6]
                if self._dxy_buf[-6] > 1e-8 else 0.0
            )
        else:
            feats["dxy_change_5d"] = None

        # spx_btc_corr_30d
        if len(self._spx_buf) >= 10 and len(self._btc_close_buf_30) >= 10:
            n = min(len(self._spx_buf), len(self._btc_close_buf_30))
            spx_vals = list(self._spx_buf)[-n:]
            btc_vals = list(self._btc_close_buf_30)[-n:]
            # Compute returns
            if n >= 2:
                spx_rets = [(spx_vals[i] - spx_vals[i-1]) / spx_vals[i-1] if spx_vals[i-1] > 0 else 0.0 for i in
                    range(1, n)]
                btc_rets = [(btc_vals[i] - btc_vals[i-1]) / btc_vals[i-1] if btc_vals[i-1] > 0 else 0.0 for i in
                    range(1, n)]
                m = len(spx_rets)
                if m >= 5:
                    mean_s = sum(spx_rets) / m
                    mean_b = sum(btc_rets) / m
                    cov = sum((spx_rets[i] - mean_s) * (btc_rets[i] - mean_b) for i in range(m)) / m
                    var_s = sum((r - mean_s) ** 2 for r in spx_rets) / m
                    var_b = sum((r - mean_b) ** 2 for r in btc_rets) / m
                    denom = sqrt(var_s * var_b)
                    feats["spx_btc_corr_30d"] = cov / denom if denom > 1e-8 else 0.0
                else:
                    feats["spx_btc_corr_30d"] = None
            else:
                feats["spx_btc_corr_30d"] = None
        else:
            feats["spx_btc_corr_30d"] = None

        # spx_overnight_ret
        if self._last_spx_close is not None and self._prev_spx_close is not None and self._prev_spx_close > 0:
            feats["spx_overnight_ret"] = (self._last_spx_close - self._prev_spx_close) / self._prev_spx_close
        else:
            feats["spx_overnight_ret"] = None

        # vix_zscore_14
        if len(self._vix_buf) >= 14:
            vals = list(self._vix_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["vix_zscore_14"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["vix_zscore_14"] = None

        # --- V11: Social sentiment features ---
        if len(self._social_vol_buf) >= 24:
            vals = list(self._social_vol_buf)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = sqrt(var) if var > 0 else 0.0
            feats["social_volume_zscore_24"] = (vals[-1] - mean) / std if std > 1e-8 else 0.0
        else:
            feats["social_volume_zscore_24"] = None

        feats["social_sentiment_score"] = self._last_sentiment_score

        # social_volume_price_div: social volume up + price down (or vice versa) → divergence
        if self._last_social_volume is not None and len(self._social_vol_buf) >= 2 and len(self.close_history) >= 2:
            sv_change = self._social_vol_buf[-1] - self._social_vol_buf[-2]
            price_change = self.close_history[-1] - self.close_history[-2]
            if (sv_change > 0 and price_change < 0) or (sv_change < 0 and price_change > 0):
                feats["social_volume_price_div"] = 1.0
            else:
                feats["social_volume_price_div"] = 0.0
        else:
            feats["social_volume_price_div"] = None

        # ── V12: ALT cross-asset features (requires btc_close parameter) ──
        if btc_close is not None and btc_close > 0:
            self._btc_ref_buf.append(btc_close)

            # BTC ret_1 (lagged BTC return as lead signal for ALT)
            if len(self._btc_ref_buf) >= 2:
                feats["btc_lead_ret_1"] = self._btc_ref_buf[-1] / self._btc_ref_buf[-2] - 1
            else:
                feats["btc_lead_ret_1"] = None

            # BTC vol_20 for relative vol
            if len(self._btc_ref_buf) >= 3:
                btc_rets = []
                for j in range(1, min(21, len(self._btc_ref_buf))):
                    btc_rets.append(self._btc_ref_buf[-j] / self._btc_ref_buf[-j - 1] - 1)
                if len(btc_rets) >= 5:
                    _m = sum(btc_rets) / len(btc_rets)
                    btc_vol = sqrt(sum((r - _m) ** 2 for r in btc_rets) / len(btc_rets))
                else:
                    btc_vol = None
                self._btc_ref_vol_buf.append(btc_vol if btc_vol else 0)
            else:
                btc_vol = None

            # ALT/BTC ratio tracking
            if close is not None and close > 0:
                ratio = close / btc_close
                self._alt_btc_ratio_buf.append(ratio)

            # btc_relative_strength_24: ALT ret_24 - BTC ret_24
            if close is not None and len(self.close_history) >= 24 and len(self._btc_ref_buf) >= 24:
                alt_ret24 = close / self.close_history[-24] - 1
                btc_ret24 = self._btc_ref_buf[-1] / self._btc_ref_buf[-24] - 1
                feats["btc_relative_strength_24"] = alt_ret24 - btc_ret24
            else:
                feats["btc_relative_strength_24"] = None

            # btc_relative_strength_6
            if close is not None and len(self.close_history) >= 6 and len(self._btc_ref_buf) >= 6:
                alt_ret6 = close / self.close_history[-6] - 1
                btc_ret6 = self._btc_ref_buf[-1] / self._btc_ref_buf[-6] - 1
                feats["btc_relative_strength_6"] = alt_ret6 - btc_ret6
            else:
                feats["btc_relative_strength_6"] = None

            # btc_ratio_ma20_dev: (ratio / MA20_ratio) - 1
            if len(self._alt_btc_ratio_buf) >= 20:
                ratio_ma = sum(list(self._alt_btc_ratio_buf)[-20:]) / 20
                if ratio_ma > 0:
                    feats["btc_ratio_ma20_dev"] = self._alt_btc_ratio_buf[-1] / ratio_ma - 1
                else:
                    feats["btc_ratio_ma20_dev"] = None
            else:
                feats["btc_ratio_ma20_dev"] = None

            # btc_dom_momentum: BTC ret_24 - ALT ret_24 (opposite of relative strength)
            _rs24 = feats.get("btc_relative_strength_24")
            feats["btc_dom_momentum"] = -float(_rs24) if _rs24 is not None else None

            # btc_vol_ratio: ALT vol_20 / BTC vol_20
            vol20 = feats.get("vol_20")
            if vol20 is not None and btc_vol is not None and btc_vol > 1e-8:
                feats["btc_vol_ratio"] = vol20 / btc_vol
            else:
                feats["btc_vol_ratio"] = None
        else:
            feats["btc_relative_strength_24"] = None
            feats["btc_relative_strength_6"] = None
            feats["btc_ratio_ma20_dev"] = None
            feats["btc_dom_momentum"] = None
            feats["btc_lead_ret_1"] = None
            feats["btc_vol_ratio"] = None

        return feats


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
    ) -> Dict[str, Optional[float]]:
        """Process a new bar and return computed features.

        btc_close: BTC price at same bar time. Required for V12 ALT cross-asset features.
        top_trader_ls_ratio: Top trader position L/S ratio (V13).
        eth_close: ETH price at same bar time. Required for V14 BTC dominance features.
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
                   sentiment_metrics=sentiment_metrics)
        return state.get_features(btc_close=btc_close)

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

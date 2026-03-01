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

    # Acceleration: track recent momentum values
    _prev_momentum: Optional[float] = None
    _last_close: Optional[float] = None
    _last_volume: float = 0.0
    _last_hour: int = -1
    _last_dow: int = -1
    _last_funding_rate: Optional[float] = None
    _last_trades: float = 0.0
    _last_taker_buy_volume: float = 0.0
    _last_quote_volume: float = 0.0
    _bar_count: int = 0

    def push(self, close: float, volume: float, high: float, low: float, open_: float,
             *, hour: int = -1, dow: int = -1, funding_rate: Optional[float] = None,
             trades: float = 0.0, taker_buy_volume: float = 0.0,
             quote_volume: float = 0.0,
             open_interest: Optional[float] = None,
             ls_ratio: Optional[float] = None) -> None:
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
                self._last_oi_change_pct = change
                self.oi_change_ema_8.push(change)
            self._last_oi = open_interest

        # LS Ratio state
        if ls_ratio is not None:
            self._last_ls_ratio = ls_ratio
            self.ls_ratio_window_24.push(ls_ratio)

        # Microstructure state
        self._last_trades = trades
        self._last_taker_buy_volume = taker_buy_volume
        self._last_quote_volume = quote_volume
        if trades > 0:
            self.trades_ema_20.push(trades)
            self.trades_ema_5.push(trades)
            # Taker buy ratio
            tbr = taker_buy_volume / volume if volume > 0 else 0.5
            self.taker_buy_ratio_ema_10.push(tbr)
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

    def get_features(self) -> Dict[str, Optional[float]]:
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
            l = self.low_history[-1]
            c = hist[-1]
            hl_range = h - l
            if hl_range > 0:
                feats["body_ratio"] = (c - o) / hl_range
                feats["upper_shadow"] = (h - max(o, c)) / hl_range
                feats["lower_shadow"] = (min(o, c) - l) / hl_range
            else:
                feats["body_ratio"] = 0.0
                feats["upper_shadow"] = 0.0
                feats["lower_shadow"] = 0.0
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
        open_interest: Optional[float] = None,
        ls_ratio: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """Process a new bar and return computed features.

        Args:
            hour: Hour of day (0-23 UTC). -1 if unknown.
            dow: Day of week (0=Mon, 6=Sun). -1 if unknown.
            funding_rate: Current funding rate (e.g. 0.0001). None if unknown.
            trades: Number of trades in bar.
            taker_buy_volume: Taker buy base volume in bar.
            quote_volume: Total quote volume in bar.
            open_interest: Current open interest (contracts). None if unknown.
            ls_ratio: Long/short account ratio. None if unknown.
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
                   open_interest=open_interest, ls_ratio=ls_ratio)
        return state.get_features()

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

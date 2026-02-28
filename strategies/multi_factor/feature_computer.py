from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Optional


@dataclass
class MultiFactorFeatures:
    sma_fast: Optional[float]
    sma_slow: Optional[float]
    sma_trend: Optional[float]  # SMA(200) long-term trend filter
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    bb_upper: Optional[float]
    bb_middle: Optional[float]
    bb_lower: Optional[float]
    bb_pct: Optional[float]
    atr: Optional[float]
    atr_pct: Optional[float]
    atr_percentile: Optional[float]
    ma_slope: Optional[float]
    close: float
    volume: float


class _RollingSum:
    """O(1) rolling sum/mean/std via deque."""

    __slots__ = ("_window", "_buf", "_sum", "_sumsq")

    def __init__(self, window: int) -> None:
        self._window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sumsq = 0.0

    def push(self, x: float) -> None:
        if len(self._buf) == self._window:
            old = self._buf[0]
            self._sum -= old
            self._sumsq -= old * old
        self._buf.append(x)
        self._sum += x
        self._sumsq += x * x

    @property
    def full(self) -> bool:
        return len(self._buf) == self._window

    @property
    def mean(self) -> Optional[float]:
        if not self.full:
            return None
        return self._sum / self._window

    @property
    def std(self) -> Optional[float]:
        if not self.full:
            return None
        var = self._sumsq / self._window - (self._sum / self._window) ** 2
        return sqrt(max(var, 0.0))

    def values(self) -> deque[float]:
        return self._buf


class MultiFactorFeatureComputer:
    def __init__(
        self,
        *,
        sma_fast_window: int = 20,
        sma_slow_window: int = 50,
        sma_trend_window: int = 200,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_window: int = 20,
        bb_std: float = 2.0,
        atr_window: int = 14,
        atr_pct_window: int = 100,
        ma_slope_window: int = 10,
    ) -> None:
        self._sma_fast_w = sma_fast_window
        self._sma_slow_w = sma_slow_window
        self._sma_trend_w = sma_trend_window
        self._rsi_w = rsi_window
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal_w = macd_signal
        self._bb_w = bb_window
        self._bb_std = bb_std
        self._atr_w = atr_window
        self._atr_pct_w = atr_pct_window
        self._ma_slope_w = ma_slope_window

        self.reset()

    def reset(self) -> None:
        # SMA
        self._sma_fast = _RollingSum(self._sma_fast_w)
        self._sma_slow = _RollingSum(self._sma_slow_w)
        self._sma_trend = _RollingSum(self._sma_trend_w)

        # RSI (Wilder smoothing)
        self._rsi_avg_gain: Optional[float] = None
        self._rsi_avg_loss: Optional[float] = None
        self._rsi_count = 0
        self._prev_close: Optional[float] = None

        # MACD EMAs
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._ema_signal: Optional[float] = None
        self._ema_fast_alpha = 2.0 / (self._macd_fast + 1)
        self._ema_slow_alpha = 2.0 / (self._macd_slow + 1)
        self._ema_signal_alpha = 2.0 / (self._macd_signal_w + 1)
        self._macd_count = 0

        # Bollinger Bands
        self._bb = _RollingSum(self._bb_w)

        # ATR (Wilder smoothing)
        self._atr_val: Optional[float] = None
        self._atr_sum = 0.0
        self._atr_count = 0
        self._atr_history: deque[float] = deque(maxlen=self._atr_pct_w)

        # MA slope
        self._sma_slow_history: deque[float] = deque(maxlen=self._ma_slope_w)

        self._bar_count = 0

    def on_bar(
        self,
        *,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> MultiFactorFeatures:
        self._bar_count += 1

        # --- SMA ---
        self._sma_fast.push(close)
        self._sma_slow.push(close)
        self._sma_trend.push(close)
        sma_fast_val = self._sma_fast.mean
        sma_slow_val = self._sma_slow.mean
        sma_trend_val = self._sma_trend.mean

        # --- RSI (Wilder) ---
        rsi_val: Optional[float] = None
        if self._prev_close is not None:
            change = close - self._prev_close
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            self._rsi_count += 1

            if self._rsi_count < self._rsi_w:
                if self._rsi_avg_gain is None:
                    self._rsi_avg_gain = 0.0
                    self._rsi_avg_loss = 0.0
                self._rsi_avg_gain += gain
                self._rsi_avg_loss += loss
            elif self._rsi_count == self._rsi_w:
                assert self._rsi_avg_gain is not None and self._rsi_avg_loss is not None
                self._rsi_avg_gain = self._rsi_avg_gain / self._rsi_w
                self._rsi_avg_loss = self._rsi_avg_loss / self._rsi_w
                if self._rsi_avg_loss == 0.0:
                    rsi_val = 100.0
                else:
                    rs = self._rsi_avg_gain / self._rsi_avg_loss
                    rsi_val = 100.0 - (100.0 / (1.0 + rs))
            else:
                assert self._rsi_avg_gain is not None and self._rsi_avg_loss is not None
                self._rsi_avg_gain = (self._rsi_avg_gain * (self._rsi_w - 1) + gain) / self._rsi_w
                self._rsi_avg_loss = (self._rsi_avg_loss * (self._rsi_w - 1) + loss) / self._rsi_w
                if self._rsi_avg_loss == 0.0:
                    rsi_val = 100.0
                else:
                    rs = self._rsi_avg_gain / self._rsi_avg_loss
                    rsi_val = 100.0 - (100.0 / (1.0 + rs))
        self._prev_close = close

        # --- MACD ---
        macd_val: Optional[float] = None
        macd_signal_val: Optional[float] = None
        macd_hist_val: Optional[float] = None

        if self._ema_fast is None:
            self._ema_fast = close
        else:
            self._ema_fast = self._ema_fast_alpha * close + (1 - self._ema_fast_alpha) * self._ema_fast

        if self._ema_slow is None:
            self._ema_slow = close
        else:
            self._ema_slow = self._ema_slow_alpha * close + (1 - self._ema_slow_alpha) * self._ema_slow

        self._macd_count += 1
        macd_line = self._ema_fast - self._ema_slow

        if self._ema_signal is None:
            self._ema_signal = macd_line
        else:
            self._ema_signal = self._ema_signal_alpha * macd_line + (1 - self._ema_signal_alpha) * self._ema_signal

        # MACD valid after slow EMA has warmed up enough
        if self._macd_count >= self._macd_slow + self._macd_signal_w - 1:
            macd_val = macd_line
            macd_signal_val = self._ema_signal
            macd_hist_val = macd_line - self._ema_signal

        # --- Bollinger Bands ---
        self._bb.push(close)
        bb_upper: Optional[float] = None
        bb_middle: Optional[float] = None
        bb_lower: Optional[float] = None
        bb_pct: Optional[float] = None

        if self._bb.full:
            mid = self._bb.mean
            std = self._bb.std
            if mid is not None and std is not None:
                bb_middle = mid
                bb_upper = mid + self._bb_std * std
                bb_lower = mid - self._bb_std * std
                width = bb_upper - bb_lower
                bb_pct = (close - bb_lower) / width if width > 0 else 0.5

        # --- ATR (Wilder smoothing) ---
        atr_val: Optional[float] = None
        if self._prev_close is not None and self._bar_count > 1:
            prev_c = self._prev_close  # already updated above, need the one before
        # Use the deque approach: track prev close separately for TR
        # Actually we already overwrite _prev_close. Let's fix: we need prev close *before* this bar.
        # We'll track a separate _prev_close_for_atr
        # Simpler: compute TR here using the passed-in data
        if self._bar_count == 1:
            tr = high - low
        else:
            # _atr_prev_close was set at end of previous call
            prev_c = getattr(self, "_atr_prev_close", close)
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))

        self._atr_count += 1
        if self._atr_count <= self._atr_w:
            self._atr_sum += tr
        elif self._atr_count == self._atr_w + 1:
            # First ATR: average of first _atr_w TRs (matches batch behavior)
            self._atr_val = self._atr_sum / self._atr_w
            atr_val = self._atr_val
        elif self._atr_val is not None:
            self._atr_val = (self._atr_val * (self._atr_w - 1) + tr) / self._atr_w
            atr_val = self._atr_val

        self._atr_prev_close = close

        # ATR percentile
        atr_pct_val: Optional[float] = None
        atr_percentile: Optional[float] = None
        if atr_val is not None:
            self._atr_history.append(atr_val)
            atr_pct_val = atr_val / close if close > 0 else None

            if len(self._atr_history) >= 20:  # need some history
                sorted_atrs = sorted(self._atr_history)
                rank = sum(1 for a in sorted_atrs if a <= atr_val)
                atr_percentile = 100.0 * rank / len(self._atr_history)

        # --- MA slope ---
        ma_slope: Optional[float] = None
        if sma_slow_val is not None:
            self._sma_slow_history.append(sma_slow_val)
            if len(self._sma_slow_history) >= self._ma_slope_w:
                old_ma = self._sma_slow_history[0]
                if old_ma != 0:
                    ma_slope = (sma_slow_val - old_ma) / (old_ma * self._ma_slope_w)

        return MultiFactorFeatures(
            sma_fast=sma_fast_val,
            sma_slow=sma_slow_val,
            sma_trend=sma_trend_val,
            rsi=rsi_val,
            macd=macd_val,
            macd_signal=macd_signal_val,
            macd_hist=macd_hist_val,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_pct=bb_pct,
            atr=atr_val,
            atr_pct=atr_pct_val,
            atr_percentile=atr_percentile,
            ma_slope=ma_slope,
            close=close,
            volume=volume,
        )

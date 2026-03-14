# features/live_computer.py
"""LiveFeatureComputer — incremental feature computation for live trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from features.rolling import RollingWindow

logger = logging.getLogger(__name__)


@dataclass
class _SymbolBuffer:
    """Per-symbol bar buffer with incremental feature windows."""
    close_window: RollingWindow
    volume_window: RollingWindow
    return_window: RollingWindow
    fast_ma_window: RollingWindow
    slow_ma_window: RollingWindow
    _last_close: Optional[float] = None

    def push(self, close: float, volume: float = 0.0) -> None:
        self.close_window.push(close)
        self.volume_window.push(volume)

        # Compute return
        if self._last_close is not None and self._last_close != 0:
            ret = (close - self._last_close) / self._last_close
            self.return_window.push(ret)
        self._last_close = close

        # MA windows
        self.fast_ma_window.push(close)
        self.slow_ma_window.push(close)


@dataclass(frozen=True, slots=True)
class LiveFeatures:
    """Computed features for a single symbol at a point in time."""
    symbol: str
    close: float
    volume: float
    ma_fast: Optional[float]
    ma_slow: Optional[float]
    volatility: Optional[float]
    momentum: Optional[float]
    vwap_ratio: Optional[float]
    return_last: Optional[float]


@dataclass
class LiveFeatureComputer:
    """Incremental feature computer for live trading.

    Maintains per-symbol rolling windows and computes features on each new bar.
    All windows are O(1) amortized via RollingWindow.

    Usage:
        computer = LiveFeatureComputer(fast_ma=10, slow_ma=30, vol_window=20)
        features = computer.on_bar("BTCUSDT", close=40000.0, volume=100.0)
        if features.ma_fast is not None:
            # use features
    """

    fast_ma: int = 10
    slow_ma: int = 30
    vol_window: int = 20
    max_buffer: int = 200

    _buffers: Dict[str, _SymbolBuffer] = field(default_factory=dict, init=False)
    _vwap_num: Dict[str, float] = field(default_factory=dict, init=False)
    _vwap_den: Dict[str, float] = field(default_factory=dict, init=False)

    def on_bar(
        self,
        symbol: str,
        *,
        close: float,
        volume: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
    ) -> LiveFeatures:
        """Process a new bar and return computed features."""
        if symbol not in self._buffers:
            self._buffers[symbol] = _SymbolBuffer(
                close_window=RollingWindow(self.max_buffer),
                volume_window=RollingWindow(self.max_buffer),
                return_window=RollingWindow(self.vol_window),
                fast_ma_window=RollingWindow(self.fast_ma),
                slow_ma_window=RollingWindow(self.slow_ma),
            )
            self._vwap_num[symbol] = 0.0
            self._vwap_den[symbol] = 0.0

        buf = self._buffers[symbol]
        buf.push(close, volume)

        # VWAP (cumulative)
        self._vwap_num[symbol] += close * volume
        self._vwap_den[symbol] += volume

        # Extract features
        ma_fast = buf.fast_ma_window.mean if buf.fast_ma_window.full else None
        ma_slow = buf.slow_ma_window.mean if buf.slow_ma_window.full else None

        # Volatility (std of returns)
        volatility = buf.return_window.std if buf.return_window.full else None

        # Momentum (fast_ma / slow_ma - 1)
        momentum = None
        if ma_fast is not None and ma_slow is not None and ma_slow != 0:
            momentum = ma_fast / ma_slow - 1.0

        # VWAP ratio
        vwap_ratio = None
        vwap_den = self._vwap_den[symbol]
        if vwap_den > 0:
            vwap_val = self._vwap_num[symbol] / vwap_den
            if vwap_val > 0:
                vwap_ratio = close / vwap_val

        # Last return
        return_last = None
        if buf.return_window.n > 0:
            return_last = buf.return_window.mean

        return LiveFeatures(
            symbol=symbol,
            close=close,
            volume=volume,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            volatility=volatility,
            momentum=momentum,
            vwap_ratio=vwap_ratio,
            return_last=return_last,
        )

    def get_features_dict(self, symbol: str) -> Dict[str, Optional[float]]:
        """Get last computed features as a flat dict (for signal models)."""
        if symbol not in self._buffers:
            return {}
        buf = self._buffers[symbol]
        ma_fast = buf.fast_ma_window.mean if buf.fast_ma_window.full else None
        ma_slow = buf.slow_ma_window.mean if buf.slow_ma_window.full else None
        vol = buf.return_window.std if buf.return_window.full else None
        momentum = None
        if ma_fast is not None and ma_slow is not None and ma_slow != 0:
            momentum = ma_fast / ma_slow - 1.0
        return {
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "vol": vol,
            "momentum": momentum,
        }

    @property
    def symbols(self) -> List[str]:
        return list(self._buffers.keys())

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset buffers for a symbol or all symbols."""
        if symbol is not None:
            self._buffers.pop(symbol, None)
            self._vwap_num.pop(symbol, None)
            self._vwap_den.pop(symbol, None)
        else:
            self._buffers.clear()
            self._vwap_num.clear()
            self._vwap_den.clear()

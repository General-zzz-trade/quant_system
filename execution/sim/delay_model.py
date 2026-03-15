# execution/sim/delay_model.py
"""Signal-to-execution delay model for realistic backtesting.

Models the real-world delay chain:
  Signal generated (bar close) → order submitted → order accepted → fill

In 1h bar backtesting:
  - Minimum delay: signal at bar N close → fill at bar N+1 open
  - The embargo.py already handles this (1-bar delay)
  - This module adds configurable sub-bar delays for higher fidelity

For the realistic_backtest engine, delay manifests as:
  1. Signal shift: signal[i] → applied at bar[i + delay_bars]
  2. Fill price slippage: fill at open + random noise (not close)
"""
from __future__ import annotations

import numpy as np


def apply_signal_delay(signal: np.ndarray, delay_bars: int = 1) -> np.ndarray:
    """Shift signal by N bars to simulate execution delay.

    delay_bars=0: signal acts on same bar (unrealistic, but matches some backtests)
    delay_bars=1: signal at bar N → position at bar N+1 (standard realistic)
    delay_bars=2: extra delay (conservative, accounts for network + matching)
    """
    if delay_bars <= 0:
        return signal.copy()
    delayed = np.zeros_like(signal)
    delayed[delay_bars:] = signal[:-delay_bars]
    return delayed


def apply_fill_price_noise(
    closes: np.ndarray,
    opens: np.ndarray,
    signal: np.ndarray,
    noise_bps: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate realistic fill prices with noise around the open.

    When signal changes at bar i, fill happens at bar i's open + noise.
    Noise simulates market impact and order book dynamics.

    Returns array of fill prices (same length as closes).
    """
    rng = np.random.RandomState(seed)
    n = len(closes)
    fill_prices = closes.copy()  # default: close price

    for i in range(1, n):
        if signal[i] != signal[i - 1]:
            # Signal change → fill at open + noise
            noise = rng.uniform(-noise_bps, noise_bps) / 10000
            fill_prices[i] = opens[i] * (1 + noise)

    return fill_prices

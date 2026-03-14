"""Runner for 5-minute Polymarket RSI strategy.

Entry point: python3 -m polymarket.strategies.runner_5m [--dry-run]

Connects to Binance 1m BTC/USDT websocket for real-time RSI,
discovers current Polymarket 5m BTC up/down market,
and places bets when RSI is extreme.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from polymarket.strategies.rsi_5m import RSI5mStrategy

logger = logging.getLogger(__name__)


@dataclass
class Runner5mConfig:
    """Config for 5-minute RSI runner."""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    bet_size_usd: float = 10.0
    rsi_period: int = 5
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    dry_run: bool = True
    log_level: str = "INFO"


class PolymarketRSI5mRunner:
    """Runs RSI(5) strategy on Polymarket 5-minute BTC markets."""

    def __init__(self, config: Runner5mConfig) -> None:
        self._config = config
        self._strategy = RSI5mStrategy(
            period=config.rsi_period,
            oversold=config.rsi_oversold,
            overbought=config.rsi_overbought,
        )
        self._running = False
        self._total_bets = 0
        self._total_pnl = 0.0

    def on_kline(self, close: float, timestamp_ms: int = 0) -> Optional[dict]:
        """Process a 1-minute kline close. Returns bet dict if signal generated."""
        signal = self._strategy.update(close, timestamp_ms)
        if signal is None:
            return None

        bet = {
            "direction": signal.direction,
            "rsi": signal.rsi_value,
            "confidence": signal.confidence,
            "size_usd": self._config.bet_size_usd,
            "timestamp_ms": timestamp_ms,
        }

        if self._config.dry_run:
            logger.info(
                "DRY_RUN BET: %s RSI=%.1f conf=%.2f size=$%.2f",
                signal.direction.upper(), signal.rsi_value,
                signal.confidence, self._config.bet_size_usd,
            )
        else:
            logger.info(
                "LIVE BET: %s RSI=%.1f conf=%.2f size=$%.2f",
                signal.direction.upper(), signal.rsi_value,
                signal.confidence, self._config.bet_size_usd,
            )
            # TODO: Place actual Polymarket order via CLOB API

        self._total_bets += 1
        return bet

    def run_backtest(self, closes: Any, timestamps: Any = None) -> dict:
        """Run strategy on historical data. Returns performance summary."""
        import numpy as np
        closes = np.asarray(closes, dtype=float)
        n = len(closes)

        bets = []
        wins = 0
        pnl = 0.0

        for i in range(n):
            ts = int(timestamps[i]) if timestamps is not None and i < len(timestamps) else 0
            signal = self._strategy.update(float(closes[i]), ts)

            if signal is not None:
                # Check result: did the next 5-min window go in predicted direction?
                # Find next 5-min boundary
                window_start = ((i // 5) + 1) * 5
                window_end = window_start + 4

                if window_end < n:
                    actual_up = closes[window_end] >= closes[window_start]
                    predicted_up = signal.direction == "up"
                    won = (predicted_up == actual_up)

                    if predicted_up:
                        bet_pnl = 0.495 if won else -0.505
                    else:
                        bet_pnl = 0.505 if won else -0.495

                    pnl += bet_pnl * self._config.bet_size_usd
                    if won:
                        wins += 1
                    bets.append({
                        "bar": i, "direction": signal.direction,
                        "rsi": signal.rsi_value, "won": won, "pnl": bet_pnl,
                    })

        total = len(bets)
        return {
            "total_bets": total,
            "wins": wins,
            "accuracy": wins / total if total > 0 else 0,
            "total_pnl": round(pnl, 2),
            "pnl_per_bet": round(pnl / total, 4) if total > 0 else 0,
            "days": n / (24 * 60),
            "bets_per_day": total / max(n / (24 * 60), 1),
        }

    @property
    def stats(self) -> dict:
        return {"total_bets": self._total_bets, "total_pnl": self._total_pnl}

    def stop(self) -> None:
        self._running = False

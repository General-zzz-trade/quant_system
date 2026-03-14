"""PolymarketDecisionModule -- entry/exit logic for prediction markets."""
from __future__ import annotations
from decimal import Decimal
from typing import Any, Dict, List
from polymarket.config import PolymarketConfig
from polymarket.signals import generate_signal
from polymarket.sizing import kelly_size


class PolymarketDecisionModule:
    def __init__(self, config: PolymarketConfig) -> None:
        self._config = config

    def evaluate(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate a single market for trading opportunity."""
        features = market_data.get("features", {})
        market_price = float(market_data.get("market_price", 0.5))
        bankroll = float(market_data.get("bankroll", 0))
        symbol = market_data.get("symbol", "")

        # Skip extreme probabilities
        if market_price < self._config.min_probability or market_price > self._config.max_probability:
            return []

        signal = generate_signal(features, threshold=self._config.signal_threshold)
        if signal == 0.0:
            return []

        # Estimate true probability from signal
        if signal > 0:
            estimated_prob = min(0.95, market_price + abs(signal) * 0.15)
        else:
            estimated_prob = max(0.05, market_price - abs(signal) * 0.15)

        size = kelly_size(
            estimated_prob=estimated_prob if signal > 0 else 1 - estimated_prob,
            market_price=market_price if signal > 0 else 1 - market_price,
            bankroll=bankroll,
            kelly_fraction=self._config.kelly_fraction,
            max_position_pct=self._config.max_position_pct,
        )

        if size < 1.0:  # minimum $1
            return []

        return [{
            "symbol": symbol,
            "side": "buy" if signal > 0 else "sell",
            "outcome": "YES" if signal > 0 else "NO",
            "size": size,
            "signal_strength": abs(signal),
            "estimated_prob": estimated_prob,
            "market_price": market_price,
        }]

    def check_exits(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check existing positions for exit conditions."""
        exits = []
        for pos in positions:
            entry = float(pos.get("entry_price", 0.5))
            current = float(pos.get("current_price", 0.5))
            move = (current - entry) / entry if entry > 0 else 0

            # Stop loss
            if move < -self._config.stop_loss_pct:
                exits.append({"symbol": pos["symbol"], "action": "close",
                             "reason": f"stop_loss: {move:.1%}", "qty": pos.get("qty")})
        return exits

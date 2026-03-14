"""PolymarketRunner -- independent runner for prediction market trading."""
from __future__ import annotations
import logging, time
from typing import Any, Optional
from polymarket.config import PolymarketConfig
from polymarket.decision import PolymarketDecisionModule

logger = logging.getLogger(__name__)


class PolymarketRunner:
    def __init__(self, config: PolymarketConfig) -> None:
        self._config = config
        self._decision = PolymarketDecisionModule(config)
        self._client: Any = None  # Set externally or created in start()
        self._running = False
        self._active_markets: list = []

    def run_once(self) -> None:
        """Single cycle: scan -> features -> decide -> execute."""
        if self._client is None:
            from execution.adapters.polymarket.client import PolymarketClient
            from execution.adapters.polymarket.auth import PolymarketAuth
            auth = PolymarketAuth(self._config.api_key, self._config.api_secret)
            self._client = PolymarketClient(auth=auth, base_url=self._config.base_url)

        # 1. Discover markets
        try:
            from execution.adapters.polymarket.market_discovery import filter_crypto_markets
            all_markets = self._client.get_markets()
            self._active_markets = filter_crypto_markets(
                all_markets, self._config.crypto_keywords,
                min_volume=self._config.min_liquidity_usd)
            logger.info("Polymarket: %d active crypto markets", len(self._active_markets))
        except Exception:
            logger.warning("Market scan failed", exc_info=True)
            return

        # 2-5. For each market: features -> signal -> size -> order
        for market in self._active_markets:
            try:
                self._process_market(market)
            except Exception:
                logger.warning("Failed to process %s", market.slug, exc_info=True)

    def _process_market(self, market: Any) -> None:
        """Process a single market."""
        import numpy as np
        from polymarket.features import compute_features

        token_id = market.token_ids[0] if market.token_ids else None
        if not token_id:
            return

        # Get orderbook
        try:
            ob = self._client.get_orderbook(token_id)
            orderbook = {"best_bid": float(ob.best_bid or 0), "best_ask": float(ob.best_ask or 0),
                        "bid_depth": 0, "ask_depth": 0}
        except Exception:
            orderbook = {"best_bid": 0, "best_ask": 0, "bid_depth": 0, "ask_depth": 0}

        # Simplified: use mid_price as probability history (single point in V1)
        mid = ob.mid_price if ob and ob.mid_price else None
        prob = float(mid) if mid else 0.5
        prob_history = np.array([prob])  # V1: single point, no history yet

        features = compute_features(
            prob_history, orderbook,
            {"count_1h": 0, "buy_volume": 0, "sell_volume": 0},
            market.hours_to_expiry("2026-03-14T12:00:00Z"),
            btc_price=0, btc_strike=0)

        # Evaluate
        from decimal import Decimal
        intents = self._decision.evaluate({
            "symbol": market.symbol("Yes"),
            "features": features,
            "market_price": Decimal(str(prob)),
            "bankroll": Decimal("1000"),
        })

        for intent in intents:
            logger.info("POLYMARKET_SIGNAL: %s %s size=$%.2f signal=%.2f",
                       intent["side"], intent["symbol"], intent["size"], intent["signal_strength"])

    def start(self) -> None:
        self._running = True
        logger.info("PolymarketRunner starting (interval=%ds)", self._config.scan_interval_sec)
        while self._running:
            self.run_once()
            time.sleep(self._config.scan_interval_sec)

    def stop(self) -> None:
        self._running = False
        logger.info("PolymarketRunner stopped")

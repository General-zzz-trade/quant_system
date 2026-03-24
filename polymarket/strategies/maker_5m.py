"""Avellaneda-Stoikov market-making strategy for Polymarket 5-minute binary markets.

Simplified A-S model adapted for binary outcome markets (prices in [0.01, 0.99]).
Computes optimal bid/ask quotes around a reservation price that adjusts for
inventory risk, volatility, and time remaining in the 5-minute window.

Optionally biased by an RSI signal to skew quotes toward the predicted direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from _quant_hotpath import RustMMQuote  # type: ignore[import-untyped]

# Rust market-maker quote type — available for Rust-native quoting
# in latency-sensitive market-making paths.
MMQuoteType = RustMMQuote


@dataclass(frozen=True)
class QuotePair:
    """Two-sided quote for a binary market."""

    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


# Price bounds for binary markets
_MIN_PRICE = 0.01
_MAX_PRICE = 0.99


class AvellanedaStoikovMaker:
    """Simplified Avellaneda-Stoikov market maker for binary prediction markets.

    Parameters
    ----------
    gamma : float
        Risk aversion parameter. Higher = tighter spreads, less inventory risk.
    kappa : float
        Order arrival intensity parameter (Poisson rate).
    max_inventory : float
        Maximum net inventory (YES - NO contracts) before quoting stops on one side.
    min_spread : float
        Minimum spread to maintain (protects against adverse selection).
    max_spread : float
        Maximum spread (prevents unreasonable quotes).
    order_size : float
        Default order size for each side.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        kappa: float = 1.5,
        max_inventory: float = 100.0,
        min_spread: float = 0.02,
        max_spread: float = 0.10,
        order_size: float = 10.0,
    ) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if min_spread < 0:
            raise ValueError(f"min_spread must be non-negative, got {min_spread}")
        if max_spread < min_spread:
            raise ValueError(f"max_spread ({max_spread}) < min_spread ({min_spread})")

        self.gamma = gamma
        self.kappa = kappa
        self.max_inventory = max_inventory
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.order_size = order_size

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: float,
        time_remaining: float,
    ) -> QuotePair:
        """Compute optimal bid/ask quotes.

        Parameters
        ----------
        mid_price : float
            Current fair-value mid price (0.01-0.99 for binary markets).
        inventory : float
            Net inventory (positive = long YES, negative = long NO).
        volatility : float
            Estimated price volatility (annualized or per-window, be consistent).
        time_remaining : float
            Fraction of time remaining in the market window [0, 1].
            1.0 = full window remaining, 0.0 = expiry.

        Returns
        -------
        QuotePair
            Optimal two-sided quote.
        """
        # Clamp inputs
        time_remaining = max(0.001, time_remaining)  # avoid division by zero
        vol_sq = volatility ** 2

        # Reservation price: shift away from inventory risk
        reservation_price = mid_price - inventory * self.gamma * vol_sq * time_remaining

        # Optimal spread from A-S formula
        optimal_spread = (
            self.gamma * vol_sq * time_remaining
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        )

        # Clamp spread
        spread = max(self.min_spread, min(self.max_spread, optimal_spread))

        # Compute raw bid/ask
        bid = reservation_price - spread / 2.0
        ask = reservation_price + spread / 2.0

        # Clamp to binary price range
        bid = max(_MIN_PRICE, min(_MAX_PRICE, bid))
        ask = max(_MIN_PRICE, min(_MAX_PRICE, ask))

        # Ensure bid < ask after clamping
        if bid >= ask:
            mid = (bid + ask) / 2.0
            bid = max(_MIN_PRICE, mid - self.min_spread / 2.0)
            ask = min(_MAX_PRICE, mid + self.min_spread / 2.0)

        return QuotePair(
            bid=round(bid, 4),
            ask=round(ask, 4),
            bid_size=self.order_size,
            ask_size=self.order_size,
        )

    def apply_signal_bias(
        self,
        quotes: QuotePair,
        rsi_signal: int,
        bias_bps: float = 0.005,
    ) -> QuotePair:
        """Skew quotes based on a directional RSI signal.

        Parameters
        ----------
        quotes : QuotePair
            Base quotes from ``compute_quotes``.
        rsi_signal : int
            +1 = RSI says UP (bid more aggressively), -1 = DOWN, 0 = no bias.
        bias_bps : float
            Price shift per unit signal (default 50 bps = 0.005).

        Returns
        -------
        QuotePair
            Biased quotes with adjusted bid/ask.
        """
        if rsi_signal == 0:
            return quotes

        shift = rsi_signal * bias_bps

        # RSI UP (+1): bid higher (more aggressive), ask higher (less aggressive)
        # RSI DOWN (-1): bid lower (less aggressive), ask lower (more aggressive)
        bid = max(_MIN_PRICE, min(_MAX_PRICE, quotes.bid + shift))
        ask = max(_MIN_PRICE, min(_MAX_PRICE, quotes.ask + shift))

        if bid >= ask:
            mid = (bid + ask) / 2.0
            bid = max(_MIN_PRICE, mid - self.min_spread / 2.0)
            ask = min(_MAX_PRICE, mid + self.min_spread / 2.0)

        return QuotePair(
            bid=round(bid, 4),
            ask=round(ask, 4),
            bid_size=quotes.bid_size,
            ask_size=quotes.ask_size,
        )

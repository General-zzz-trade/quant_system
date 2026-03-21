"""Avellaneda-Stoikov quoter adapted for perpetual futures.

Key adaptations vs vanilla A-S:
  - No expiry: uses rolling time horizon (T resets every N seconds)
  - Funding bias: shifts reservation price by predicted funding cost
  - VPIN widening: widens spread when order flow is toxic
"""

from __future__ import annotations

import math
import dataclasses as dc

from .config import MarketMakerConfig


@dc.dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    reservation: float
    spread: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class PerpQuoter:
    """Compute bid/ask quotes using adapted A-S model.

    Supports three quoting modes:
      - "as" (default): Standard A-S spread, quotes outside BBO
      - "bbo": Join BBO, earn maker rebate only
      - "adaptive": Auto-switch between AS/BBO based on vol + VPIN
    """

    def __init__(self, cfg: MarketMakerConfig) -> None:
        self._cfg = cfg

    def compute_quotes(
        self,
        mid: float,
        inventory: float,
        vol: float,
        time_remaining: float,
        funding_rate: float = 0.0,
        vpin: float = 0.0,
        best_bid: float = 0.0,
        best_ask: float = 0.0,
        ob_imbalance: float = 0.0,
        mode: str = "adaptive",
    ) -> Quote | None:
        """Return a Quote or None if inputs are invalid.

        Args:
            mid: Current mid price.
            inventory: Net position in base asset (positive = long).
            vol: Volatility estimate (per-trade, not annualised).
            time_remaining: Fraction of time horizon remaining [0, 1].
            funding_rate: Current 8h funding rate (e.g. 0.0001 = 1 bps).
            vpin: Volume-synchronised PIN [0, 1].
            best_bid: Current BBO bid (for BBO/adaptive mode).
            best_ask: Current BBO ask (for BBO/adaptive mode).
            ob_imbalance: Orderbook imbalance [-1,+1] (positive=bid-heavy).
            mode: "as", "bbo", or "adaptive".
        """
        if mid <= 0 or vol <= 0 or time_remaining <= 0:
            return None

        cfg = self._cfg
        tick = cfg.tick_size

        # ── Mode selection ──────────────────────────────────
        if mode == "adaptive":
            # Low vol + low VPIN → BBO mode (safe, collect rebate)
            # High vol or high VPIN → A-S mode (wider spread, protection)
            if vpin > cfg.vpin_threshold or vol > 0.001:
                mode = "as"
            else:
                mode = "bbo"

        # ── BBO mode: join best bid/ask ─────────────────────
        if mode == "bbo" and best_bid > 0 and best_ask > 0:
            bid = best_bid
            ask = best_ask

            # Inventory skew: shift towards reducing inventory
            if inventory > 0:
                # Long → want to sell → move ask closer (improve)
                ask = max(best_bid + tick, ask - tick)
            elif inventory < 0:
                # Short → want to buy → move bid closer
                bid = min(best_ask - tick, bid + tick)

            # OB imbalance skew
            if abs(ob_imbalance) > 0.3:
                if ob_imbalance > 0:
                    # Bid-heavy → price likely up → shift ask closer
                    ask = max(best_bid + tick, ask - tick)
                else:
                    bid = min(best_ask - tick, bid + tick)

            # H15 fix: clamp instead of returning None when crossed
            if ask <= bid:
                ask = bid + tick
            if bid <= 0:
                return None

            return Quote(
                bid=bid, ask=ask,
                bid_size=cfg.order_size_eth, ask_size=cfg.order_size_eth,
                reservation=mid, spread=ask - bid,
            )

        # ── A-S mode: standard spread calculation ───────────
        # Adaptive gamma: lower in calm markets → tighter spread
        gamma = cfg.gamma
        if vpin < 0.3 and vol < 0.0005:
            gamma *= 0.5  # half gamma in calm markets → tighter quotes
        elif vpin > cfg.vpin_threshold:
            gamma *= 1.5  # increase gamma when toxic → wider quotes

        T = time_remaining
        var = vol * vol

        # Reservation price
        reservation = mid - inventory * gamma * var * T
        reservation -= funding_rate * cfg.funding_bias_mult * mid

        # OB imbalance bias: shift reservation towards imbalance direction
        if abs(ob_imbalance) > 0.2:
            reservation += ob_imbalance * 0.5 * tick * 10  # shift by ~5 ticks max

        # Optimal spread from A-S
        optimal_spread = gamma * var * T + (2.0 / gamma) * math.log(1.0 + gamma / cfg.kappa)

        # VPIN toxicity widening
        if vpin > cfg.vpin_threshold:
            optimal_spread *= cfg.vpin_spread_mult

        # Clamp spread to bounds
        min_spread = cfg.min_spread_bps * 1e-4 * mid
        max_spread = cfg.max_spread_bps * 1e-4 * mid
        spread = max(min_spread, min(optimal_spread, max_spread))

        bid = _round_down(reservation - spread / 2.0, tick)
        ask = _round_up(reservation + spread / 2.0, tick)

        if bid <= 0 or ask <= bid:
            return None

        return Quote(
            bid=bid, ask=ask,
            bid_size=cfg.order_size_eth, ask_size=cfg.order_size_eth,
            reservation=reservation, spread=ask - bid,
        )


def _round_down(value: float, step: float) -> float:
    return round(math.floor(value / step) * step, 10)


def _round_up(value: float, step: float) -> float:
    return round(math.ceil(value / step) * step, 10)

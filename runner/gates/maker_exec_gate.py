# runner/gates/maker_exec_gate.py
"""Maker execution gate — convert market orders to limit orders.

At 100x leverage, fee impact is critical:
  Taker: 4bps × notional = significant cost
  Maker: -1bps × notional = REBATE (we get paid)

This gate intercepts order events and converts them to limit orders
placed at BBO or slightly inside spread, with a timeout fallback
to taker if not filled within max_wait_s.

Usage in gate chain: insert AFTER all risk/sizing gates, BEFORE execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from runner.gate_chain import GateResult

_log = logging.getLogger(__name__)


@dataclass
class MakerExecConfig:
    """Configuration for maker execution."""
    enabled: bool = True
    # Limit order placement
    offset_ticks: int = 0          # 0 = at BBO, 1 = 1 tick inside
    tick_size: float = 0.01        # ETH tick
    # Timeout
    max_wait_s: float = 5.0        # fallback to taker after this
    # Chase
    chase_ticks: int = 2           # reprice up to N ticks if not filled
    chase_interval_s: float = 1.0  # reprice every N seconds
    # Minimum spread to attempt maker (if spread=1 tick, no room)
    min_spread_ticks: int = 1      # need at least 1 tick spread


class MakerExecGate:
    """Gate: annotate order events with maker execution parameters.

    This gate does NOT reject orders. It adds metadata to the event
    that the execution layer uses to place limit orders instead of market.

    The gate reads current BBO from context and computes the optimal
    limit price. If spread is too tight (1 tick), falls back to taker.
    """

    name = "MakerExec"

    def __init__(
        self,
        cfg: MakerExecConfig | None = None,
        get_bbo: Optional[Callable[[str], tuple[float, float]]] = None,
    ) -> None:
        self._cfg = cfg or MakerExecConfig()
        self._get_bbo = get_bbo
        self._maker_attempts = 0
        self._maker_fills = 0
        self._taker_fallbacks = 0

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._cfg.enabled:
            return GateResult(allowed=True, scale=1.0)

        # Get BBO from context
        best_bid, best_ask = self._get_current_bbo(context)
        if best_bid <= 0 or best_ask <= 0:
            return GateResult(allowed=True, scale=1.0, reason="no_bbo_taker")

        spread_ticks = round((best_ask - best_bid) / self._cfg.tick_size)

        # If spread is too tight, can't improve on taker
        if spread_ticks < self._cfg.min_spread_ticks:
            return GateResult(
                allowed=True, scale=1.0,
                reason=f"spread={spread_ticks}ticks<min, taker",
            )

        # Determine side from event
        side = ""
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            side = meta.get("side", "")
        if not side:
            side = context.get("side", "")

        if not side:
            return GateResult(allowed=True, scale=1.0)

        # Compute limit price
        tick = self._cfg.tick_size
        offset = self._cfg.offset_ticks * tick

        if side.lower() in ("buy", "long"):
            # Buy: place limit at best_bid + offset (join BBO or improve)
            limit_price = best_bid + offset
            # Don't cross the spread
            limit_price = min(limit_price, best_ask - tick)
        else:
            # Sell: place limit at best_ask - offset
            limit_price = best_ask - offset
            limit_price = max(limit_price, best_bid + tick)

        self._maker_attempts += 1

        # Annotate event metadata for execution layer
        maker_params = {
            "exec_type": "limit",
            "limit_price": limit_price,
            "max_wait_s": self._cfg.max_wait_s,
            "chase_ticks": self._cfg.chase_ticks,
            "chase_interval_s": self._cfg.chase_interval_s,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_ticks": spread_ticks,
        }

        # Store in context for downstream execution
        context["maker_exec"] = maker_params

        _log.debug(
            "MakerExec: side=%s bbo=%.2f/%.2f spread=%dticks → limit=%.2f",
            side, best_bid, best_ask, spread_ticks, limit_price,
        )

        return GateResult(
            allowed=True, scale=1.0,
            reason=f"maker limit={limit_price:.2f} spread={spread_ticks}t",
        )

    def _get_current_bbo(self, context: Dict[str, Any]) -> tuple[float, float]:
        """Get best bid/offer from context or callback."""
        bb = context.get("best_bid", 0.0)
        ba = context.get("best_ask", 0.0)
        if bb > 0 and ba > 0:
            return float(bb), float(ba)

        if self._get_bbo is not None:
            symbol = context.get("symbol", "")
            try:
                return self._get_bbo(symbol)
            except Exception:
                pass

        return 0.0, 0.0

    def record_fill(self, was_maker: bool) -> None:
        """Record whether fill was maker or taker (for tracking)."""
        if was_maker:
            self._maker_fills += 1
        else:
            self._taker_fallbacks += 1

    @property
    def maker_rate(self) -> float:
        """Fraction of fills that were maker."""
        total = self._maker_fills + self._taker_fallbacks
        return self._maker_fills / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "attempts": self._maker_attempts,
            "maker_fills": self._maker_fills,
            "taker_fallbacks": self._taker_fallbacks,
            "maker_rate": self.maker_rate,
        }

# runner/gates/vpin_entry_gate.py
"""VPIN + orderbook imbalance execution timing gate (Tier 2a + 2c).

Delays order entry until microstructure conditions are favorable:
  - VPIN < threshold (low toxicity = safe to enter)
  - OB imbalance aligned with trade direction (depth supports our side)
  - Spread not abnormally wide (liquidity present)

This gate does NOT reject trades permanently — it returns scale=0.0
to delay entry, signaling the engine to retry on the next tick/bar.
When conditions are favorable, returns scale=1.0 (or >1.0 if strongly
aligned).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

from runner.gate_chain import GateResult
from _quant_hotpath import (  # type: ignore[import-untyped]
    RustVPINCalculator,
    RustVPINResult,
)

_log = logging.getLogger(__name__)

# Rust-accelerated VPIN calculator — available for tick-level VPIN
# computation in the binary hot path.
VPINCalculatorType = RustVPINCalculator
VPINResultType = RustVPINResult


@dataclass
class VPINEntryConfig:
    """Configuration for VPIN entry timing."""
    enabled: bool = True

    # VPIN thresholds
    vpin_safe: float = 0.3       # VPIN < 0.3 = safe to enter (low toxicity)
    vpin_caution: float = 0.5    # VPIN 0.3-0.5 = proceed with smaller size
    vpin_danger: float = 0.7     # VPIN > 0.7 = delay entry

    # Imbalance alignment
    imbalance_boost: float = 0.3   # |imbalance| > 0.3 aligned → boost
    imbalance_scale: float = 1.3   # scale when imbalance aligned

    # Spread filter
    max_spread_bps: float = 5.0    # don't enter if spread > 5 bps

    # Max wait before forcing entry
    max_delay_s: float = 30.0      # force entry after 30s even if unfavorable


class VPINEntryGate:
    """Gate: optimize entry timing using microstructure signals.

    Reads VPIN, ob_imbalance, spread_bps from context (populated by
    tick collector or RustStreamingMicrostructure).

    Scaling:
      VPIN < 0.3 + imbalance aligned → 1.3x (great entry)
      VPIN < 0.3                     → 1.0x (normal)
      VPIN 0.3-0.5                   → 0.7x (cautious)
      VPIN 0.5-0.7                   → 0.3x (wait if possible)
      VPIN > 0.7                     → 0.0x (delay entry)
    """

    name = "VPINEntry"

    def __init__(self, cfg: VPINEntryConfig | None = None) -> None:
        self._cfg = cfg or VPINEntryConfig()
        self._pending_entries: dict[str, float] = {}  # symbol → first_attempt_ts
        self._total_checks = 0
        self._delayed_count = 0
        self._boosted_count = 0

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._cfg.enabled:
            return GateResult(allowed=True, scale=1.0)

        self._total_checks += 1
        cfg = self._cfg

        vpin = float(context.get("vpin", 0.0))
        imbalance = float(context.get("ob_imbalance", 0.0))
        spread_bps = float(context.get("spread_bps", 0.0))
        symbol = context.get("symbol", "")

        # Get trade direction
        signal = 0
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            signal = int(meta.get("signal", 0))
        if signal == 0:
            signal = int(context.get("signal", 0))

        # Check max delay
        now = time.monotonic()
        if symbol and symbol in self._pending_entries:
            elapsed = now - self._pending_entries[symbol]
            if elapsed > cfg.max_delay_s:
                self._pending_entries.pop(symbol, None)
                _log.debug("VPINEntry: max delay reached for %s, forcing entry", symbol)
                return GateResult(allowed=True, scale=0.7, reason="max_delay_forced")

        # Spread check
        if spread_bps > cfg.max_spread_bps > 0:
            self._delayed_count += 1
            if symbol:
                self._pending_entries.setdefault(symbol, now)
            return GateResult(
                allowed=True, scale=0.3,
                reason=f"wide_spread={spread_bps:.1f}bps",
            )

        # VPIN check
        if vpin > cfg.vpin_danger:
            self._delayed_count += 1
            if symbol:
                self._pending_entries.setdefault(symbol, now)
            return GateResult(
                allowed=True, scale=0.0,
                reason=f"vpin_danger={vpin:.3f}",
            )

        if vpin > cfg.vpin_caution:
            return GateResult(
                allowed=True, scale=0.3,
                reason=f"vpin_caution={vpin:.3f}",
            )

        # VPIN is safe — check imbalance alignment
        scale = 1.0 if vpin <= cfg.vpin_safe else 0.7

        if signal != 0 and abs(imbalance) > cfg.imbalance_boost:
            # Check alignment: buy + positive imbalance (bid-heavy = support)
            aligned = (signal > 0 and imbalance > 0) or (signal < 0 and imbalance < 0)
            if aligned:
                scale = cfg.imbalance_scale
                self._boosted_count += 1

        # Clear pending
        self._pending_entries.pop(symbol, None)

        return GateResult(allowed=True, scale=scale)

    @property
    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "delayed": self._delayed_count,
            "boosted": self._boosted_count,
            "delay_rate": self._delayed_count / max(self._total_checks, 1),
            "boost_rate": self._boosted_count / max(self._total_checks, 1),
        }

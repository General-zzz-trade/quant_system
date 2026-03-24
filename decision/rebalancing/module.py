# decision/rebalancing/module.py
"""RebalanceModule — DecisionModule that produces IntentEvents from target weight drift."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Iterable, Mapping, Optional

from types import SimpleNamespace

from decision.market_access import get_decimal_attr
from decision.rebalancing.schedule import RebalanceSchedule, AlwaysRebalance
from event.types import IntentEvent
from _quant_hotpath import rust_compute_rebalance_intents  # type: ignore[import-untyped]

# Rust-accelerated rebalance intent computation — used for batch rebalance
# across many symbols when latency matters.
_rust_rebalance_intents = rust_compute_rebalance_intents


def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _make_header(ts: Any) -> Any:
    """Build minimal header for IntentEvent."""
    return SimpleNamespace(
        event_type="intent",
        ts=ts,
        event_id=f"rebal-{uuid.uuid4().hex[:12]}",
    )


@dataclass
class RebalanceModule:
    """DecisionModule that rebalances portfolio towards target weights.

    Reads from snapshot:
      - markets[sym].close → current price
      - positions[sym].qty → current position qty
      - account.balance / account.equity → equity for weight calculation

    Parameters
    ----------
    target_weights : Mapping[str, float]
        Target portfolio weights per symbol (e.g., {"BTCUSDT": 0.3, "ETHUSDT": 0.2}).
        Updated via ``set_targets()``.
    drift_threshold : float
        Minimum absolute weight drift to trigger rebalance for a symbol (default 0.05 = 5%).
    schedule : RebalanceSchedule
        Controls when rebalancing is evaluated (default: always).
    min_rebalance_interval : timedelta
        Hard minimum between rebalance actions (default: 1 hour).
    cost_threshold : float
        If expected trade cost (as fraction of notional) > expected alpha improvement,
        skip the rebalance. Set to 0 to disable cost filtering.
    cost_bps : float
        Estimated round-trip cost in basis points (default: 10 bps = 0.1%).
    origin : str
        Origin tag for IntentEvent (default: "rebalance_module").
    """

    target_weights: Mapping[str, float] = field(default_factory=dict)
    drift_threshold: float = 0.05
    schedule: RebalanceSchedule = field(default_factory=AlwaysRebalance)
    min_rebalance_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    cost_threshold: float = 0.0
    cost_bps: float = 10.0
    origin: str = "rebalance_module"

    _last_rebalance_ts: Optional[datetime] = field(default=None, init=False, repr=False)

    def set_targets(self, weights: Mapping[str, float]) -> None:
        """Update target weights (callable between bars)."""
        object.__setattr__(self, "target_weights", dict(weights))

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """DecisionModule protocol: snapshot → IntentEvents."""
        if not self.target_weights:
            return ()

        # Schedule gate
        if not self.schedule.should_rebalance(snapshot):
            return ()

        # Time interval gate
        ts = getattr(snapshot, "ts", None)
        if ts is not None and self._last_rebalance_ts is not None:
            if ts - self._last_rebalance_ts < self.min_rebalance_interval:
                return ()

        # Extract state
        equity = self._get_equity(snapshot)
        if equity is None or equity <= 0:
            return ()

        markets = getattr(snapshot, "markets", {})
        positions = getattr(snapshot, "positions", {})

        # Compute current weights and drifts
        current_weights = self._compute_current_weights(positions, markets, equity)
        intents = self._generate_intents(
            snapshot, current_weights, markets, equity, ts,
        )

        if intents:
            self._last_rebalance_ts = ts

        return intents

    def _get_equity(self, snapshot: Any) -> Optional[Decimal]:
        account = getattr(snapshot, "account", None)
        if account is None:
            return None
        for attr in ("equity", "balance", "nav"):
            v = get_decimal_attr(account, attr)
            if v is not None:
                return _d(v)
        return None

    def _compute_current_weights(
        self,
        positions: Mapping[str, Any],
        markets: Mapping[str, Any],
        equity: Decimal,
    ) -> dict[str, Decimal]:
        """Compute current position weights = |notional_i| / equity (signed)."""
        weights: dict[str, Decimal] = {}
        for sym in self.target_weights:
            pos = positions.get(sym)
            qty = _d(get_decimal_attr(pos, "qty") if pos else Decimal("0"))
            px = self._get_price(sym, markets)
            if px is None or px <= 0:
                weights[sym] = Decimal("0")
                continue
            weights[sym] = (qty * px) / equity
        return weights

    def _get_price(self, sym: str, markets: Mapping[str, Any]) -> Optional[Decimal]:
        m = markets.get(sym)
        if m is None:
            return None
        return get_decimal_attr(m, "close", "last_price", "mark_price")

    def _generate_intents(
        self,
        snapshot: Any,
        current_weights: dict[str, Decimal],
        markets: Mapping[str, Any],
        equity: Decimal,
        ts: Any,
    ) -> list[Any]:
        intents: list[Any] = []
        drift_thresh = _d(self.drift_threshold)

        for sym, target_w in self.target_weights.items():
            tw = _d(target_w)
            cw = current_weights.get(sym, Decimal("0"))
            drift = tw - cw

            if _abs(drift) < drift_thresh:
                continue

            px = self._get_price(sym, markets)
            if px is None or px <= 0:
                continue

            # Cost filter: skip if expected cost > drift benefit
            if self.cost_threshold > 0:
                trade_notional = _abs(drift) * equity
                cost_frac = _d(self.cost_bps) / Decimal("10000")
                expected_cost = trade_notional * cost_frac
                benefit = _abs(drift) * equity  # simplistic: drift * equity as benefit proxy
                if expected_cost > _d(self.cost_threshold) * benefit:
                    continue

            # Compute delta qty
            delta_notional = drift * equity
            delta_qty = delta_notional / px
            side = "buy" if delta_qty > 0 else "sell"

            header = _make_header(ts)
            intents.append(IntentEvent(
                header=header,
                intent_id=f"rebal-{sym}-{uuid.uuid4().hex[:8]}",
                symbol=sym,
                side=side,
                target_qty=_abs(delta_qty),
                reason_code="rebalance",
                origin=self.origin,
            ))

        return intents

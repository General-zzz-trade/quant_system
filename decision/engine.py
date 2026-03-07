from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence

from state.snapshot import StateSnapshot
from decision.config import DecisionConfig
from decision.context import DecisionContext
from decision.types import (
    DecisionExplain,
    DecisionOutput,
    OrderSpec,
    SignalResult,
    TargetPosition,
)
from decision.utils import stable_hash, dec_str, canonical_meta
from decision.selectors import UniverseSelector
from decision.signals.base import SignalModel, NullSignal
from decision.risk_overlay.kill_conditions import BasicKillOverlay
from decision.composer import DefaultComposer
from decision.audit import DecisionAuditor
from decision.intents.validators import IntentValidator
from decision.market_access import get_decimal_attr


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DecisionEngine:
    """Institutional-grade decision engine (pure, deterministic).

    Snapshot -> (targets, orders, explain)
    """

    cfg: DecisionConfig
    signal_model: SignalModel = NullSignal()
    composer: DefaultComposer = DefaultComposer()

    auditor: Optional[DecisionAuditor] = None

    def run(self, snapshot: StateSnapshot, *, ctx: Optional[DecisionContext] = None) -> DecisionOutput:
        if ctx is None:
            ts = getattr(snapshot, "ts", None) or _utc_now()
            ctx = DecisionContext(mode="live", now=ts, run_id="run-0", actor=self.cfg.origin)

        gates: dict[str, Any] = {}

        # 0) Risk overlay gate (soft decision gate; hard gating remains in risk layer).
        allow, reasons = BasicKillOverlay().allow(snapshot)
        gates["risk_overlay_allowed"] = allow
        gates["risk_overlay_reasons"] = list(reasons)

        if not allow:
            explain = DecisionExplain(
                ts=ctx.now,
                strategy_id=self.cfg.strategy_id,
                gates=gates,
                universe=(),
                signals=(),
                candidates=(),
                targets=(),
                orders=(),
            )
            out = DecisionOutput(ts=ctx.now, strategy_id=self.cfg.strategy_id, targets=(), orders=(), explain=explain)
            if self.auditor:
                self.auditor.record(out)
            return out

        # 1) Universe selection
        universe = UniverseSelector(self.cfg.symbols).select(snapshot)

        # 2) Signals
        sigs: List[SignalResult] = [self.signal_model.compute(snapshot, sym) for sym in universe]

        # 3) Candidates
        cg = self.composer.build_candidate_generator(max_candidates=max(1, self.cfg.max_positions))
        cfilter = self.composer.build_candidate_filter()
        candidates = list(cfilter.apply(cg.generate(sigs)))

        # 4) Allocation
        allocator = self.composer.build_allocator()
        weights = allocator.allocate(candidates)
        weights = self.composer.build_constraints(self.cfg.max_positions).apply(weights)

        # 5) Targets (signed absolute target position qty)
        sizer = self.composer.build_sizer(self.cfg.risk_fraction)
        targets: List[TargetPosition] = []
        for c in candidates:
            if c.symbol not in weights:
                continue
            w = weights[c.symbol]
            qty = sizer.target_qty(snapshot, c.symbol, w)
            if qty <= 0:
                continue
            if c.side == "buy":
                signed = qty
            else:
                signed = -qty if self.cfg.allow_short else Decimal("0")
            if signed == 0:
                continue
            targets.append(
                TargetPosition(
                    symbol=c.symbol,
                    target_qty=signed,
                    reason_code="signal",
                    origin=self.cfg.origin,
                )
            )

        # 6) Intent -> OrderSpec -> Execution policy -> Validate
        intent_builder = self.composer.build_intent_builder()
        policy = self.composer.build_execution_policy(self.cfg.execution_policy, self.cfg.price_slippage_bps)
        validator = IntentValidator(min_notional=self.cfg.min_notional, min_qty=self.cfg.min_qty)

        orders: List[OrderSpec] = []
        price_hint = get_decimal_attr(snapshot.market, "close", "last_price")
        if price_hint is None:
            raise ValueError("No price available in market snapshot (close and last_price are both None)")
        for t in targets:
            ospec = intent_builder.build(snapshot, t)
            if ospec is None:
                continue

            intent_id = stable_hash([self.cfg.strategy_id, t.symbol, dec_str(t.target_qty)], prefix="intent")
            order_id = stable_hash(
                [intent_id, ospec.side, dec_str(ospec.qty), canonical_meta(ospec.meta)],
                prefix="order",
            )

            ospec = replace(ospec, intent_id=intent_id, order_id=order_id)
            ospec = policy.apply(snapshot, ospec)
            validator.validate(ospec, price_hint=price_hint)

            orders.append(ospec)

        explain = DecisionExplain(
            ts=ctx.now,
            strategy_id=self.cfg.strategy_id,
            gates=gates,
            universe=tuple(universe),
            signals=[
                {
                    "symbol": s.symbol,
                    "side": s.side,
                    "score": str(s.score),
                    "confidence": str(s.confidence),
                    "meta": s.meta,
                }
                for s in sigs
            ],
            candidates=[
                {
                    "symbol": c.symbol,
                    "side": c.side,
                    "score": str(c.score),
                    "meta": c.meta,
                }
                for c in candidates
            ],
            targets=[
                {
                    "symbol": t.symbol,
                    "target_qty": str(t.target_qty),
                    "reason_code": t.reason_code,
                    "origin": t.origin,
                }
                for t in targets
            ],
            orders=[
                {
                    "order_id": o.order_id,
                    "intent_id": o.intent_id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": str(o.qty),
                    "price": (str(o.price) if o.price is not None else None),
                    "tif": o.tif,
                }
                for o in orders
            ],
        )

        out = DecisionOutput(ts=ctx.now, strategy_id=self.cfg.strategy_id, targets=tuple(targets), orders=tuple(orders), explain=explain)
        if self.auditor:
            self.auditor.record(out)
        return out

    def decide(self, snapshot: StateSnapshot) -> Sequence[Any]:
        """Compatibility method for engine.DecisionBridge.

        Emits:
        - IntentEvent (audit)
        - OrderEvent (execution)
        """
        from event.types import IntentEvent, OrderEvent  # local import

        out = self.run(snapshot)
        events: List[Any] = []
        for o in out.orders:
            events.append(
                IntentEvent(
                    intent_id=o.intent_id,
                    symbol=o.symbol,
                    side=o.side,
                    target_qty=o.qty,  # delta qty produced by intent_builder
                    reason_code=str((o.meta or {}).get("reason_code", "signal")),
                    origin=str((o.meta or {}).get("origin", self.cfg.origin)),
                )
            )
            events.append(
                OrderEvent(
                    order_id=o.order_id,
                    intent_id=o.intent_id,
                    symbol=o.symbol,
                    side=o.side,
                    qty=o.qty,
                    price=o.price,
                )
            )
        return events

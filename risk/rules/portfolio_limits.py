# risk/rules/portfolio_limits.py
"""Cross-asset portfolio risk rules: Gross Exposure, Net Exposure, Concentration."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional

from event.types import IntentEvent, OrderEvent, Side, Symbol
from risk.decisions import (
    RiskAdjustment,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


def _d(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def _abs(x: Decimal) -> Decimal:
    return x if x >= 0 else -x


def _norm_symbol(s: Symbol | str) -> str:
    return s.normalized if hasattr(s, "normalized") else str(s)


def _get_from_meta(meta: Mapping[str, Any], *keys: str, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    return default


def _get_equity(meta: Mapping[str, Any]) -> Optional[Decimal]:
    v = _get_from_meta(meta, "equity", "account_equity", "nav", default=None)
    return None if v is None else _d(v)


def _get_market_price(meta: Mapping[str, Any]) -> Optional[Decimal]:
    v = _get_from_meta(meta, "market_price", "mark_price", "last_price", "px", default=None)
    return None if v is None else _d(v)


def _get_positions_notional(meta: Mapping[str, Any]) -> Optional[Mapping[str, Decimal]]:
    """Return {symbol: signed_notional} from meta, or None if unavailable."""
    pn = _get_from_meta(meta, "positions_notional", default=None)
    if isinstance(pn, Mapping):
        return {str(k): _d(v) for k, v in pn.items()}

    positions = _get_from_meta(meta, "positions_exposure", "positions_mark", "positions_detail", default=None)
    if not isinstance(positions, Mapping):
        return None

    result: dict[str, Decimal] = {}
    for k, rec in positions.items():
        if isinstance(rec, Mapping) and "qty" in rec and ("mark_price" in rec or "price" in rec):
            qty = _d(rec["qty"])
            px = _d(rec.get("mark_price", rec.get("price")))
            mult = _d(rec.get("multiplier", "1"))
            sym = str(k[1]) if isinstance(k, tuple) and len(k) == 2 else str(k)
            if ":" in sym:
                sym = sym.split(":", 1)[1]
            result[sym] = qty * px * mult
    return result if result else None


def _extract_qty(qty_obj: Any) -> Decimal:
    """Extract Decimal qty from Qty wrapper or raw Decimal."""
    if hasattr(qty_obj, "value"):
        return _d(qty_obj.value)
    return _d(qty_obj)


def _signed_delta_qty(side: Side, qty: Decimal) -> Decimal:
    return qty if side == Side.BUY else -qty


# ============================================================
# GrossExposureRule
# ============================================================

@dataclass(frozen=True, slots=True)
class GrossExposureRule:
    """
    Gross exposure limit: sum(|notional_i|) / equity <= max_gross_leverage.

    Reads from meta:
      - equity / account_equity / nav
      - gross_notional / gross_exposure  (pre-computed, preferred)
      - OR positions_notional / positions_exposure (computed fallback)
      - market_price (for order-level delta estimation)
    """

    name: str = "gross_exposure"
    max_gross_leverage: Decimal = Decimal("3")
    allow_auto_reduce: bool = True

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        equity = _get_equity(meta)
        if equity is None or equity <= 0:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_GROSS, message="Missing equity for gross exposure check",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        gross = self._get_gross(meta)
        if gross is None:
            return RiskDecision.allow(tags=(self.name, "intent_skip_missing_gross"))

        lev = gross / equity
        if lev <= self.max_gross_leverage:
            return RiskDecision.allow(tags=(self.name,))

        return RiskDecision.reject(
            (RiskViolation(code=RiskCode.MAX_GROSS,
                           message=f"Gross exposure {lev:.2f}x exceeds limit {self.max_gross_leverage}x",
                           scope=RiskScope.PORTFOLIO, severity="error",
                           details={"gross": str(gross), "equity": str(equity), "leverage": str(lev)}),),
            scope=RiskScope.PORTFOLIO, tags=(self.name,),
        )

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        equity = _get_equity(meta)
        if equity is None or equity <= 0:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_GROSS, message="Missing equity",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        gross = self._get_gross(meta)
        if gross is None:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_GROSS, message="Missing gross_notional for order evaluation",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        # Estimate post-trade gross
        px = self._order_price(order, meta)
        if px is None:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_GROSS, message="Missing price for gross exposure projection",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        mult = _d(_get_from_meta(meta, "multiplier", "contract_multiplier", default="1"))
        order_qty = _extract_qty(order.qty)
        delta_notional = _abs(order_qty) * px * mult
        projected_gross = gross + delta_notional
        projected_lev = projected_gross / equity

        if projected_lev <= self.max_gross_leverage:
            return RiskDecision.allow(tags=(self.name,))

        v = RiskViolation(
            code=RiskCode.MAX_GROSS,
            message=f"Order would push gross exposure to {projected_lev:.2f}x (limit {self.max_gross_leverage}x)",
            scope=RiskScope.PORTFOLIO, severity="error",
            details={"gross": str(gross), "delta": str(delta_notional), "projected_lev": str(projected_lev)},
        )

        if not self.allow_auto_reduce:
            return RiskDecision.reject((v,), scope=RiskScope.PORTFOLIO, tags=(self.name,))

        max_gross = self.max_gross_leverage * equity
        headroom = max_gross - gross
        if headroom <= 0:
            return RiskDecision.reject((v,), scope=RiskScope.PORTFOLIO, tags=(self.name, "no_headroom"))

        denom = px * mult
        if denom <= 0:
            return RiskDecision.reject((v,), scope=RiskScope.PORTFOLIO, tags=(self.name,))

        max_qty = headroom / denom
        return RiskDecision.reduce(
            (v,), adjustment=RiskAdjustment(max_qty=float(max_qty), tags=(self.name,)),
            scope=RiskScope.PORTFOLIO, tags=(self.name,),
        )

    def _get_gross(self, meta: Mapping[str, Any]) -> Optional[Decimal]:
        v = _get_from_meta(meta, "gross_notional", "gross_exposure", default=None)
        if v is not None:
            return _d(v)
        pn = _get_positions_notional(meta)
        if pn is not None:
            return sum(_abs(n) for n in pn.values())
        return None

    @staticmethod
    def _order_price(order: OrderEvent, meta: Mapping[str, Any]) -> Optional[Decimal]:
        if getattr(order, "limit_price", None) is not None:
            return _d(order.limit_price.value)
        return _get_market_price(meta)


# ============================================================
# NetExposureRule
# ============================================================

@dataclass(frozen=True, slots=True)
class NetExposureRule:
    """
    Net exposure limit: |sum(notional_i)| / equity <= max_net_leverage.

    Reads from meta:
      - equity / account_equity / nav
      - net_notional / net_exposure (pre-computed, preferred)
      - OR positions_notional / positions_exposure (computed fallback)
    """

    name: str = "net_exposure"
    max_net_leverage: Decimal = Decimal("1")

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        equity = _get_equity(meta)
        if equity is None or equity <= 0:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_NET, message="Missing equity for net exposure check",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        net = self._get_net(meta)
        if net is None:
            return RiskDecision.allow(tags=(self.name, "intent_skip_missing_net"))

        lev = _abs(net) / equity
        if lev <= self.max_net_leverage:
            return RiskDecision.allow(tags=(self.name,))

        return RiskDecision.reject(
            (RiskViolation(code=RiskCode.MAX_NET,
                           message=f"Net exposure {lev:.2f}x exceeds limit {self.max_net_leverage}x",
                           scope=RiskScope.PORTFOLIO, severity="error",
                           details={"net": str(net), "equity": str(equity), "leverage": str(lev)}),),
            scope=RiskScope.PORTFOLIO, tags=(self.name,),
        )

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        equity = _get_equity(meta)
        if equity is None or equity <= 0:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_NET, message="Missing equity",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        net = self._get_net(meta)
        if net is None:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_NET, message="Missing net exposure for order evaluation",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        px = _get_market_price(meta)
        if px is None and getattr(order, "limit_price", None) is not None:
            px = _d(order.limit_price.value)
        if px is None:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_NET, message="Missing price for net exposure projection",
                               scope=RiskScope.PORTFOLIO, severity="error"),),
                scope=RiskScope.PORTFOLIO, tags=(self.name,),
            )

        mult = _d(_get_from_meta(meta, "multiplier", "contract_multiplier", default="1"))
        order_qty = _extract_qty(order.qty)
        delta = _signed_delta_qty(order.side, order_qty) * px * mult
        projected_net = net + delta
        projected_lev = _abs(projected_net) / equity

        if projected_lev <= self.max_net_leverage:
            return RiskDecision.allow(tags=(self.name,))

        return RiskDecision.reject(
            (RiskViolation(code=RiskCode.MAX_NET,
                           message=f"Order would push net exposure to {projected_lev:.2f}x (limit {self.max_net_leverage}x)",
                           scope=RiskScope.PORTFOLIO, severity="error",
                           details={"net": str(net), "delta": str(delta), "projected_lev": str(projected_lev)}),),
            scope=RiskScope.PORTFOLIO, tags=(self.name,),
        )

    def _get_net(self, meta: Mapping[str, Any]) -> Optional[Decimal]:
        v = _get_from_meta(meta, "net_notional", "net_exposure", default=None)
        if v is not None:
            return _d(v)
        pn = _get_positions_notional(meta)
        if pn is not None:
            return sum(pn.values())
        return None


# ============================================================
# ConcentrationRule
# ============================================================

@dataclass(frozen=True, slots=True)
class ConcentrationRule:
    """
    Single-asset concentration limit: |notional_i| / gross_notional <= max_weight.

    Reads from meta:
      - gross_notional / gross_exposure
      - positions_notional / positions_exposure (per-symbol notionals)
      - market_price (for order-level projection)
    """

    name: str = "concentration"
    max_weight: Decimal = Decimal("0.4")
    per_symbol_max_weight: Mapping[str, Decimal] = field(default_factory=dict)

    def _cap_for(self, symbol: str) -> Decimal:
        return _d(self.per_symbol_max_weight.get(symbol, self.max_weight))

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        sym = _norm_symbol(intent.symbol)
        cap = self._cap_for(sym)

        pn = _get_positions_notional(meta)
        if pn is None:
            return RiskDecision.allow(tags=(self.name, "intent_skip_missing_positions"))

        gross = sum(_abs(n) for n in pn.values())
        if gross <= 0:
            return RiskDecision.allow(tags=(self.name,))

        sym_notional = _abs(pn.get(sym, Decimal("0")))
        weight = sym_notional / gross

        if weight <= cap:
            return RiskDecision.allow(tags=(self.name,))

        return RiskDecision.reject(
            (RiskViolation(code=RiskCode.MAX_POSITION,
                           message=f"{sym} concentration {weight:.1%} exceeds limit {cap:.1%}",
                           scope=RiskScope.SYMBOL, symbol=sym, severity="error",
                           details={"weight": str(weight), "cap": str(cap), "notional": str(sym_notional)}),),
            scope=RiskScope.SYMBOL, tags=(self.name,),
        )

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        sym = _norm_symbol(order.symbol)
        cap = self._cap_for(sym)

        pn = _get_positions_notional(meta)
        if pn is None:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_POSITION,
                               message="Missing position notionals for concentration check",
                               scope=RiskScope.SYMBOL, symbol=sym, severity="error"),),
                scope=RiskScope.SYMBOL, tags=(self.name,),
            )

        px = _get_market_price(meta)
        if px is None and getattr(order, "limit_price", None) is not None:
            px = _d(order.limit_price.value)
        if px is None:
            return RiskDecision.reject(
                (RiskViolation(code=RiskCode.MAX_POSITION,
                               message="Missing price for concentration projection",
                               scope=RiskScope.SYMBOL, symbol=sym, severity="error"),),
                scope=RiskScope.SYMBOL, tags=(self.name,),
            )

        mult = _d(_get_from_meta(meta, "multiplier", "contract_multiplier", default="1"))
        order_qty = _extract_qty(order.qty)
        delta_notional = _abs(order_qty) * px * mult

        cur_sym_notional = _abs(pn.get(sym, Decimal("0")))
        projected_sym = cur_sym_notional + delta_notional
        cur_gross = sum(_abs(n) for n in pn.values())
        projected_gross = cur_gross + delta_notional

        if projected_gross <= 0:
            return RiskDecision.allow(tags=(self.name,))

        weight = projected_sym / projected_gross

        if weight <= cap:
            return RiskDecision.allow(tags=(self.name,))

        return RiskDecision.reject(
            (RiskViolation(code=RiskCode.MAX_POSITION,
                           message=f"{sym} projected concentration {weight:.1%} exceeds limit {cap:.1%}",
                           scope=RiskScope.SYMBOL, symbol=sym, severity="error",
                           details={"weight": str(weight), "cap": str(cap),
                                    "sym_notional": str(projected_sym), "gross": str(projected_gross)}),),
            scope=RiskScope.SYMBOL, tags=(self.name,),
        )

"""Backtest execution adapter with realistic trading simulation.

Capabilities (all opt-in, backward compatible):
- Immediate market order fills (default, always on)
- Limit order fill logic: fills only if bar touches limit price
- Partial fills: optional volume-based partial fill model
- Trading rules: min_qty, step_size, tick_size, min_notional → reject if violated
- Funding settlement: accrues funding cost on open positions
- ATR adaptive stop-loss: 3-phase (initial → breakeven → trailing)
- Execution summary: gross/net pnl, fees, slippage, funding, rejections, partials
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from event.header import EventHeader
from event.types import EventType


def _sign(side: str) -> int:
    s = str(side).strip().lower()
    if s in ("buy", "long"):
        return 1
    if s in ("sell", "short"):
        return -1
    raise ValueError(f"unsupported side: {side!r}")


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# ── Trading Rules ────────────────────────────────────────────────

@dataclass(frozen=True)
class TradingRules:
    """Exchange trading rules for order validation."""
    min_qty: Decimal = Decimal("0.001")
    step_size: Decimal = Decimal("0.001")
    tick_size: Decimal = Decimal("0.01")
    min_notional: Decimal = Decimal("5")     # minimum order value in quote currency
    max_qty: Optional[Decimal] = None

    def round_qty(self, qty: Decimal) -> Decimal:
        """Round qty down to nearest step_size."""
        if self.step_size <= 0:
            return qty
        return (qty / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size

    def round_price(self, price: Decimal) -> Decimal:
        """Round price to nearest tick_size."""
        if self.tick_size <= 0:
            return price
        return (price / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size

    def validate(self, qty: Decimal, price: Decimal) -> Optional[str]:
        """Return rejection reason string, or None if valid."""
        if qty < self.min_qty:
            return f"qty {qty} < min_qty {self.min_qty}"
        if self.max_qty is not None and qty > self.max_qty:
            return f"qty {qty} > max_qty {self.max_qty}"
        notional = qty * price
        if notional < self.min_notional:
            return f"notional {notional} < min_notional {self.min_notional}"
        return None


# ── Execution Summary ────────────────────────────────────────────

@dataclass
class ExecutionSummary:
    """Accumulated execution statistics for backtest reporting."""
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    expired_orders: int = 0
    partial_fill_count: int = 0
    total_fills: int = 0
    gross_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")
    total_funding: Decimal = Decimal("0")
    liquidation_count: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "rejected_orders": self.rejected_orders,
            "expired_orders": self.expired_orders,
            "partial_fill_count": self.partial_fill_count,
            "total_fills": self.total_fills,
            "gross_pnl": float(self.gross_pnl),
            "net_pnl": float(self.net_pnl),
            "total_fees": float(self.total_fees),
            "total_slippage": float(self.total_slippage),
            "total_funding": float(self.total_funding),
            "liquidation_count": self.liquidation_count,
            "rejection_reasons": dict(self.rejection_reasons),
        }


# ── Main Adapter ─────────────────────────────────────────────────

class BacktestExecutionAdapter:
    """Realistic backtest execution adapter.

    All new capabilities are opt-in via constructor parameters.
    Default behavior (no new params) is identical to the original adapter.
    """

    def __init__(
        self,
        *,
        price_source: Callable[[str], Optional[Decimal]],
        ts_source: Callable[[], Optional[datetime]],
        fee_bps: Decimal = Decimal("0"),
        slippage_bps: Decimal = Decimal("0"),
        source: str = "paper",
        on_fill: Optional[Callable[[Any], None]] = None,
        # ATR adaptive stop
        adaptive_stop: bool = False,
        atr_stop_mult: float = 2.0,
        atr_trail_trigger: float = 0.8,
        atr_trail_step: float = 0.3,
        atr_breakeven_trigger: float = 1.0,
        # NEW: trading rules (opt-in)
        trading_rules: Optional[TradingRules] = None,
        # NEW: partial fill model (opt-in)
        partial_fill_rate: float = 1.0,  # 1.0 = always full fill, 0.5 = 50% fill rate
        # NEW: limit order fill model (opt-in)
        enable_limit_check: bool = False,
        # NEW: funding settlement
        on_reject: Optional[Callable[[Any], None]] = None,
        # NEW: volume-based slippage (Almgren-Chriss √participation model)
        volume_impact_factor: float = 0.0,  # 0 = disabled, 0.1 = realistic
        # NEW: margin/liquidation model
        maintenance_margin: float = 0.0,  # 0 = disabled, 0.005 = 0.5% (Binance-like)
        initial_equity: float = 0.0,      # required when maintenance_margin > 0
        leverage: float = 1.0,
        # NEW: gate chain integration (simple position cap)
        max_position_pct: float = 1.0,    # 1.0 = no cap, 0.3 = 30% of equity per symbol
    ) -> None:
        self._price_source = price_source
        self._ts_source = ts_source
        self._fee_bps = Decimal(str(fee_bps))
        self._slippage_bps = Decimal(str(slippage_bps))
        self._source = source
        self._on_fill = on_fill
        self._on_reject = on_reject

        self._pos_qty: Dict[str, Decimal] = {}
        self._avg_px: Dict[str, Optional[Decimal]] = {}

        # Trading rules
        self._rules = trading_rules
        self._partial_fill_rate = partial_fill_rate
        self._enable_limit_check = enable_limit_check

        # ATR adaptive stop state
        self._adaptive_stop = adaptive_stop
        self._atr_stop_mult = atr_stop_mult
        self._atr_trail_trigger = atr_trail_trigger
        self._atr_trail_step = atr_trail_step
        self._atr_breakeven_trigger = atr_breakeven_trigger
        self._peak_price: Dict[str, float] = {}
        self._entry_price: Dict[str, float] = {}
        self._bar_high: float = 0.0
        self._bar_low: float = 0.0
        self._atr_buffer: List[float] = []

        # Volume-based slippage
        self._volume_impact_factor = volume_impact_factor
        self._bar_volume: float = 0.0

        # Margin model
        self._maintenance_margin = maintenance_margin
        self._equity = Decimal(str(initial_equity)) if initial_equity > 0 else Decimal("100000")
        self._leverage = Decimal(str(leverage))
        self._max_position_pct = Decimal(str(max_position_pct))
        self._liquidation_count = 0

        # Funding state
        self._cum_funding: Dict[str, Decimal] = {}

        # Pending limit orders (cross-bar support)
        self._pending_orders: List[Any] = []

        # Execution summary
        self.summary = ExecutionSummary()

    # ── Bar-level hooks ──────────────────────────────────────────

    def set_bar_hlc(self, high: float, low: float, close: float,
                    prev_close: float = 0.0, volume: float = 0.0) -> None:
        """Set current bar data for intra-bar checks. Call BEFORE decision cycle."""
        self._bar_high = high
        self._bar_low = low
        self._bar_volume = volume
        if prev_close > 0:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            atr_pct = tr / close if close > 0 else 0
            self._atr_buffer.append(atr_pct)
            if len(self._atr_buffer) > 50:
                self._atr_buffer = self._atr_buffer[-50:]

    def check_margin(self, symbol: str, current_price: float) -> Optional[Any]:
        """Check if position should be liquidated. Returns fill event or None."""
        if self._maintenance_margin <= 0:
            return None
        sym = symbol.upper()
        qty = self._pos_qty.get(sym, Decimal("0"))
        if qty == 0:
            return None
        avg = self._avg_px.get(sym)
        if avg is None:
            return None

        # Unrealized PnL
        px = Decimal(str(current_price))
        sign = Decimal("1") if qty > 0 else Decimal("-1")
        unrealized = (px - avg) * abs(qty) * sign
        position_notional = abs(qty) * avg
        current_equity = self._equity + unrealized

        # Liquidation check: equity < position × maintenance_margin
        if current_equity < position_notional * Decimal(str(self._maintenance_margin)):
            self._liquidation_count += 1
            close_side = "sell" if qty > 0 else "buy"
            order = SimpleNamespace(
                header=EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source),
                symbol=sym, side=close_side, qty=abs(qty), price=None, order_type="market",
            )
            fills = self.send_order(order)
            return fills[0] if fills else None
        return None

    def submit_limit_order(self, order_event: Any, ttl_bars: int = 24) -> None:
        """Submit a limit order to the pending queue with TTL expiry.

        The order will be checked each bar via process_pending_orders().
        If not filled within ttl_bars, it expires.
        """
        order_event._bt_ttl_remaining = ttl_bars
        order_event._bt_submitted_bar = 0
        self._pending_orders.append(order_event)

    def process_pending_orders(self, bar_index: int = 0) -> List[Any]:
        """Try to fill pending limit orders against current bar.

        Supports:
        - FIFO priority (first submitted = first checked)
        - TTL expiry (order expires after N bars)
        - Continuous partial fills (remaining qty stays in queue)

        Call after set_bar_hlc() each bar.
        """
        filled = []
        remaining = []
        for pending in self._pending_orders:
            ttl = getattr(pending, "_bt_ttl_remaining", 999)
            if ttl <= 0:
                self.summary.expired_orders += 1
                if self._on_reject:
                    self._on_reject(SimpleNamespace(
                        event_type="REJECTION", symbol=getattr(pending, "symbol", ""),
                        reason="ttl_expired", status="expired",
                        side=getattr(pending, "side", ""), qty=getattr(pending, "qty", 0),
                        ts=self._ts_source(),
                        header=EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source),
                    ))
                continue  # Drop expired

            result = self.send_order(pending)
            if result:
                fill = result[0]
                filled.extend(result)
                # Check if partial — remaining qty stays in queue
                original_qty = Decimal(str(getattr(pending, "qty", 0)))
                filled_qty = fill.qty
                if filled_qty < original_qty:
                    # Requeue remainder
                    remainder = SimpleNamespace(**{
                        k: getattr(pending, k) for k in ("header", "symbol", "side", "price", "order_type")
                    })
                    remainder.qty = original_qty - filled_qty
                    remainder._bt_ttl_remaining = ttl - 1
                    remaining.append(remainder)
            else:
                # Not filled this bar — decrement TTL
                pending._bt_ttl_remaining = ttl - 1
                remaining.append(pending)
        self._pending_orders = remaining
        return filled

    def accrue_funding(self, symbol: str, funding_rate: float) -> None:
        """Accrue funding cost for open position. Call once per funding settlement."""
        sym = symbol.upper()
        qty = self._pos_qty.get(sym, Decimal("0"))
        if qty == 0:
            return
        avg = self._avg_px.get(sym)
        if avg is None:
            return
        # Funding = position_notional × funding_rate
        # Long pays positive funding, short receives
        notional = abs(qty) * avg
        cost = notional * Decimal(str(funding_rate))
        if qty > 0:
            # Long pays positive funding
            self._cum_funding[sym] = self._cum_funding.get(sym, Decimal("0")) + cost
        else:
            # Short receives positive funding (cost is negative)
            self._cum_funding[sym] = self._cum_funding.get(sym, Decimal("0")) - cost
        self.summary.total_funding += abs(cost)

    # ── Adaptive stop ────────────────────────────────────────────

    def check_adaptive_stop(self, symbol: str) -> Optional[Any]:
        """Check if adaptive stop-loss triggered. Returns fill event or None."""
        if not self._adaptive_stop:
            return None
        sym = symbol.upper()
        qty = self._pos_qty.get(sym, Decimal("0"))
        if qty == 0:
            return None

        entry = self._entry_price.get(sym, 0.0)
        if entry <= 0:
            return None

        side = 1 if qty > 0 else -1
        atr = sum(self._atr_buffer[-14:]) / max(len(self._atr_buffer[-14:]), 1) if self._atr_buffer else 0.015

        if side > 0:
            self._peak_price[sym] = max(self._peak_price.get(sym, entry), self._bar_high)
            profit = (self._peak_price[sym] - entry) / entry
        else:
            self._peak_price[sym] = min(self._peak_price.get(sym, entry), self._bar_low)
            profit = (entry - self._peak_price[sym]) / entry

        if profit >= atr * self._atr_trail_trigger:
            sd = atr * self._atr_trail_step
            stop = self._peak_price[sym] * (1 - sd) if side > 0 else self._peak_price[sym] * (1 + sd)
        elif profit >= atr * self._atr_breakeven_trigger:
            buf = atr * 0.1
            stop = entry * (1 + buf) if side > 0 else entry * (1 - buf)
        else:
            sd = min(atr * self._atr_stop_mult, 0.05)
            sd = max(sd, 0.003)
            stop = entry * (1 - sd) if side > 0 else entry * (1 + sd)

        triggered = False
        if side > 0 and self._bar_low <= stop:
            triggered = True
        elif side < 0 and self._bar_high >= stop:
            triggered = True

        if not triggered:
            return None

        close_side = "sell" if side > 0 else "buy"
        close_qty = abs(qty)
        order = SimpleNamespace(
            header=EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source),
            symbol=sym, side=close_side, qty=close_qty, price=None,
            order_type="market",
        )
        fills = self.send_order(order)
        return fills[0] if fills else None

    # ── Core order processing ────────────────────────────────────

    def send_order(self, order_event: Any) -> List[Any]:
        """Process an order event. Returns list of fill events (0 or 1+).

        Enhanced flow:
        1. Extract order fields
        2. Apply trading rules → reject if violated
        3. Determine fill price (market vs limit)
        4. Apply partial fill model
        5. Apply slippage
        6. Update position + compute PnL
        7. Track execution summary
        8. Emit fill event
        """
        self.summary.total_orders += 1

        sym = str(getattr(order_event, "symbol")).upper()
        side = str(getattr(order_event, "side"))
        qty = Decimal(str(getattr(order_event, "qty")))
        if qty <= 0:
            return []

        order_type = str(getattr(order_event, "order_type", "market")).lower()

        # ── Step 1: Get fill price ──
        px: Optional[Decimal]
        raw_price = getattr(order_event, "price", None)

        if order_type == "limit" and raw_price is not None:
            limit_px = Decimal(str(raw_price))
            # Limit order: check if bar price would fill it
            if self._enable_limit_check:
                would_fill = self._check_limit_fill(side, limit_px)
                if not would_fill:
                    self.summary.expired_orders += 1
                    return self._make_rejection(order_event, sym, "limit_not_touched")
            px = limit_px
        elif raw_price is not None:
            px = Decimal(str(raw_price))
        else:
            px = self._price_source(sym)

        if px is None:
            self.summary.rejected_orders += 1
            return self._make_rejection(order_event, sym, "no_price")

        # ── Step 2: Trading rules validation ──
        if self._rules is not None:
            qty = self._rules.round_qty(qty)
            px = self._rules.round_price(px)
            rejection = self._rules.validate(qty, px)
            if rejection is not None:
                self.summary.rejected_orders += 1
                reason_key = rejection.split()[0]  # "qty", "notional", etc.
                self.summary.rejection_reasons[reason_key] = \
                    self.summary.rejection_reasons.get(reason_key, 0) + 1
                return self._make_rejection(order_event, sym, rejection)

        # ── Step 3: Partial fill model ──
        fill_qty = qty
        is_partial = False
        if self._partial_fill_rate < 1.0 and order_type != "market":
            fill_qty = self._rules.round_qty(qty * Decimal(str(self._partial_fill_rate))) \
                if self._rules else qty * Decimal(str(self._partial_fill_rate))
            if fill_qty < (self._rules.min_qty if self._rules else Decimal("0.001")):
                self.summary.expired_orders += 1
                return self._make_rejection(order_event, sym, "partial_below_min")
            if fill_qty < qty:
                is_partial = True
                self.summary.partial_fill_count += 1

        # ── Step 3b: Position cap (gate chain proxy) ──
        if self._max_position_pct < Decimal("1"):
            max_notional = self._equity * self._max_position_pct * self._leverage
            current_notional = abs(self._pos_qty.get(sym, Decimal("0"))) * px
            available = max_notional - current_notional
            if available <= 0 and _sign(side) == (1 if self._pos_qty.get(sym, Decimal("0")) >= 0 else -1):
                # Would exceed cap on same-direction add — reject
                self.summary.rejected_orders += 1
                self.summary.rejection_reasons["position_cap"] = \
                    self.summary.rejection_reasons.get("position_cap", 0) + 1
                return self._make_rejection(order_event, sym, "position_cap_exceeded")
            max_qty = available / px if px > 0 else fill_qty
            if fill_qty > max_qty and max_qty > 0:
                fill_qty = self._rules.round_qty(max_qty) if self._rules else max_qty
                is_partial = True
                self.summary.partial_fill_count += 1

        # ── Step 4: Slippage (base + volume impact) ──
        slippage_cost = Decimal("0")
        base_slip = self._slippage_bps / Decimal("10000") if self._slippage_bps > 0 else Decimal("0")

        # Volume-based impact: Almgren-Chriss √(qty/volume)
        vol_impact = Decimal("0")
        if self._volume_impact_factor > 0 and self._bar_volume > 0:
            notional = float(fill_qty * px)
            bar_volume_usd = self._bar_volume * float(px)
            participation = notional / bar_volume_usd if bar_volume_usd > 0 else 0.01
            vol_impact = Decimal(str(self._volume_impact_factor)) * Decimal(str(participation ** 0.5))

        total_slip = base_slip + vol_impact
        slippage_cost = px * fill_qty * total_slip
        if total_slip > 0:
            if _sign(side) > 0:
                px = px * (Decimal("1") + total_slip)
            else:
                px = px * (Decimal("1") - total_slip)

        # ── Step 5: Position update + PnL ──
        signed = fill_qty * Decimal(_sign(side))
        prev_qty = self._pos_qty.get(sym, Decimal("0"))
        prev_avg = self._avg_px.get(sym, None)

        fee = (px * fill_qty) * (self._fee_bps / Decimal("10000"))
        realized = Decimal("0")

        if prev_qty != 0 and prev_avg is not None and (prev_qty > 0) != (signed > 0):
            closed = min(abs(prev_qty), abs(signed))
            sign_prev = Decimal("1") if prev_qty > 0 else Decimal("-1")
            realized = (px - prev_avg) * closed * sign_prev

        new_qty = prev_qty + signed

        if new_qty == 0:
            new_avg = None
            # Settle accumulated funding on close
            funding_settled = self._cum_funding.pop(sym, Decimal("0"))
        else:
            funding_settled = Decimal("0")
            if prev_qty == 0 or (prev_qty > 0) == (signed > 0):
                base_qty = abs(prev_qty)
                add_qty = abs(signed)
                base_avg = prev_avg if prev_avg is not None else px
                new_avg = (base_avg * base_qty + px * add_qty) / (base_qty + add_qty)
            else:
                if (prev_qty > 0 and new_qty > 0) or (prev_qty < 0 and new_qty < 0):
                    new_avg = prev_avg
                else:
                    new_avg = px

        self._pos_qty[sym] = new_qty
        self._avg_px[sym] = new_avg

        # Track entry for adaptive stop
        if self._adaptive_stop:
            if new_qty != 0 and prev_qty == 0:
                self._entry_price[sym] = float(px)
                self._peak_price[sym] = float(px)
            elif new_qty == 0:
                self._entry_price.pop(sym, None)
                self._peak_price.pop(sym, None)

        # ── Step 6: Update equity + summary ──
        gross = realized
        net = realized - fee - slippage_cost - funding_settled
        if self._maintenance_margin > 0:
            self._equity += net  # Track equity for margin model
        self.summary.gross_pnl += gross
        self.summary.net_pnl += net
        self.summary.total_fees += fee
        self.summary.total_slippage += slippage_cost
        self.summary.filled_orders += 1
        self.summary.total_fills += 1

        # ── Step 7: Build fill event ──
        parent = getattr(order_event, "header", None)
        if isinstance(parent, EventHeader):
            h = EventHeader.from_parent(parent=parent, event_type=EventType.FILL, version=1, source=self._source)
        else:
            h = EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source)

        fill = SimpleNamespace(
            header=h,
            event_type=EventType.FILL,
            ts=self._ts_source(),
            symbol=sym,
            side=side,
            qty=fill_qty,
            price=px,
            fee=fee,
            slippage=slippage_cost,
            funding_settled=funding_settled,
            realized_pnl=realized,
            gross_pnl=gross,
            net_pnl=net,
            is_partial=is_partial,
            status="partially_filled" if is_partial else "filled",
            cash_delta=0.0,
            margin_change=0.0,
        )

        if self._on_fill is not None:
            self._on_fill(fill)

        return [fill]

    # ── Helpers ──────────────────────────────────────────────────

    def _check_limit_fill(self, side: str, limit_price: Decimal) -> bool:
        """Check if a limit order would fill given current bar high/low."""
        if self._bar_high == 0 and self._bar_low == 0:
            return True  # No bar data → assume fill (backward compat)
        if _sign(side) > 0:  # buy limit: fills if bar low <= limit
            return Decimal(str(self._bar_low)) <= limit_price
        else:  # sell limit: fills if bar high >= limit
            return Decimal(str(self._bar_high)) >= limit_price

    def _make_rejection(self, order_event: Any, symbol: str, reason: str) -> List[Any]:
        """Create a rejection event (not a fill)."""
        rejection = SimpleNamespace(
            header=EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source),
            event_type="REJECTION",
            ts=self._ts_source(),
            symbol=symbol,
            side=getattr(order_event, "side", ""),
            qty=getattr(order_event, "qty", 0),
            reason=reason,
            status="rejected",
        )
        if self._on_reject is not None:
            self._on_reject(rejection)
        return []  # No fills produced

    def apply_gate_chain(self, order_event: Any, *,
                         alpha_health_scale: float = 1.0,
                         regime_scale: float = 1.0) -> Optional[Any]:
        """Simplified gate chain proxy for backtest.

        Applies three gate-like checks (matching live gate_chain.py concepts):
        1. Position cap: rejects if adding to same-direction exceeds max_position_pct
        2. Alpha health scale: reduces qty by health factor (0.0/0.5/1.0)
        3. Regime scale: reduces qty by regime factor

        Returns modified order event, or None if rejected.
        """
        if alpha_health_scale <= 0 or regime_scale <= 0:
            self.summary.rejected_orders += 1
            self.summary.rejection_reasons["gate_health"] = \
                self.summary.rejection_reasons.get("gate_health", 0) + 1
            return None

        qty = Decimal(str(getattr(order_event, "qty", 0)))
        scaled_qty = qty * Decimal(str(alpha_health_scale)) * Decimal(str(regime_scale))

        if self._rules:
            scaled_qty = self._rules.round_qty(scaled_qty)
            if scaled_qty < self._rules.min_qty:
                self.summary.rejected_orders += 1
                self.summary.rejection_reasons["gate_below_min"] = \
                    self.summary.rejection_reasons.get("gate_below_min", 0) + 1
                return None

        # Return modified order with scaled qty
        modified = SimpleNamespace(**{
            k: getattr(order_event, k) for k in ("header", "symbol", "side", "price", "order_type")
            if hasattr(order_event, k)
        })
        modified.qty = scaled_qty
        return modified

    def get_position(self, symbol: str) -> Decimal:
        return self._pos_qty.get(symbol.upper(), Decimal("0"))

    def get_avg_price(self, symbol: str) -> Optional[Decimal]:
        return self._avg_px.get(symbol.upper())

    def get_pnl(self, symbol: str) -> Decimal:
        """Unrealized PnL for symbol."""
        sym = symbol.upper()
        qty = self._pos_qty.get(sym, Decimal("0"))
        avg = self._avg_px.get(sym)
        if qty == 0 or avg is None:
            return Decimal("0")
        px = self._price_source(sym)
        if px is None:
            return Decimal("0")
        sign = Decimal("1") if qty > 0 else Decimal("-1")
        return (px - avg) * abs(qty) * sign

# decision/ml_decision.py
"""ML-based decision module — sizes positions as % of equity based on ml_score."""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Any, Iterable, Optional


class MLDecisionModule:
    """Decision module that sizes positions as % of equity based on ml_score.

    New optional parameters (all default to current behavior):
    - threshold_short: separate threshold for short signals (None = same as threshold)
    - atr_stop: hard stop-loss in ATR multiples (0 = disabled)
    - trailing_atr: trailing stop in ATR multiples (0 = disabled)
    - min_hold_bars: minimum bars to hold before allowing signal flip (0 = disabled)
    - vol_target: volatility-target sizing (0 = fixed risk_pct)
    """

    def __init__(
        self,
        *,
        symbol: str,
        risk_pct: float = 0.5,
        threshold: float = 0.001,
        threshold_short: Optional[float] = None,
        atr_stop: float = 0.0,
        trailing_atr: float = 0.0,
        min_hold_bars: int = 0,
        vol_target: float = 0.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.risk_pct = Decimal(str(risk_pct))
        self.threshold = threshold
        self.threshold_short = threshold_short if threshold_short is not None else threshold
        self.atr_stop = atr_stop
        self.trailing_atr = trailing_atr
        self.min_hold_bars = min_hold_bars
        self.vol_target = vol_target

        # Position tracking state
        self._entry_price: Optional[float] = None
        self._peak_price: Optional[float] = None
        self._bars_held: int = 0
        self._entry_atr: Optional[float] = None

    def decide(self, snapshot: Any) -> Iterable[Any]:
        if isinstance(snapshot, dict):
            market = snapshot.get("market")
            if market is None:
                markets = snapshot.get("markets") or {}
                market = next(iter(markets.values()), None)
            positions = snapshot.get("positions") or {}
            features = snapshot.get("features") or {}
            account = snapshot.get("account")
        else:
            market = getattr(snapshot, "market", None)
            positions = getattr(snapshot, "positions", {})
            features = getattr(snapshot, "features", {})
            account = getattr(snapshot, "account", None)

        if market is None:
            return ()

        close = getattr(market, "close", None)
        if close is None:
            return ()
        close_f = float(close)
        close_d = Decimal(str(close))
        if close_d <= 0:
            return ()

        ml_score = features.get("ml_score")
        if ml_score is None:
            return ()

        # Current position
        pos = positions.get(self.symbol)
        current_qty = Decimal(str(getattr(pos, "qty", 0))) if pos else Decimal("0")
        current_side = "long" if current_qty > 0 else ("short" if current_qty < 0 else "flat")

        # ATR from features (used for stops and vol sizing)
        atr_norm = features.get("atr_norm_14")

        # Update bars held counter
        if current_side != "flat":
            self._bars_held += 1
        else:
            self._bars_held = 0

        # Update peak price for trailing stop
        if self._peak_price is not None:
            if current_side == "long":
                self._peak_price = max(self._peak_price, close_f)
            elif current_side == "short":
                self._peak_price = min(self._peak_price, close_f)

        # ── Phase 1: Stop-loss checks (before signal) ──
        stop_exit = self._check_stops(current_side, close_f, atr_norm)
        if stop_exit is not None:
            orders = self._flatten(current_qty, close, stop_exit)
            self._clear_entry_state()
            return orders

        # ── Phase 2: Signal with asymmetric thresholds ──
        if ml_score > self.threshold:
            desired = "long"
        elif ml_score < -self.threshold_short:
            desired = "short"
        else:
            desired = "flat"

        # No change needed
        if desired == current_side:
            return ()

        # ── Phase 3: Min hold suppression ──
        if self.min_hold_bars > 0 and current_side != "flat":
            if self._bars_held < self.min_hold_bars:
                return ()

        # ── Phase 4: Compute target qty ──
        balance = Decimal("10000")
        if account is not None:
            balance = Decimal(str(getattr(account, "balance", balance)))

        if desired in ("long", "short"):
            target_qty = self._compute_qty(balance, close_d, close_f, atr_norm)
        else:
            target_qty = Decimal("0")

        if desired != "flat" and target_qty <= 0:
            return ()

        # ── Phase 5: Generate orders ──
        orders = []
        if desired == "long":
            if current_qty < 0:
                orders.append(self._make_order("BUY", abs(current_qty), close, "close_short"))
            orders.append(self._make_order("BUY", target_qty, close, "open_long"))
            self._record_entry(close_f, atr_norm)
        elif desired == "short":
            if current_qty > 0:
                orders.append(self._make_order("SELL", current_qty, close, "close_long"))
            orders.append(self._make_order("SELL", target_qty, close, "open_short"))
            self._record_entry(close_f, atr_norm)
        elif desired == "flat" and current_qty != 0:
            side = "SELL" if current_qty > 0 else "BUY"
            orders.append(self._make_order(side, abs(current_qty), close, "flatten"))
            self._clear_entry_state()

        return orders

    def _check_stops(
        self, current_side: str, close: float, atr_norm: Optional[float],
    ) -> Optional[str]:
        """Check hard stop and trailing stop. Returns exit reason or None."""
        if current_side == "flat":
            return None
        if atr_norm is None:
            return None
        if self._entry_price is None:
            return None

        is_long = current_side == "long"

        # Hard stop-loss: entry_atr * atr_stop from entry price
        if self.atr_stop > 0 and self._entry_atr is not None:
            stop_dist = self._entry_atr * self._entry_price * self.atr_stop
            if is_long and close <= self._entry_price - stop_dist:
                return "stop_loss"
            if not is_long and close >= self._entry_price + stop_dist:
                return "stop_loss"

        # Trailing stop: trailing_atr * atr_norm * close from peak
        if self.trailing_atr > 0 and self._peak_price is not None:
            trail_dist = atr_norm * close * self.trailing_atr
            if is_long and close <= self._peak_price - trail_dist:
                return "trailing_stop"
            if not is_long and close >= self._peak_price + trail_dist:
                return "trailing_stop"

        return None

    def _compute_qty(
        self, balance: Decimal, close_d: Decimal, close_f: float,
        atr_norm: Optional[float],
    ) -> Decimal:
        """Compute position size. Uses vol-target sizing when enabled."""
        if self.vol_target > 0 and atr_norm is not None and atr_norm > 0 and self.atr_stop > 0:
            # qty = equity * risk_pct / (atr_norm * atr_stop) / price
            risk_budget = float(balance) * float(self.risk_pct)
            stop_dist = atr_norm * self.atr_stop
            raw_qty = risk_budget / (stop_dist * close_f)
            if raw_qty <= 0:
                return Decimal("0")
            return Decimal(str(raw_qty)).quantize(Decimal("0.001"), rounding=ROUND_DOWN)

        # Default: fixed % of equity
        target_notional = balance * self.risk_pct
        return (target_notional / close_d).quantize(Decimal("0.001"), rounding=ROUND_DOWN)

    def _record_entry(self, close: float, atr_norm: Optional[float]) -> None:
        self._entry_price = close
        self._peak_price = close
        self._bars_held = 0
        self._entry_atr = atr_norm

    def _clear_entry_state(self) -> None:
        self._entry_price = None
        self._peak_price = None
        self._bars_held = 0
        self._entry_atr = None

    def _flatten(self, current_qty: Decimal, close: Any, reason: str) -> list:
        side = "SELL" if current_qty > 0 else "BUY"
        return [self._make_order(side, abs(current_qty), close, reason)]

    def _make_order(self, side: str, qty: Decimal, price: Any, reason: str) -> Any:
        from types import SimpleNamespace
        from event.header import EventHeader
        from event.types import EventType

        h = EventHeader.new_root(event_type=EventType.ORDER, version=1, source="ml_decision")
        return SimpleNamespace(
            header=h,
            event_type=EventType.ORDER,
            symbol=self.symbol,
            side=side,
            qty=qty,
            price=price,
            order_type="MARKET",
            origin="ml_lgbm",
            reason=reason,
        )

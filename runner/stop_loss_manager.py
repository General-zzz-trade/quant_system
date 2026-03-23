"""StopLossManager — stop-loss, drawdown, and exit logic.

Extracted from AlphaRunner to reduce god-class size.
Handles: ATR trailing stop, quick-loss exit, drawdown kill,
realtime stop-loss checks, regime-aware stop scaling.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from execution.balance_utils import get_total_and_free_balance
from execution.order_utils import reliable_close_position

logger = logging.getLogger(__name__)


class StopLossManager:
    """Manages stop-loss, drawdown protection, and position exit logic.

    Owns: ATR buffer, trailing stop state, stop multipliers.
    """

    def __init__(
        self,
        symbol: str,
        atr_stop_mult_base: float = 1.2,
        trail_atr_mult: float = 0.5,
        trail_step: float = 0.2,
        breakeven_atr: float = 0.5,
    ):
        self._symbol = symbol
        self._atr_stop_mult_base = atr_stop_mult_base
        self._atr_stop_mult = atr_stop_mult_base
        self._trail_atr_mult = trail_atr_mult
        self._trail_step = trail_step
        self._breakeven_atr = breakeven_atr

        self._atr_buffer: list[float] = []
        self._trade_peak_price: float = 0.0

    @property
    def trade_peak_price(self) -> float:
        return self._trade_peak_price

    @trade_peak_price.setter
    def trade_peak_price(self, value: float) -> None:
        self._trade_peak_price = value

    @property
    def atr_buffer(self) -> list[float]:
        return self._atr_buffer

    @atr_buffer.setter
    def atr_buffer(self, value: list[float]) -> None:
        self._atr_buffer = value

    def current_atr(self) -> float:
        """Get current ATR (14-bar average). Falls back to 1.5%."""
        if len(self._atr_buffer) < 5:
            return 0.015
        return float(np.mean(self._atr_buffer[-14:]))

    def update_atr(self, bar: dict, closes: list[float]) -> None:
        """Update ATR buffer with new bar data."""
        if len(closes) < 2:
            return
        prev_close = closes[-2]
        tr = max(
            bar["high"] - bar["low"],
            abs(bar["high"] - prev_close),
            abs(bar["low"] - prev_close),
        )
        atr_pct = tr / bar["close"] if bar["close"] > 0 else 0
        self._atr_buffer.append(atr_pct)
        if len(self._atr_buffer) > 50:
            self._atr_buffer = self._atr_buffer[-50:]

    def compute_stop_price(
        self,
        current_price: float,
        entry_price: float,
        current_signal: int,
    ) -> float:
        """Compute adaptive stop price based on ATR + trailing logic.

        Three-phase stop:
        1. Initial: entry ± ATR × mult (wide, let trade breathe)
        2. Breakeven: after breakeven_atr×ATR profit, move stop to entry
        3. Trailing: after trail_atr_mult×ATR profit, trail at peak - trail_step×ATR

        Hard floor: 5% (capital protection). Hard ceiling: 0.3% (avoid noise).
        """
        if entry_price <= 0 or current_signal == 0:
            return 0.0

        atr = self.current_atr()
        side = current_signal
        entry = entry_price

        # Regime-aware stop scaling
        if len(self._atr_buffer) >= 20:
            recent_vol = np.mean(self._atr_buffer[-5:])
            median_vol = np.median(self._atr_buffer[-20:])
            if median_vol > 0:
                vol_ratio = recent_vol / median_vol
                regime_scale = np.clip(0.7 + 0.3 * vol_ratio, 0.8, 1.3)
                self._atr_stop_mult = self._atr_stop_mult_base * regime_scale

        # Update trade peak
        if self._trade_peak_price <= 0.0:
            self._trade_peak_price = entry
        if side > 0:
            self._trade_peak_price = max(self._trade_peak_price, current_price)
            profit_pct = (self._trade_peak_price - entry) / entry
        else:
            self._trade_peak_price = min(self._trade_peak_price, current_price)
            profit_pct = (entry - self._trade_peak_price) / entry

        # Phase 1: Initial stop
        initial_stop_dist = atr * self._atr_stop_mult

        # Phase 2+3: Breakeven → Trailing
        if profit_pct >= atr * self._breakeven_atr:
            if profit_pct >= atr * self._trail_atr_mult:
                trail_dist = atr * self._trail_step
                if side > 0:
                    stop = self._trade_peak_price * (1 - trail_dist)
                else:
                    stop = self._trade_peak_price * (1 + trail_dist)
            else:
                buffer = atr * 0.1
                if side > 0:
                    stop = entry * (1 + buffer)
                else:
                    stop = entry * (1 - buffer)
        else:
            if side > 0:
                stop = entry * (1 - initial_stop_dist)
            else:
                stop = entry * (1 + initial_stop_dist)

        # Hard floor: max 5% loss
        if side > 0:
            stop = max(stop, entry * 0.95)
        else:
            stop = min(stop, entry * 1.05)

        # Hard ceiling: min 0.3% distance
        min_dist = entry * 0.003
        if side > 0 and current_price - stop < min_dist:
            stop = min(stop, current_price - min_dist)
        elif side < 0 and stop - current_price < min_dist:
            stop = max(stop, current_price + min_dist)

        return stop

    def check_realtime_stoploss(
        self,
        price: float,
        *,
        current_signal: int,
        entry_price: float,
        entry_size: float,
        position_size: float,
        killed: bool,
        dry_run: bool,
        trade_lock,
        adapter: Any,
        osm: Any,
        circuit_breaker: Any,
        pnl_tracker: Any,
        inference: Any,
        record_fill_fn,
        symbol: str,
    ) -> bool:
        """Check adaptive stop-loss against real-time tick price.

        Called on every tick (~100ms). Uses ATR-based trailing stop.
        Returns True if stop was triggered and position closed.
        """
        if current_signal == 0 or entry_price <= 0 or killed:
            return False

        with trade_lock:
            if current_signal == 0 or entry_price <= 0:
                return False

            stop = self.compute_stop_price(price, entry_price, current_signal)

            triggered = False
            if current_signal > 0 and price <= stop:
                triggered = True
            elif current_signal < 0 and price >= stop:
                triggered = True

            if not triggered:
                return False

            if current_signal > 0:
                unrealized = (price - entry_price) / entry_price
            else:
                unrealized = (entry_price - price) / entry_price

            atr = self.current_atr()
            phase = "TRAIL" if unrealized > 0 else ("BREAKEVEN" if abs(unrealized) < atr else "INITIAL")

            logger.warning(
                "%s ADAPTIVE STOP [%s]: price=$%.2f stop=$%.2f entry=$%.2f "
                "pnl=%.2f%% atr=%.2f%% peak=$%.2f",
                symbol, phase, price, stop, entry_price,
                unrealized * 100, atr * 100, self._trade_peak_price,
            )

            if not dry_run:
                if not circuit_breaker.allow_request():
                    logger.warning("%s STOP CLOSE blocked by circuit breaker", symbol)
                    return False
                stop_id = f"stop_{symbol}_{int(time.time())}"
                osm.register(stop_id, symbol,
                             "sell" if current_signal > 0 else "buy",
                             "market", str(position_size))
                close_result = reliable_close_position(adapter, symbol)
                if close_result["status"] == "failed":
                    logger.error("%s STOP CLOSE FAILED after retries", symbol)
                    osm.transition(stop_id, "rejected", reason="reliable_close_failed")
                    circuit_breaker.record_failure()
                    return False
                if not close_result.get("verified", True):
                    logger.warning("%s STOP CLOSE: verification failed, proceeding", symbol)
                osm.transition(stop_id, "filled", filled_qty=str(position_size),
                               avg_price=str(price))
                circuit_breaker.record_success()

            _entry_size = entry_size if entry_size > 0 else position_size
            trade = pnl_tracker.record_close(
                symbol=symbol, side=current_signal,
                entry_price=entry_price, exit_price=price,
                size=_entry_size, reason="stop_loss",
            )
            logger.info(
                "%s STOP CLOSED: pnl=$%.4f total=$%.4f trades=%d/%d",
                symbol, trade["pnl_usd"], pnl_tracker.total_pnl,
                pnl_tracker.win_count, pnl_tracker.trade_count,
            )

            close_side = "sell" if current_signal > 0 else "buy"
            record_fill_fn(close_side, _entry_size, price,
                           realized_pnl=trade["pnl_usd"])

            inference.set_position(symbol, 0, 1)
            self._trade_peak_price = 0.0
            return True

    def check_drawdown_kill(
        self,
        adapter: Any,
        symbol: str,
        pnl_tracker: Any,
        risk_evaluator: Any,
        kill_switch: Any,
        dry_run: bool,
    ) -> dict | None:
        """Check drawdown limits. Returns kill action dict or None."""
        try:
            _dd_equity, _ = get_total_and_free_balance(adapter.get_balances())
            _dd_equity = _dd_equity or 0.0
        except Exception:
            _dd_equity = 0.0

        _dd_peak = max(pnl_tracker.peak_equity, _dd_equity) if pnl_tracker.peak_equity > 0 else _dd_equity

        if risk_evaluator is not None and kill_switch is not None and _dd_peak > 0:
            breached = risk_evaluator.check_drawdown(
                equity=_dd_equity, peak_equity=_dd_peak,
            )
            if breached:
                dd = pnl_tracker.drawdown_pct
                reason = f"{symbol} drawdown {dd:.1f}%"
                kill_switch.arm("global", "*", "halt", reason,
                                source="AlphaRunner")
                logger.critical(
                    "%s DRAWDOWN KILL (Rust): dd=%.1f%% peak=$%.2f current=$%.2f",
                    symbol, dd, pnl_tracker.peak_equity, pnl_tracker.total_pnl,
                )
                if not dry_run:
                    reliable_close_position(adapter, symbol)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}

        elif risk_evaluator is None and pnl_tracker.peak_equity > 0:
            dd = pnl_tracker.drawdown_pct
            if dd >= 15.0:
                if kill_switch is not None:
                    kill_switch.arm("global", "*", "halt",
                                    f"{symbol} drawdown {dd:.1f}%",
                                    source="AlphaRunner_fallback")
                logger.critical(
                    "%s DRAWDOWN KILL (fallback): dd=%.1f%% peak=$%.2f current=$%.2f",
                    symbol, dd, pnl_tracker.peak_equity, pnl_tracker.total_pnl,
                )
                if not dry_run:
                    reliable_close_position(adapter, symbol)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}

        return None

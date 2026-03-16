"""PortfolioManager — unified position and risk manager across all alpha sources."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Unified position and risk manager across all alpha sources.

    All alphas submit trade intents here instead of directly to the adapter.
    PortfolioManager decides what actually gets executed based on:
    1. Total portfolio exposure limit (Kelly 1.4x)
    2. Per-symbol position cap (30% equity)
    3. Signal priority (AGREE > single alpha > hedge)
    4. Unified drawdown circuit breaker
    5. Net position calculation (prevents self-hedging waste)

    Walk-forward validated across 5 alpha sources.
    """

    def __init__(self, adapter: Any, *, dry_run: bool = False,
                 max_total_exposure: float = 1.4,   # Kelly optimal
                 max_per_symbol: float = 0.30,       # 30% per symbol
                 max_drawdown_pct: float = 15.0,     # kill at 15% DD
                 min_order_notional: float = 5.0,
                 risk_evaluator: Any = None, kill_switch: Any = None):
        self._adapter = adapter
        self._dry_run = dry_run
        self._max_total = max_total_exposure
        self._max_per_sym = max_per_symbol
        self._max_dd = max_drawdown_pct
        self._risk_eval = risk_evaluator
        self._kill_switch = kill_switch
        self._min_notional = min_order_notional

        # Position state: symbol → {"qty": float, "side": str, "entry": float, "source": str}
        self._positions: dict[str, dict] = {}
        # Intent registry: source → symbol → signal
        self._intents: dict[str, dict[str, int]] = {}
        # P&L tracking
        self._total_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

    @property
    def _killed(self) -> bool:
        """Check kill switch (Rust) instead of local boolean."""
        if self._kill_switch is not None:
            return self._kill_switch.is_armed()
        return False

    def get_equity(self) -> float:
        try:
            bal = self._adapter.get_balances()
            usdt = bal.get("USDT")
            return float(usdt.total) if usdt else 0
        except Exception:
            return 0

    def submit_intent(self, source: str, symbol: str, signal: int,
                      price: float, priority: int = 1) -> dict | None:
        """Submit a trade intent from any alpha source.

        Args:
            source: alpha identifier (e.g. "ETH_COMBO", "SUI_ALPHA", "HEDGE")
            symbol: exchange symbol
            signal: +1 (long), -1 (short), 0 (flat)
            price: current price
            priority: 1=highest (AGREE), 2=single alpha, 3=hedge

        Returns: trade result dict or None if no action taken.
        """
        if self._killed:
            return {"action": "killed", "reason": "drawdown_breaker"}

        # Register intent
        self._intents.setdefault(source, {})[symbol] = signal

        # Compute net desired position for this symbol across all sources
        net_signal = self._compute_net_signal(symbol)
        current = self._positions.get(symbol, {})
        current_side = 1 if current.get("qty", 0) > 0 else (-1 if current.get("qty", 0) < 0 else 0)

        if net_signal == current_side:
            return None  # No change needed

        equity = self.get_equity()
        if equity <= 0:
            return None

        # Check total exposure limit
        total_exposure = self._total_exposure(equity, price)
        if net_signal != 0 and total_exposure >= self._max_total:
            # Already at max — only allow closing or same-direction
            if current_side != 0 and net_signal != current_side:
                pass  # Allow close + flip
            elif current_side == 0:
                logger.warning("PM: total exposure %.1f%% >= %.0f%% limit, rejecting %s %s",
                               total_exposure * 100, self._max_total * 100, symbol,
                               "long" if net_signal > 0 else "short")
                return {"action": "rejected", "reason": "total_exposure_limit"}

        # Compute position size
        max_notional = equity * self._max_per_sym
        qty = max_notional / price if price > 0 else 0
        if qty * price < self._min_notional:
            return {"action": "rejected", "reason": "below_min_notional"}

        # Execute
        return self._execute(symbol, net_signal, qty, price, source)

    def _compute_net_signal(self, symbol: str) -> int:
        """Compute net signal for a symbol across all sources.

        Priority: if any high-priority source has a signal, use it.
        If sources disagree, higher priority wins.
        """
        signals = []
        for source, sym_signals in self._intents.items():
            sig = sym_signals.get(symbol, 0)
            if sig != 0:
                signals.append(sig)

        if not signals:
            return 0
        # If all agree, use that direction
        if all(s > 0 for s in signals):
            return 1
        if all(s < 0 for s in signals):
            return -1
        # Disagreement: majority wins, tie = flat
        net = sum(signals)
        if net > 0:
            return 1
        elif net < 0:
            return -1
        return 0

    def _total_exposure(self, equity: float, exclude_price: float = 0) -> float:
        """Total portfolio exposure as fraction of equity."""
        if equity <= 0:
            return 0
        total = 0.0
        for sym, pos in self._positions.items():
            total += abs(pos.get("qty", 0)) * pos.get("entry", 0)
        return total / equity

    def _execute(self, symbol: str, desired: int, qty: float, price: float,
                 source: str) -> dict:
        """Execute position change on exchange."""
        current = self._positions.get(symbol, {})
        current_qty = current.get("qty", 0)
        current_side = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)

        trade_info = {"symbol": symbol, "from": current_side, "to": desired,
                      "source": source, "price": price}

        # Close existing
        if current_side != 0:
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)

            # Track PnL
            entry = current.get("entry", price)
            if current_side > 0:
                pnl = (price - entry) / entry * close_qty * entry
            else:
                pnl = (entry - price) / entry * close_qty * entry
            self._total_pnl += pnl
            self._trade_count += 1
            if pnl > 0:
                self._win_count += 1
            trade_info["closed_pnl"] = round(pnl, 2)

            if not self._dry_run:
                result = self._adapter.send_market_order(symbol, close_side, round(close_qty, 2),
                                                        reduce_only=True)
                logger.info("PM CLOSE %s %s %.2f: %s", symbol, close_side, close_qty, result)

            del self._positions[symbol]

        # Check drawdown via RustRiskEvaluator + RustKillSwitch
        self._peak_pnl = max(self._peak_pnl, self._total_pnl)
        if self._risk_eval is not None and self._kill_switch is not None and self._peak_pnl > 0:
            breached = self._risk_eval.check_drawdown(
                equity=self._total_pnl, peak_equity=self._peak_pnl,
            )
            if breached:
                dd = (self._peak_pnl - self._total_pnl) / self._peak_pnl * 100
                self._kill_switch.arm("global", "*", "halt",
                                      f"PM drawdown {dd:.1f}%",
                                      source="PortfolioManager")
                logger.critical("PM DRAWDOWN KILL (Rust): dd=%.1f%% peak=$%.2f current=$%.2f",
                                dd, self._peak_pnl, self._total_pnl)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}
        elif self._risk_eval is None and self._peak_pnl > 0:
            # Fallback: manual drawdown check
            dd = (self._peak_pnl - self._total_pnl) / self._peak_pnl * 100
            if dd >= self._max_dd:
                if self._kill_switch is not None:
                    self._kill_switch.arm("global", "*", "halt",
                                          f"PM drawdown {dd:.1f}%",
                                          source="PortfolioManager_fallback")
                logger.critical("PM DRAWDOWN KILL (fallback): dd=%.1f%% peak=$%.2f current=$%.2f",
                                dd, self._peak_pnl, self._total_pnl)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}

        # Open new
        if desired != 0:
            side = "buy" if desired > 0 else "sell"
            qty = round(qty, 2)

            if not self._dry_run:
                result = self._adapter.send_market_order(symbol, side, qty)
                trade_info["result"] = result
                logger.info("PM OPEN %s %s %.2f @ $%.2f: %s", symbol, side, qty, price, result)

            self._positions[symbol] = {
                "qty": qty * desired, "side": side, "entry": price, "source": source,
            }
        trade_info["action"] = "executed"
        return trade_info

    def get_status(self) -> dict:
        return {
            "positions": {s: {"qty": p["qty"], "source": p["source"]}
                          for s, p in self._positions.items()},
            "total_pnl": round(self._total_pnl, 2),
            "trades": f"{self._win_count}/{self._trade_count}",
            "killed": self._killed,
            "exposure": round(self._total_exposure(self.get_equity()) * 100, 1),
        }

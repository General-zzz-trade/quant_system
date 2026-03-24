"""PortfolioCombiner — combines signals from multiple alphas into a single net position."""
from __future__ import annotations

import logging
import time
from typing import Any

from execution.balance_utils import get_total_and_free_balance
from execution.order_utils import reliable_close_position
from attribution.pnl_tracker import PnLTracker

try:
    from _quant_hotpath import RustFillEvent as _RustFillEvent, RustProcessResult
    _HAS_RUST_FILL = True
except ImportError:
    _HAS_RUST_FILL = False
    RustProcessResult = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class PortfolioCombiner:
    """Combines signals from multiple alphas into a single net position.

    Each alpha produces signal in {-1, 0, +1} with a weight.
    Net signal = weighted average -> discretize to {-1, 0, +1}.

    Prevents: double-sizing when both agree, fee waste when they disagree,
    and oversized positions from independent execution.

    Position management:
    - Net signal > threshold -> long
    - Net signal < -threshold -> short
    - Otherwise -> flat
    - Single position on exchange, sized by combined conviction
    """

    def __init__(self, adapter: Any, symbol: str, weights: dict[str, float],
                 threshold: float = 0.3, dry_run: bool = False,
                 min_size: float = 0.01,
                 pnl_tracker: PnLTracker | None = None,
                 state_store: Any = None):
        self._adapter = adapter
        self._symbol = symbol
        self._weights = weights  # runner_key -> weight (e.g. {"ETHUSDT": 0.5, "ETHUSDT_15m": 0.5})
        self._threshold = threshold
        self._dry_run = dry_run
        self._min_size = min_size
        self._pnl = pnl_tracker if pnl_tracker is not None else PnLTracker()
        self._state_store = state_store  # RustStateStore shared with all runners

        self._signals: dict[str, int] = {k: 0 for k in weights}
        self._current_position: int = 0  # -1, 0, +1
        self._position_size: float = 0.0
        self._entry_price: float = 0.0

    @property
    def _trade_count(self) -> int:
        return self._pnl.trade_count

    @property
    def _win_count(self) -> int:
        return self._pnl.win_count

    @property
    def _total_pnl(self) -> float:
        return self._pnl.total_pnl

    def reconcile_position(self) -> None:
        """Reconcile combiner state with actual exchange position.

        Called on startup to sync _current_position with exchange truth.
        Prevents the bug where combiner restarts with _current_position=0
        but exchange still has an open position.
        """
        try:
            positions = self._adapter.get_positions(symbol=self._symbol)
        except Exception:
            logger.debug("COMBO %s reconcile: failed to fetch positions",
                         self._symbol, exc_info=True)
            return

        exchange_side = 0
        exchange_qty = 0.0
        exchange_price = 0.0
        for pos in positions:
            if pos.symbol == self._symbol and not pos.is_flat:
                exchange_side = 1 if pos.is_long else -1
                exchange_qty = float(pos.abs_qty)
                exchange_price = float(pos.entry_price) if pos.entry_price else 0.0
                break

        if exchange_side != self._current_position:
            logger.warning(
                "COMBO %s RECONCILE: combiner_pos=%d exchange_side=%d qty=%.4f",
                self._symbol, self._current_position, exchange_side, exchange_qty,
            )
            self._current_position = exchange_side
            if exchange_side == 0:
                self._entry_price = 0.0
                self._position_size = 0.0
            else:
                # Use exchange entry_price, fall back to ticker if unavailable
                if exchange_price <= 0:
                    try:
                        tick = self._adapter.get_ticker(self._symbol)
                        exchange_price = float(tick.get("lastPrice", 0))
                    except Exception as e:
                        logger.warning("COMBO %s: ticker fetch for reconcile failed: %s", self._symbol, e)
                self._entry_price = exchange_price if exchange_price > 0 else 0.0
                self._position_size = exchange_qty
                if exchange_price > 0:
                    self._record_state_store_fill(
                        "buy" if exchange_side == 1 else "sell",
                        exchange_qty, exchange_price,
                    )
        else:
            logger.debug("COMBO %s reconcile OK: pos=%d", self._symbol, self._current_position)

    def update_signal(self, runner_key: str, signal: int, price: float) -> dict | None:
        """Update one alpha's signal and recompute net position.

        Returns trade info dict if position changed, None otherwise.
        """
        if runner_key not in self._signals:
            return None

        old_signal = self._signals[runner_key]
        if signal == old_signal:
            return None  # no change

        self._signals[runner_key] = signal

        # Compute weighted net signal
        net = 0.0
        total_weight = 0.0
        for k, w in self._weights.items():
            net += self._signals[k] * w
            total_weight += w
        if total_weight > 0:
            net /= total_weight

        # AGREE ONLY: trade only when ALL alphas agree direction
        # Backtest: AGREE Sharpe=5.48 vs weighted COMBO Sharpe=3.18 (+72%)
        n_long = sum(1 for s in self._signals.values() if s > 0)
        n_short = sum(1 for s in self._signals.values() if s < 0)
        n_total = len(self._signals)

        if n_long == n_total:
            desired = 1    # unanimous long
        elif n_short == n_total:
            desired = -1   # unanimous short
        else:
            desired = 0    # any disagreement -> flat

        if desired == self._current_position:
            return None  # net position unchanged

        # Position change needed
        trade = self._execute_change(desired, price)
        return trade

    def force_flat(self, price: float, reason: str = "external") -> dict | None:
        """Force the combiner flat and clear any stale runner signals."""
        had_signals = any(self._signals.values())
        had_position = self._current_position != 0
        if not had_signals and not had_position:
            return None

        self._signals = {k: 0 for k in self._signals}
        if had_position:
            trade = self._execute_change(0, price)
        else:
            trade = {"from": 0, "to": 0, "price": price, "signals": dict(self._signals)}

        trade["action"] = "forced_flat" if self._current_position == 0 else "forced_flat_failed"
        trade["reason"] = reason
        return trade

    def _execute_change(self, desired: int, price: float) -> dict:
        """Execute net position change on exchange."""
        prev = self._current_position
        trade_info = {
            "from": prev, "to": desired, "price": price,
            "signals": dict(self._signals),
        }

        # Close existing
        if prev != 0 and self._entry_price > 0:
            entry_size = self._position_size

            if not self._dry_run:
                close_result = reliable_close_position(self._adapter, self._symbol)
                if close_result["status"] == "failed":
                    logger.error("COMBO %s CLOSE FAILED after retries — keeping state", self._symbol)
                    return trade_info  # abort, don't record PnL or desync StateStore
                if not close_result.get("verified", True):
                    logger.warning("COMBO %s CLOSE: position verification failed", self._symbol)
                # Record close fill in StateStore (side is opposite of position)
                close_side = "sell" if prev > 0 else "buy"
                self._record_state_store_fill(close_side, self._position_size, price)

            # Record PnL only AFTER successful close (or in dry_run mode)
            trade = self._pnl.record_close(
                symbol=self._symbol,
                side=prev,
                entry_price=self._entry_price,
                exit_price=price,
                size=entry_size,
                reason="combo_signal_change",
            )
            trade_info["closed_pnl"] = round(trade["pnl_usd"], 4)
            logger.info(
                "COMBO CLOSE %s: pnl=$%.4f total=$%.4f wins=%d/%d",
                "long" if prev > 0 else "short",
                trade["pnl_usd"], self._pnl.total_pnl,
                self._pnl.win_count, self._pnl.trade_count,
            )

        # Compute new position size
        if desired != 0:
            try:
                equity, _free = get_total_and_free_balance(self._adapter.get_balances())
                if equity is None:
                    logger.error("COMBO %s: USDT total unavailable", self._symbol)
                    equity = 0.0
            except Exception as e:
                logger.error("COMBO: equity fetch failed: %s", e)
                equity = 0

            if equity <= 0 or price <= 0:
                logger.error("COMBO %s: equity=%.2f price=%.2f invalid, skipping open",
                             self._symbol, equity, price)
                self._current_position = 0
                self._entry_price = 0.0
                self._position_size = 0.0
                return trade_info

            # Conviction scaling: both agree = full size, one only = half
            agree_count = sum(1 for s in self._signals.values() if s == desired)
            conviction = agree_count / len(self._signals)  # 0.5 = one alpha, 1.0 = both
            # Use equity-based leverage ladder
            from runner.strategy_config import LEVERAGE_LADDER
            leverage = 2.0  # default
            for threshold, lev_val in LEVERAGE_LADDER:
                if equity >= threshold:
                    leverage = lev_val
            size = (equity * leverage * conviction) / price
            size = max(self._min_size, round(size, 2))

            # Cap at 30% of equity per symbol (leave room for other symbols)
            max_notional = equity * 0.30 * leverage
            size = min(size, max_notional / price)

            # Enforce dynamic safety limit (scales with equity)
            from runner.strategy_config import get_max_order_notional
            dynamic_cap = get_max_order_notional(equity)
            notional = size * price
            if notional > dynamic_cap:
                logger.warning(
                    "COMBO %s notional $%.2f exceeds limit $%.2f — clamping",
                    self._symbol, notional, dynamic_cap,
                )
                size = dynamic_cap / price

            size = max(self._min_size, round(size, 2))
            self._position_size = size
            side = "buy" if desired > 0 else "sell"

            if not self._dry_run:
                # Pre-flight margin check: avoid "ab not enough" errors
                notional = size * price
                lev_int = max(2, int(round(leverage)))
                margin_needed = notional / lev_int
                try:
                    _equity, avail = get_total_and_free_balance(self._adapter.get_balances())
                    if avail is None:
                        logger.warning(
                            "COMBO %s MARGIN PRECHECK skipped: USDT free balance unavailable",
                            self._symbol,
                        )
                    elif margin_needed > avail * 0.95:
                        logger.warning(
                            "COMBO %s MARGIN SKIP: need $%.0f margin but only $%.0f available",
                            self._symbol, margin_needed, avail,
                        )
                        self._current_position = 0
                        self._entry_price = 0.0
                        self._position_size = 0.0
                        return trade_info
                except Exception as exc:
                    logger.warning("COMBO %s MARGIN PRECHECK failed: %s", self._symbol, exc)

                # Ensure exchange leverage is set (Bybit requires integer >= 2)
                try:
                    result = self._adapter._client.post("/v5/position/set-leverage", {
                        "category": "linear", "symbol": self._symbol,
                        "buyLeverage": str(lev_int),
                        "sellLeverage": str(lev_int),
                    })
                    if isinstance(result, dict):
                        ret_code = result.get("retCode", -1)
                        if ret_code != 0:
                            logger.warning(
                                "COMBO %s set_leverage failed: retCode=%s retMsg=%s",
                                self._symbol, ret_code, result.get("retMsg"),
                            )
                except Exception as e:
                    logger.warning("COMBO %s set_leverage failed (non-fatal): %s", self._symbol, e)

                result = self._adapter.send_market_order(self._symbol, side, size)
                trade_info["result"] = result
                logger.info(
                    "COMBO ORDER result: %s %s %.2f -> %s",
                    side, self._symbol, size, result,
                )

                # Use actual fill price for correct PnL/stop tracking
                actual_price = price
                if result.get("orderId"):
                    try:
                        time.sleep(0.3)
                        fills = self._adapter.get_recent_fills(symbol=self._symbol)
                        if fills:
                            actual_price = float(fills[0].price)
                            if abs(actual_price - price) / price > 0.001:
                                logger.info(
                                    "COMBO %s SLIPPAGE: bar=$%.2f fill=$%.2f (%.3f%%)",
                                    self._symbol, price, actual_price,
                                    (actual_price - price) / price * 100,
                                )
                    except Exception as e:
                        logger.warning("COMBO %s failed to get fill price: %s",
                                       self._symbol, e)
                self._entry_price = actual_price
                trade_info["fill_price"] = actual_price
                # Record open fill in StateStore
                self._record_state_store_fill(side, size, actual_price)
            else:
                self._entry_price = price
                trade_info["fill_price"] = price

            logger.info(
                "COMBO OPEN %s %.2f @ $%.1f conviction=%.0f%% signals=%s",
                side, size, price, conviction * 100, self._signals,
            )
        else:
            self._entry_price = 0.0
            self._position_size = 0.0

        self._current_position = desired
        return trade_info

    def _record_state_store_fill(self, side: str, qty: float, price: float) -> None:
        """Record a fill in the shared RustStateStore for position truth tracking."""
        if self._state_store is None or not _HAS_RUST_FILL:
            return
        try:
            fill = _RustFillEvent(
                symbol=self._symbol,
                side=side,
                qty=qty,
                price=price,
                realized_pnl=0.0,
                ts=str(int(time.time() * 1000)),
            )
            result: RustProcessResult = self._state_store.process_event(fill, self._symbol)
            if result.advanced:
                logger.debug(
                    "COMBO fill state advanced: index=%d kind=%s",
                    result.event_index, result.kind,
                )
        except Exception as e:
            logger.debug("COMBO StateStore fill recording failed: %s", e)

    def get_status(self) -> dict:
        return {
            "position": self._current_position,
            "signals": dict(self._signals),
            "pnl": f"${self._pnl.total_pnl:.2f}",
            "trades": f"{self._pnl.win_count}/{self._pnl.trade_count}",
            "size": self._position_size,
        }

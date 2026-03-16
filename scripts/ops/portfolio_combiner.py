"""PortfolioCombiner — combines signals from multiple alphas into a single net position."""
from __future__ import annotations

import logging
from typing import Any

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
                 min_size: float = 0.01):
        self._adapter = adapter
        self._symbol = symbol
        self._weights = weights  # runner_key -> weight (e.g. {"ETHUSDT": 0.5, "ETHUSDT_15m": 0.5})
        self._threshold = threshold
        self._dry_run = dry_run
        self._min_size = min_size

        self._signals: dict[str, int] = {k: 0 for k in weights}
        self._current_position: int = 0  # -1, 0, +1
        self._position_size: float = 0.0
        self._entry_price: float = 0.0
        self._total_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

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

    def _execute_change(self, desired: int, price: float) -> dict:
        """Execute net position change on exchange."""
        prev = self._current_position
        trade_info = {
            "from": prev, "to": desired, "price": price,
            "signals": dict(self._signals),
        }

        # Close existing
        if prev != 0 and self._entry_price > 0:
            if prev > 0:
                pnl_pct = (price - self._entry_price) / self._entry_price
            else:
                pnl_pct = (self._entry_price - price) / self._entry_price
            pnl_usd = pnl_pct * self._entry_price * self._position_size
            self._total_pnl += pnl_usd
            self._trade_count += 1
            if pnl_usd > 0:
                self._win_count += 1
            trade_info["closed_pnl"] = round(pnl_usd, 4)
            logger.info(
                "COMBO CLOSE %s: pnl=$%.4f total=$%.4f wins=%d/%d",
                "long" if prev > 0 else "short",
                pnl_usd, self._total_pnl, self._win_count, self._trade_count,
            )

            if not self._dry_run:
                self._adapter.close_position(self._symbol)

        # Compute new position size
        if desired != 0:
            try:
                bal = self._adapter.get_balances()
                usdt = bal.get("USDT")
                equity = float(usdt.total) if usdt else 0
            except Exception:
                equity = 100

            # Conviction scaling: both agree = full size, one only = half
            agree_count = sum(1 for s in self._signals.values() if s == desired)
            conviction = agree_count / len(self._signals)  # 0.5 = one alpha, 1.0 = both
            leverage = 1.5
            size = (equity * leverage * conviction) / price
            size = max(self._min_size, round(size, 2))

            self._position_size = size
            self._entry_price = price

            # Cap at 30% of equity per symbol (leave room for other symbols)
            max_notional = equity * 0.30 * leverage
            size = min(size, max_notional / price)
            size = max(self._min_size, round(size, 2))

            self._position_size = size
            side = "buy" if desired > 0 else "sell"
            if not self._dry_run:
                result = self._adapter.send_market_order(self._symbol, side, size)
                trade_info["result"] = result
                logger.info(
                    "COMBO ORDER result: %s %s %.2f -> %s",
                    side, self._symbol, size, result,
                )
            logger.info(
                "COMBO OPEN %s %.2f @ $%.1f conviction=%.0f%% signals=%s",
                side, size, price, conviction * 100, self._signals,
            )
        else:
            self._entry_price = 0.0
            self._position_size = 0.0

        self._current_position = desired
        return trade_info

    def get_status(self) -> dict:
        return {
            "position": self._current_position,
            "signals": dict(self._signals),
            "pnl": f"${self._total_pnl:.2f}",
            "trades": f"{self._win_count}/{self._trade_count}",
            "size": self._position_size,
        }

"""Paper trading runner — connects to live market data but simulates execution.

This is the recommended first step before live trading. Run this for at least
1-2 weeks and compare results against your backtest before going live.

Usage:
    export BINANCE_API_KEY=...
    export BINANCE_API_SECRET=...
    python -m runner.paper_runner --symbol BTCUSDT --balance 10000 --fee-bps 4

Requirements (install separately):
    pip install websocket-client python-dotenv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("paper_runner")


@dataclass
class PaperRunnerConfig:
    symbol: str = "BTCUSDT"
    starting_balance: Decimal = Decimal("10000")
    order_qty: Decimal = Decimal("0.01")
    ma_window: int = 20
    fee_bps: Decimal = Decimal("4")       # 0.04% taker fee (Binance standard)
    slippage_bps: Decimal = Decimal("2")  # 0.02% estimated slippage
    log_interval_sec: float = 60.0        # How often to log P&L
    out_dir: Optional[Path] = None        # Where to write JSON state snapshots


class _SimpleMAStrategy:
    """Simple MA-cross strategy for paper trading demo.

    Replace with your actual DecisionBridge + strategy modules.
    """

    def __init__(self, *, symbol: str, window: int) -> None:
        self._symbol = symbol.upper()
        self._window = window
        self._closes: list[Decimal] = []

    def on_price(self, close: Decimal) -> Optional[str]:
        """Returns 'buy', 'sell', or None (hold)."""
        self._closes.append(close)
        if len(self._closes) > self._window:
            self._closes.pop(0)
        if len(self._closes) < self._window:
            return None

        ma = sum(self._closes) / Decimal(str(self._window))
        if close > ma:
            return "buy"
        elif close < ma:
            return "sell"
        return None


class PaperPortfolio:
    """Tracks simulated P&L with fee and slippage."""

    def __init__(
        self,
        *,
        balance: Decimal,
        fee_bps: Decimal,
        slippage_bps: Decimal,
    ) -> None:
        self._balance = balance
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._position_qty: Decimal = Decimal("0")
        self._avg_price: Optional[Decimal] = None
        self._realized_pnl: Decimal = Decimal("0")
        self._total_fees: Decimal = Decimal("0")
        self._trade_count: int = 0

    def execute(self, side: str, qty: Decimal, market_price: Decimal) -> Dict[str, Any]:
        """Simulate order fill with fee + slippage."""
        # Apply slippage
        slip = self._slippage_bps / Decimal("10000")
        if side == "buy":
            exec_price = market_price * (Decimal("1") + slip)
        else:
            exec_price = market_price * (Decimal("1") - slip)

        # Calculate fee
        notional = qty * exec_price
        fee = notional * (self._fee_bps / Decimal("10000"))
        self._total_fees += fee

        # Update position
        signed = qty if side == "buy" else -qty
        prev_qty = self._position_qty

        if prev_qty == Decimal("0"):
            # Opening
            self._position_qty = signed
            self._avg_price = exec_price
        elif (prev_qty > 0 and signed > 0) or (prev_qty < 0 and signed < 0):
            # Adding to position
            base_avg = self._avg_price or exec_price
            total_qty = abs(prev_qty) + qty
            self._avg_price = (base_avg * abs(prev_qty) + exec_price * qty) / total_qty
            self._position_qty = prev_qty + signed
        else:
            # Reducing or reversing
            closed = min(abs(prev_qty), qty)
            sign_prev = Decimal("1") if prev_qty > 0 else Decimal("-1")
            if self._avg_price:
                realized = (exec_price - self._avg_price) * closed * sign_prev
                self._realized_pnl += realized

            new_qty = prev_qty + signed
            if new_qty == Decimal("0"):
                self._avg_price = None
            elif (prev_qty > 0 and new_qty < 0) or (prev_qty < 0 and new_qty > 0):
                self._avg_price = exec_price  # reversed
            self._position_qty = new_qty

        # Update balance
        if side == "buy":
            self._balance -= notional + fee
        else:
            self._balance += notional - fee

        self._trade_count += 1

        return {
            "side": side,
            "qty": str(qty),
            "exec_price": str(exec_price),
            "fee": str(fee),
            "balance": str(self._balance),
            "position_qty": str(self._position_qty),
        }

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        if self._position_qty == Decimal("0") or self._avg_price is None:
            return Decimal("0")
        return (current_price - self._avg_price) * self._position_qty

    def equity(self, current_price: Decimal) -> Decimal:
        return self._balance + self.unrealized_pnl(current_price)

    def summary(self, current_price: Decimal) -> Dict[str, Any]:
        eq = self.equity(current_price)
        return {
            "balance": str(self._balance),
            "position_qty": str(self._position_qty),
            "avg_price": str(self._avg_price) if self._avg_price else "",
            "unrealized_pnl": str(self.unrealized_pnl(current_price)),
            "realized_pnl": str(self._realized_pnl),
            "total_fees": str(self._total_fees),
            "equity": str(eq),
            "trade_count": self._trade_count,
        }


def run_paper(config: PaperRunnerConfig) -> None:
    """Run paper trading mode.

    This function demonstrates the paper trading loop structure.
    For full live data integration, wire in the Binance WebSocket market feed.
    """
    symbol = config.symbol.upper()
    logger.info("Starting paper trading: symbol=%s balance=%s", symbol, config.starting_balance)

    portfolio = PaperPortfolio(
        balance=config.starting_balance,
        fee_bps=config.fee_bps,
        slippage_bps=config.slippage_bps,
    )
    strategy = _SimpleMAStrategy(symbol=symbol, window=config.ma_window)

    if config.out_dir:
        config.out_dir.mkdir(parents=True, exist_ok=True)

    last_log = time.monotonic()
    last_price = Decimal("0")

    # ── Market data loop ──────────────────────────────────────
    # In production: replace this with BinanceWebSocket market data feed
    # For now: reads from stdin as CSV prices (ts,close) for testability
    logger.info("Reading price feed from stdin (format: close_price per line).")
    logger.info("Press Ctrl+C to stop.")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                close = Decimal(line.split(",")[-1])
            except Exception:
                continue

            last_price = close
            signal = strategy.on_price(close)
            pos = portfolio._position_qty

            # Simple position management: no position → open on signal; open → close on opposite
            if signal == "buy" and pos <= Decimal("0"):
                if pos < Decimal("0"):
                    # Close short first
                    fill = portfolio.execute("buy", abs(pos), close)
                    logger.info("CLOSE SHORT  %s", json.dumps(fill))
                fill = portfolio.execute("buy", config.order_qty, close)
                logger.info("OPEN LONG    %s", json.dumps(fill))

            elif signal == "sell" and pos >= Decimal("0"):
                if pos > Decimal("0"):
                    # Close long first
                    fill = portfolio.execute("sell", pos, close)
                    logger.info("CLOSE LONG   %s", json.dumps(fill))
                fill = portfolio.execute("sell", config.order_qty, close)
                logger.info("OPEN SHORT   %s", json.dumps(fill))

            # Periodic logging
            now = time.monotonic()
            if now - last_log >= config.log_interval_sec:
                summary = portfolio.summary(last_price)
                logger.info("STATUS  price=%s  %s", last_price, json.dumps(summary))
                if config.out_dir:
                    snap_path = config.out_dir / "paper_state.json"
                    summary["ts"] = datetime.now(timezone.utc).isoformat()
                    summary["price"] = str(last_price)
                    snap_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                last_log = now

    except KeyboardInterrupt:
        pass

    # Final summary
    if last_price > 0:
        final = portfolio.summary(last_price)
        logger.info("FINAL SUMMARY: %s", json.dumps(final, indent=2))
        if config.out_dir:
            out_path = config.out_dir / "paper_final.json"
            final["ts"] = datetime.now(timezone.utc).isoformat()
            out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
            logger.info("Results written to %s", out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper trading runner")
    p.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    p.add_argument("--balance", default="10000", help="Starting balance (USDT)")
    p.add_argument("--qty", default="0.01", help="Order quantity")
    p.add_argument("--ma", type=int, default=20, help="MA window")
    p.add_argument("--fee-bps", default="4", help="Fee in basis points")
    p.add_argument("--slippage-bps", default="2", help="Slippage in basis points")
    p.add_argument("--log-interval", type=float, default=60.0, help="Status log interval (seconds)")
    p.add_argument("--out", default=None, help="Output directory for state snapshots")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PaperRunnerConfig(
        symbol=args.symbol,
        starting_balance=Decimal(args.balance),
        order_qty=Decimal(args.qty),
        ma_window=args.ma,
        fee_bps=Decimal(args.fee_bps),
        slippage_bps=Decimal(args.slippage_bps),
        log_interval_sec=args.log_interval,
        out_dir=Path(args.out) if args.out else None,
    )
    run_paper(config)


if __name__ == "__main__":
    main()

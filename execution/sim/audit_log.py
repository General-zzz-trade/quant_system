# execution/sim/audit_log.py
"""Backtest audit log — records every order, fill, and position change.

Produces a complete, replayable trade journal for post-mortem analysis.
Addresses P2-1: no audit trail in backtests.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class BacktestAuditLog:
    """Append-only audit log for backtest trades and events."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: list[dict[str, Any]] = []
        self._path = Path(path) if path else None

    def record_trade(self, trade: Any) -> None:
        """Record a completed trade."""
        entry = asdict(trade) if hasattr(trade, "__dataclass_fields__") else dict(trade)
        entry["event_type"] = "trade"
        self._entries.append(entry)

    def record_equity(self, bar: int, equity: float, position: int,
                      unrealized_pnl: float = 0) -> None:
        """Record equity snapshot at bar."""
        self._entries.append({
            "event_type": "equity_snapshot",
            "bar": bar,
            "equity": round(equity, 4),
            "position": position,
            "unrealized_pnl": round(unrealized_pnl, 4),
        })

    def record_event(self, bar: int, event_type: str, **kwargs: Any) -> None:
        """Record arbitrary event (stop-loss, liquidation, regime change, etc)."""
        self._entries.append({
            "event_type": event_type,
            "bar": bar,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in kwargs.items()},
        })

    @property
    def entries(self) -> list[dict]:
        return self._entries

    @property
    def trades(self) -> list[dict]:
        return [e for e in self._entries if e["event_type"] == "trade"]

    def save_csv(self, path: str | Path | None = None) -> None:
        """Save trade log to CSV."""
        out = Path(path) if path else self._path
        if not out or not self._entries:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        trades = self.trades
        if not trades:
            return
        keys = list(trades[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(trades)

    def save_json(self, path: str | Path | None = None) -> None:
        """Save full audit log to JSON."""
        out = Path(path) if path else self._path
        if not out:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".json"), "w") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def summary(self) -> dict:
        """Summary statistics from the audit log."""
        trades = self.trades
        if not trades:
            return {"n_trades": 0}

        pnls = [t.get("pnl_net", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        stops = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        liqs = [t for t in trades if t.get("exit_reason") == "liquidation"]

        return {
            "n_trades": len(trades),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate": len(wins) / max(len(trades), 1),
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / max(len(trades), 1),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "n_stop_losses": len(stops),
            "n_liquidations": len(liqs),
            "total_fees": sum(t.get("fees", 0) for t in trades),
            "total_slippage": sum(t.get("slippage", 0) for t in trades),
            "total_funding": sum(t.get("funding", 0) for t in trades),
        }

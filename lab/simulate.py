"""Per-symbol sleeve simulation: targets -> rebalancer -> research broker.

The loop is exactly what the paper runtime does each hour (process the bar,
settle funding, decide, send orders), so a paper run over the same bars must
reproduce these fills.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal as D

import numpy as np
import pandas as pd

from lab.orders import make_order
from lab.rebalance import plan
from runner.offline.broker import BarBroker, ReplayConfig

HOUR = 3_600_000


@dataclass(frozen=True)
class SymbolRules:
    step: D
    min_qty: D
    min_notional: D


SYMBOL_RULES = {
    "BTCUSDT": SymbolRules(D("0.001"), D("0.001"), D("100")),
    "ETHUSDT": SymbolRules(D("0.001"), D("0.001"), D("20")),
}


def broker_config(spec, rules: SymbolRules, *, starting_balance: str = "5000",
                  cost_multiplier: int = 1) -> ReplayConfig:
    m = D(cost_multiplier)
    return ReplayConfig(starting_balance=starting_balance, maker_bps=str(D(2) * m), taker_bps=str(D(5) * m),
                        slippage_bps=str(D(3) * m), limit_offset_bps="2", participation="0.01", max_exposure="1",
                        qty_step=str(rules.step), min_notional=str(rules.min_notional), ttl_bars=spec.ttl_bars,
                        entry_mode=spec.execution)


@dataclass
class SleeveResult:
    symbol: str
    curve: pd.DataFrame            # at, equity, qty, mark, fees, funding
    fills: list = field(default_factory=list)
    orders: int = 0


class Sleeve:
    """One symbol's broker + rebalancing state; used by research AND paper runtime."""

    def __init__(self, symbol: str, spec, rules: SymbolRules, config: ReplayConfig) -> None:
        self.symbol, self.spec, self.rules = symbol, spec, rules
        self.broker = BarBroker(config)
        self.orders = 0

    def on_bar(self, bar, funding_rate: D | None) -> None:
        """Advance the broker through one closed bar (fills at its open/close)."""
        b = self.broker
        b.now, b.mark = bar.open_time, bar.open
        if funding_rate is not None:
            b.settle_funding(bar.open_time, funding_rate, bar.open)
        b.process_bar(bar)

    def decide(self, bar, target: float, ready: bool) -> list:
        """Rebalance toward ``target`` at this bar's close; returns submitted intents."""
        if not ready:
            return []
        b = self.broker
        intents = plan(target, equity=b.equity, price=bar.close, current_qty=b.position.qty,
                       entry_pending=any(not r["reduce_only"] for r in b.pending), band=self.spec.band,
                       step=self.rules.step, min_qty=self.rules.min_qty, min_notional=self.rules.min_notional)
        sent = []
        for k, intent in enumerate(intents):
            order = make_order(order_id=f"{self.symbol}:{bar.available_at}:{k}", source=f"target.{self.symbol}",
                               symbol=self.symbol, side=intent.side, qty=intent.qty, price=bar.close,
                               reduce_only=intent.reduce_only, reason="rebalance")
            b.send_order(order)
            self.orders += 1
            sent.append(order)
        return sent


def simulate(symbol: str, bars: list, targets: pd.DataFrame, funding: dict, spec, *, start_ms: int,
             end_ms: int | None = None, rules: SymbolRules | None = None, cost_multiplier: int = 1,
             starting_balance: str = "5000") -> SleeveResult:
    rules = rules or SYMBOL_RULES[symbol]
    sleeve = Sleeve(symbol, spec, rules, broker_config(spec, rules, starting_balance=starting_balance,
                                                       cost_multiplier=cost_multiplier))
    target_map = dict(zip(targets["available_at"].to_numpy(dtype=np.int64).tolist(),
                          zip(targets["target"].tolist(), targets["ready"].tolist())))
    rows = []
    for bar in bars:
        if bar.available_at <= start_ms or (end_ms is not None and bar.open_time >= end_ms):
            continue
        rate = funding[bar.open_time][0] if bar.open_time in funding else None
        sleeve.on_bar(bar, rate)
        target, ready = target_map.get(bar.available_at, (0.0, False))
        sleeve.decide(bar, target, ready)
        b = sleeve.broker
        rows.append((bar.available_at, float(b.equity), float(b.position.qty), float(b.mark), float(b.fees),
                     float(b.funding)))
    curve = pd.DataFrame(rows, columns=["at", "equity", "qty", "mark", "fees", "funding"])
    return SleeveResult(symbol, curve, list(sleeve.broker.fills), sleeve.orders)


__all__ = ["simulate", "Sleeve", "SymbolRules", "SYMBOL_RULES", "broker_config", "replace"]

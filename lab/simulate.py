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
# The research lot constants and broker settings live with the broker (S10); research imports them from here.
from trading.adapters.simulated.bar_broker import SYMBOL_RULES, BarBroker, ReplayConfig, SymbolRules, broker_config

HOUR = 3_600_000


@dataclass
class SleeveResult:
    symbol: str
    curve: pd.DataFrame            # at, equity, qty, mark, fees, funding
    fills: list = field(default_factory=list)
    orders: int = 0
    order_rows: list = field(default_factory=list)   # the broker's order rows; a fill's "order" is a row's "id"


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
             starting_balance: str = "5000", stop_pct: str = "0",
             pause_after_stop_ms: int | None = None, reenter_after_stop: str = "pause",
             stop_pct_at: dict[int, str] | None = None) -> SleeveResult:
    """``stop_pct`` gives the broker the live exchange stop; after a stop fills, entries pause for
    ``pause_after_stop_ms`` (the live entry pause, released by itself after that time: tech lead 2026-11-16).

    ``reenter_after_stop="signal_reset"`` is the other rule a study may test (research/lab/runs/2026-11-17/plan.md):
    after a stop fills there is no entry until the target has been exactly 0 on a ready decision since the stop (the
    decision at the close of the stop's bar counts); the next non-zero target may enter. A target on the other side
    of the stopped position is a new signal and may enter too (decided 2026-11-16, before the first run that uses the
    rule: a long/short idea can change direction without a decision at 0). It takes no pause time and needs a stop.

    ``reenter_after_stop="pause_then_reset"`` (tech lead 2026-11-17) is the live runtime's ``after_stop:
    pause_then_reset``: no entry until both the pause has passed and a ready decision since the stop wanted exactly 0
    (a 0 inside the pause counts; the other side is no exception). It needs a pause and a stop.

    ``stop_pct_at`` (research only, tech lead 2026-11-17): the stop distance decided at each decision time
    (``available_at``), in (0, ``stop_pct``]; a position the decision's orders open keeps it until it is flat. A
    decision time it does not list has the fixed ``stop_pct``."""
    if reenter_after_stop not in ("pause", "signal_reset", "pause_then_reset"):
        raise ValueError("reenter_after_stop must be 'pause', 'signal_reset' or 'pause_then_reset', "
                         f"not {reenter_after_stop!r}")
    wait_for_reset = reenter_after_stop in ("signal_reset", "pause_then_reset")
    zero_only = reenter_after_stop == "pause_then_reset"         # the live rule: no exception for the other side
    if reenter_after_stop == "signal_reset" and pause_after_stop_ms:
        raise ValueError("reenter_after_stop='signal_reset' takes no pause_after_stop_ms: one rule per stop")
    if zero_only and not pause_after_stop_ms:
        raise ValueError("reenter_after_stop='pause_then_reset' needs a pause_after_stop_ms")
    if wait_for_reset and D(stop_pct) <= 0:
        raise ValueError(f"reenter_after_stop={reenter_after_stop!r} needs a stop_pct above 0")
    rules = rules or SYMBOL_RULES[symbol]
    sleeve = Sleeve(symbol, spec, rules, broker_config(spec, rules, starting_balance=starting_balance,
                                                       cost_multiplier=cost_multiplier, stop_pct=stop_pct))
    target_map = dict(zip(targets["available_at"].to_numpy(dtype=np.int64).tolist(),
                          zip(targets["target"].tolist(), targets["ready"].tolist())))
    rows, paused_until, waiting, stopped_side = [], 0, False, 0
    for bar in bars:
        if bar.available_at <= start_ms or (end_ms is not None and bar.open_time >= end_ms):
            continue
        rate = funding[bar.open_time][0] if bar.open_time in funding else None
        stops = len(sleeve.broker.fills)
        sleeve.on_bar(bar, rate)
        stop_fills = [f for f in sleeve.broker.fills[stops:] if f["order"] == "protective_stop"] \
            if (pause_after_stop_ms or wait_for_reset) else []
        stopped = bool(stop_fills)
        if stopped and pause_after_stop_ms:
            paused_until = bar.available_at + pause_after_stop_ms
        target, ready = target_map.get(bar.available_at, (0.0, False))
        if wait_for_reset:
            if stopped:                                    # a sell stop closed a long, a buy stop a short
                waiting, stopped_side = True, (1 if stop_fills[-1]["side"] == "sell" else -1)
            if waiting and ready and (target == 0.0 or (not zero_only and target * stopped_side < 0)):
                waiting = False
        if bar.available_at < paused_until or waiting:
            target = 0.0                                   # paused: no entry (a stopped sleeve holds nothing)
        if stop_pct_at is not None:
            sleeve.broker.set_next_stop_pct(stop_pct_at.get(bar.available_at))
        sleeve.decide(bar, target, ready)
        b = sleeve.broker
        rows.append((bar.available_at, float(b.equity), float(b.position.qty), float(b.mark), float(b.fees),
                     float(b.funding)))
    curve = pd.DataFrame(rows, columns=["at", "equity", "qty", "mark", "fees", "funding"])
    return SleeveResult(symbol, curve, list(sleeve.broker.fills), sleeve.orders, list(sleeve.broker.orders))


__all__ = ["simulate", "Sleeve", "SymbolRules", "SYMBOL_RULES", "broker_config", "replace"]

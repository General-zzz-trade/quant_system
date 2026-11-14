"""Turn a target weight into order intents (shared by research and live).

Rules:
- sign change or target 0 with a position -> reduce the whole position first,
  then (if the target is non-zero) one entry for the new side;
- otherwise trade only when |target - current weight| exceeds ``band``;
  shrinking a position is a reduce-only order, growing it is an entry;
- never stack a second entry while one is still working (a reduction cancels
  working entries, so a flip may place its new entry immediately);
- quantities are floored to the lot step; entries below the exchange minimum
  quantity/notional are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

D = Decimal


@dataclass(frozen=True)
class Intent:
    side: str          # buy | sell
    qty: Decimal       # > 0
    reduce_only: bool


def floor_step(qty: Decimal, step: Decimal) -> Decimal:
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def plan(target_weight: float, *, equity: Decimal, price: Decimal, current_qty: Decimal,
         entry_pending: bool, band: float, step: Decimal, min_qty: Decimal,
         min_notional: Decimal) -> list[Intent]:
    if equity <= 0 or price <= 0:
        return []
    target = D(str(target_weight))
    raw = target * equity / price
    target_qty = floor_step(abs(raw), step) * (1 if raw >= 0 else -1)
    intents: list[Intent] = []
    if current_qty != 0 and (target_qty == 0 or (target_qty > 0) != (current_qty > 0)):
        intents.append(Intent("sell" if current_qty > 0 else "buy", abs(current_qty), True))
        if target_qty != 0:  # the reduction cancels any working entry first
            qty = abs(target_qty)
            if qty >= min_qty and qty * price >= min_notional:
                intents.append(Intent("buy" if target_qty > 0 else "sell", qty, False))
        return intents
    current_weight = current_qty * price / equity
    if abs(target - current_weight) <= D(str(band)):
        return []
    delta = abs(target_qty) - abs(current_qty)
    if delta < 0:
        qty = floor_step(-delta, step)
        if qty >= step:
            intents.append(Intent("sell" if current_qty > 0 else "buy", qty, True))
    elif delta > 0 and not entry_pending:
        qty = floor_step(delta, step)
        if qty >= min_qty and qty * price >= min_notional:
            intents.append(Intent("buy" if target_qty > 0 else "sell", qty, False))
    return intents

"""What the exchange says about a contract: its kind, whether it trades, a scheduled delisting, and its margin tiers
(plan D1, docs/arch/dynamic-universe-plan.md). The adapter translates the exchange's words into these; nothing here
names them (tests/architecture: no exchange word in the domain).

``contract_problems`` is the start-up rule for one symbol: a USDT perpetual (crypto or TradFi, owner 2026-11-19)
that is trading, has no delisting scheduled, allows the configured exchange leverage, and - with isolated margin -
keeps the protective stop inside ``fraction`` of the liquidation distance at its own maintenance margin tier for the
planned position (the 0.5 % rate assumed before D1 holds for BTC and ETH, not for most other contracts)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from trading.domain.risk.liquidation import liquidation_distance

PERPETUAL_KINDS = ("crypto", "tradfi")


@dataclass(frozen=True, slots=True)
class Contract:
    symbol: str
    kind: str                     # "crypto" (a perpetual), "tradfi" (a TradFi perpetual) or "other" (delivery, ...)
    state: str                    # "open" (trading), "halted", "settling" (being delisted), "pending", "other"
    quote: str
    delisting_at: int | None      # ms of a scheduled settlement; None when none is scheduled


@dataclass(frozen=True, slots=True)
class Bracket:
    floor: Decimal                # notional from (inclusive)
    cap: Decimal                  # notional to (exclusive)
    maintenance_rate: Decimal
    max_leverage: int


@dataclass(frozen=True, slots=True)
class MarginTiers:
    symbol: str
    brackets: tuple[Bracket, ...]  # lowest notional first

    def max_leverage(self) -> int:
        return self.brackets[0].max_leverage if self.brackets else 0

    def at(self, notional: Decimal) -> Bracket | None:
        """The tier of a position of this notional; None beyond the last one."""
        return next((b for b in self.brackets if b.floor <= notional < b.cap), None)


def contract_problems(symbol: str, contract: Contract | None, tiers: MarginTiers | None, *, stop_pct: Decimal,
                      leverage: int, margin: str | None, notional: Decimal, fraction: float = 0.5) -> list[str]:
    """Why ``symbol`` may not be traded; empty when it may."""
    if contract is None:
        return [f"{symbol}: not listed by the exchange"]
    problems = []
    if contract.kind not in PERPETUAL_KINDS or contract.quote != "USDT":
        problems.append(f"{symbol}: not a USDT perpetual ({contract.kind}, quoted in {contract.quote or '?'})")
    if contract.state != "open":
        problems.append(f"{symbol}: not trading ({contract.state})")
    if contract.delisting_at is not None:
        when = datetime.fromtimestamp(contract.delisting_at / 1000, timezone.utc).isoformat(timespec="minutes")
        problems.append(f"{symbol}: delisting scheduled at {when}")
    if tiers is None or not tiers.brackets:
        return problems + [f"{symbol}: margin tiers unknown"]
    if tiers.max_leverage() < leverage:
        problems.append(f"{symbol}: the exchange allows at most {tiers.max_leverage()}x, the configuration asks "
                        f"{leverage}x")
    if margin == "ISOLATED":
        bracket = tiers.at(notional)
        if bracket is None:
            problems.append(f"{symbol}: no margin tier for a position of {notional} USDT")
        else:
            limit = fraction * liquidation_distance(leverage, float(bracket.maintenance_rate))
            if float(stop_pct) > limit:
                problems.append(f"{symbol}: protective stop {float(stop_pct):.2%} is not inside {fraction:.0%} of the "
                                f"{leverage}x isolated liquidation distance at its maintenance margin rate "
                                f"{float(bracket.maintenance_rate):.2%} (at most {limit:.4f})")
    return problems


__all__ = ["Bracket", "Contract", "MarginTiers", "PERPETUAL_KINDS", "contract_problems"]

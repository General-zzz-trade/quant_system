"""One state per symbol (ADR-0014 point 3, S3d), derived every cycle from that cycle's problems and the kill
switch, drawdown and pause. Only READY lets an entry through; every state lets a reduction through.

The state is a classification with a fixed priority, so the allowed transitions are those the inputs can produce:
anything may follow anything, except going back to WARMING once warmed up (the bars of a running process are
never thrown away). A transition outside the table is treated as BLOCKED and alerted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SymbolState(Enum):
    WARMING = "warming"          # history not loaded yet
    READY = "ready"              # entries allowed
    BLOCKED = "blocked"          # a problem of this cycle blocks this symbol's (or every) entry
    PAUSED = "paused"            # an entry pause holds: a stop's ends by itself after 4 hours, others a person ends
    FLATTENING = "flattening"    # a flatten applies and the position is not flat yet
    HALTED = "halted"            # the kill switch is active or the drawdown guard is latched


@dataclass(frozen=True)
class StateInputs:
    warmed: bool
    flattening: bool
    halted: bool
    paused: bool
    blocked: bool


def classify(inputs: StateInputs) -> SymbolState:
    if not inputs.warmed:
        return SymbolState.WARMING
    if inputs.flattening:
        return SymbolState.FLATTENING
    if inputs.halted:
        return SymbolState.HALTED
    if inputs.paused:
        return SymbolState.PAUSED
    if inputs.blocked:
        return SymbolState.BLOCKED
    return SymbolState.READY


ALLOWED: frozenset[tuple[SymbolState, SymbolState]] = frozenset(
    (a, b) for a in SymbolState for b in SymbolState if b is not SymbolState.WARMING or a is SymbolState.WARMING)


__all__ = ["ALLOWED", "StateInputs", "SymbolState", "classify"]

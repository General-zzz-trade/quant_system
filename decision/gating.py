from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class GateResult:
    allowed: bool
    reasons: Sequence[str]


class Gate:
    def check(self, snapshot: object) -> GateResult:  # pragma: no cover (interface)
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FeatureFlagGate(Gate):
    required: Sequence[str] = ()

    def check(self, snapshot: object) -> GateResult:
        flags = getattr(snapshot, "feature_flags", None)
        if not self.required:
            return GateResult(True, ())
        if not isinstance(flags, Mapping):
            return GateResult(False, ("missing_feature_flags",))
        missing = [f for f in self.required if not bool(flags.get(f, False))]
        if missing:
            return GateResult(False, tuple([f"missing:{m}" for m in missing]))
        return GateResult(True, ())


@dataclass(frozen=True, slots=True)
class SymbolBlacklistGate(Gate):
    blacklist: Sequence[str] = ()

    def check(self, snapshot: object) -> GateResult:
        sym = getattr(snapshot, "symbol", None)
        if sym is None:
            return GateResult(True, ())
        if sym in set(self.blacklist):
            return GateResult(False, (f"blacklisted:{sym}",))
        return GateResult(True, ())

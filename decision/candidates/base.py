from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from decision.types import Candidate, SignalResult


class CandidateGenerator(Protocol):
    def generate(self, signals: Sequence[SignalResult]) -> Sequence[Candidate]:
        ...


@dataclass(frozen=True, slots=True)
class PassthroughCandidates:
    def generate(self, signals: Sequence[SignalResult]) -> Sequence[Candidate]:
        out = []
        for s in signals:
            if s.side == "flat" or s.score == 0:
                continue
            side = "buy" if s.score > 0 else "sell"
            out.append(Candidate(symbol=s.symbol, score=s.score, side=side, meta=s.meta))
        return out

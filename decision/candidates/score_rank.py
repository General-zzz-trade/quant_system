from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from decision.types import Candidate, SignalResult, Side


@dataclass(frozen=True, slots=True)
class ScoreRankCandidates:
    max_candidates: int = 5

    def generate(self, signals: Sequence[SignalResult]) -> Sequence[Candidate]:
        cands = []
        for s in signals:
            if s.side == "flat" or s.score == 0:
                continue
            side: Side = "buy" if s.score > 0 else "sell"
            cands.append(Candidate(symbol=s.symbol, score=s.score, side=side, meta=s.meta))
        cands.sort(key=lambda c: abs(c.score), reverse=True)
        return cands[: self.max_candidates]

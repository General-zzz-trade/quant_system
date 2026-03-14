"""Auto-generate candidate features from existing feature operators."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureCandidate:
    """A candidate feature ready for evaluation.

    Attributes:
        name: Human-readable identifier (e.g. ``sma_20``).
        compute_fn: Callable that takes bars and returns feature values.
        category: Origin category: ``technical``, ``rolling``, ``composite``, or ``auto``.
    """
    name: str
    compute_fn: Callable
    category: str


class FeatureGenerator:
    """Auto-generate candidate features from existing feature operators.

    Combines registered operators with different look-back windows to produce
    a combinatorial set of feature candidates for downstream selection.

    Usage:
        gen = FeatureGenerator()
        gen.register_operator("sma", sma_fn, category="technical")
        gen.register_operator("rsi", rsi_fn, category="technical")
        candidates = gen.generate_candidates(windows=(5, 10, 20, 50))
    """

    def __init__(self, *, operators: Optional[Sequence[Callable]] = None) -> None:
        self._named_operators: list[tuple[str, Callable]] = [
            (getattr(op, "__name__", "op"), op) for op in (operators or [])
        ]
        self._candidates: list[FeatureCandidate] = []

    def register_operator(
        self,
        name: str,
        fn: Callable,
        category: str = "technical",
    ) -> None:
        """Register a named operator for candidate generation."""
        self._named_operators.append((name, fn))
        self._candidates.append(
            FeatureCandidate(name=name, compute_fn=fn, category=category),
        )

    def generate_candidates(
        self,
        *,
        windows: Sequence[int] = (5, 10, 20, 50),
    ) -> list[FeatureCandidate]:
        """Generate candidate features by combining operators with different windows.

        For each registered operator and each window size, creates a new
        ``FeatureCandidate`` that wraps ``operator(bars, window)``.

        Returns:
            List of all candidates (explicit registrations + auto-generated).
        """
        candidates = list(self._candidates)

        for op_name, op in self._named_operators:
            for w in windows:
                name = f"{op_name}_{w}"

                def _make_fn(bound_op: Callable = op, bound_w: int = w) -> Callable:
                    return lambda bars: bound_op(bars, bound_w)

                candidates.append(
                    FeatureCandidate(
                        name=name,
                        compute_fn=_make_fn(),
                        category="auto",
                    ),
                )

        logger.debug("Generated %d feature candidates", len(candidates))
        return candidates

    @property
    def operator_count(self) -> int:
        return len(self._named_operators)

    @property
    def explicit_candidates(self) -> list[FeatureCandidate]:
        return list(self._candidates)

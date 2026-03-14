from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from decision.candidates.score_rank import ScoreRankCandidates
from decision.candidates.filters import CandidateFilter
from decision.allocators.single_asset import SingleAssetAllocator
from decision.allocators.constraints import AllocationConstraints
from decision.sizing.fixed_fraction import FixedFractionSizer
from decision.intents.target_position import TargetPositionIntentBuilder
from decision.execution_policy.marketable_limit import MarketableLimitPolicy
from decision.execution_policy.passive import PassivePolicy


@dataclass(frozen=True, slots=True)
class DefaultComposer:
    """Builds default components for DecisionEngine."""

    def build_candidate_generator(self, max_candidates: int) -> ScoreRankCandidates:
        return ScoreRankCandidates(max_candidates=max_candidates)

    def build_candidate_filter(self) -> CandidateFilter:
        return CandidateFilter()

    def build_allocator(self) -> SingleAssetAllocator:
        return SingleAssetAllocator()

    def build_constraints(self, max_positions: int) -> AllocationConstraints:
        return AllocationConstraints(max_positions=max_positions)

    def build_sizer(self, fraction: Decimal) -> FixedFractionSizer:
        return FixedFractionSizer(fraction=fraction)

    def build_intent_builder(self) -> TargetPositionIntentBuilder:
        return TargetPositionIntentBuilder()

    def build_execution_policy(self, name: str, slippage_bps: Decimal):
        if name == "passive":
            return PassivePolicy(offset_bps=slippage_bps)
        return MarketableLimitPolicy(slippage_bps=slippage_bps)

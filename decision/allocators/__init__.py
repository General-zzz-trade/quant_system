# decision/allocators
"""Decision allocators."""
from decision.allocators.base import Allocator, EqualWeightAllocator
from decision.allocators.constraints import AllocationConstraints
from decision.allocators.single_asset import SingleAssetAllocator

__all__ = [
    "Allocator",
    "EqualWeightAllocator",
    "AllocationConstraints",
    "SingleAssetAllocator",
]

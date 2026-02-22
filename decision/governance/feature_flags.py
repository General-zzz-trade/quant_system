from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    flags: Mapping[str, bool]

    def enabled(self, key: str) -> bool:
        return bool(self.flags.get(key, False))

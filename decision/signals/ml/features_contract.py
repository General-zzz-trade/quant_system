from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class FeaturesContract:
    required: Sequence[str]
    optional: Sequence[str] = ()

    def validate(self, features: Mapping[str, object]) -> tuple[bool, list[str]]:
        missing = [k for k in self.required if k not in features]
        return (len(missing) == 0, missing)

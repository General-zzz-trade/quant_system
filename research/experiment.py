from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Experiment:
    """A minimal experiment definition.

    This object is intentionally framework-agnostic.
    Use it to define a run configuration and an output directory.
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def with_params(self, **kwargs: Any) -> "Experiment":
        p = dict(self.params)
        p.update(kwargs)
        return Experiment(name=self.name, params=p, dataset=dict(self.dataset), created_at=self.created_at)

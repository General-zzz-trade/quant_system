from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

from .experiment import Experiment


@dataclass
class WalkForwardSpec:
    """Walk-forward specification.

    train_window and test_window are user-defined units, typically bars or days.
    """

    train_window: int
    test_window: int
    step: int


def walk_forward(
    *,
    base: Experiment,
    spec: WalkForwardSpec,
    run_fn: Callable[[Experiment], Dict[str, Any]],
    windows: Iterable[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Run walk-forward over provided windows.

    windows yields (train_end, test_end) indexes in user space.
    """

    out: List[Dict[str, Any]] = []
    for i, (train_end, test_end) in enumerate(windows):
        exp = base.with_params(train_end=train_end, test_end=test_end, wf_i=i)
        metrics = run_fn(exp)
        row = {"experiment": exp.name, "wf_i": i, "train_end": train_end, "test_end": test_end}
        row.update(metrics)
        out.append(row)
    return out

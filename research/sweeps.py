from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

from .experiment import Experiment


def _grid(params: Dict[str, Sequence[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(params.keys())
    values = [list(params[k]) for k in keys]
    for combo in product(*values):
        yield {k: v for k, v in zip(keys, combo)}


@dataclass
class Sweep:
    """Simple parameter sweep runner.

    run_fn signature:
      run_fn(experiment: Experiment) -> dict metrics

    The sweep does not assume a particular backtest engine.
    """

    base: Experiment
    grid: Dict[str, Sequence[Any]]

    def run(self, run_fn: Callable[[Experiment], Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in _grid(self.grid):
            exp = self.base.with_params(**p)
            metrics = run_fn(exp)
            row = {"experiment": exp.name}
            row.update(exp.params)
            row.update(metrics)
            out.append(row)
        return out

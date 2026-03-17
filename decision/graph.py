from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    deps: Sequence[str] = ()
    fn: Callable[..., object] | None = None


@dataclass
class DecisionGraph:
    """Lightweight DAG container; execution handled externally."""
    nodes: Dict[str, Node]

    def topo(self) -> List[str]:
        visited: set[str] = set()
        order: list[str] = []
        def dfs(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            for d in self.nodes[n].deps:
                if d in self.nodes:
                    dfs(d)
            order.append(n)
        for n in self.nodes:
            dfs(n)
        return order

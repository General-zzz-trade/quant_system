from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterable, Optional, Protocol, Tuple, TypeVar, runtime_checkable

from state.errors import ReducerError

S = TypeVar("S")

@dataclass(frozen=True, slots=True)
class ReducerResult(Generic[S]):
    state: S
    changed: bool
    note: Optional[str] = None

@runtime_checkable
class Reducer(Protocol[S]):
    def reduce(self, state: S, event: Any) -> ReducerResult[S]: ...

def apply_one(state: S, reducer: Reducer[S], event: Any) -> ReducerResult[S]:
    try:
        res = reducer.reduce(state, event)
    except ReducerError:
        raise
    except Exception as e:
        raise ReducerError(f"{reducer.__class__.__name__} failed: {e}") from e
    if res.state is None:
        raise ReducerError(f"{reducer.__class__.__name__} returned None state")
    return res

def apply_all(state: S, reducers: Iterable[Reducer[S]], event: Any) -> Tuple[S, bool]:
    cur = state
    any_changed = False
    for r in reducers:
        res = apply_one(cur, r, event)
        cur = res.state
        any_changed = any_changed or res.changed
    return cur, any_changed

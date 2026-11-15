"""A problem found during a cycle (ADR-0014): its text, for people and the event log, plus the kind and symbol
that every trading decision uses.

``Issue`` is a ``str`` so the text stays exactly what the event log, alerts and tests have always shown; code
in this layer never reads that text to decide anything (tests/architecture/test_issue_rules.py).
"""
from __future__ import annotations

from trading.domain.issues import Effect, IssueKind, effects


class Issue(str):
    kind: IssueKind
    symbol: str | None

    def __new__(cls, kind: IssueKind, symbol: str | None, text: str) -> "Issue":
        if not isinstance(kind, IssueKind):
            raise TypeError(f"an issue needs an IssueKind, not {kind!r}")
        issue = super().__new__(cls, text)
        issue.kind, issue.symbol = kind, symbol
        return issue

    @property
    def effects(self) -> frozenset[Effect]:
        return effects(self.kind)


def with_effect(issues: list[Issue], effect: Effect) -> list[Issue]:
    return [issue for issue in issues if effect in issue.effects]


__all__ = ["Issue", "with_effect"]

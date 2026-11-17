"""After a stop, wait for the strategy's reset (tech lead 2026-11-17).

With the spec's ``after_stop: pause_then_reset``, once a stop's pause has ended by itself, a flat symbol takes no
entry until its strategy has decided a target of exactly 0 since that pause began. One rule, used by the entry gate
(before the order limiter counts anything; audit of PR #142, finding 1) and by the exit side's pause status.

The wait ends for good once an entry of the symbol has filled after the pause began (the order journal knows, also
after a restart; audit finding 2), or once a person releases it (``entry_pause release``, which records a release
row: then the latest pause is no longer one that ended by itself). A 0 seen before a restart is not remembered:
until the next one, or an entry, the symbol waits. Reads that fail raise; callers block.
"""
from __future__ import annotations

from trading.application.ports import EntryPauses
from trading.application.session import TargetSession
from trading.domain.issues import AFTER_STOP_PAUSE_THEN_RESET

RESET_WAIT = "waiting for the strategy's target to return to 0 after a stop"


def waits_for_reset(session: TargetSession, pauses: EntryPauses) -> bool:
    if session.after_stop != AFTER_STOP_PAUSE_THEN_RESET:
        return False
    since = pauses.ended_by_itself(session.symbol)
    if since is None or session.position_qty() != 0:
        return False
    seen = session.flat_target_at
    if seen is not None and seen > since * 1000:
        return False
    entered = session.entered_since
    return not (entered is not None and entered(since))


__all__ = ["RESET_WAIT", "waits_for_reset"]

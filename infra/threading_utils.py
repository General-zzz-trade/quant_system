from __future__ import annotations

import threading


def safe_join_thread(thread: threading.Thread | None, *, timeout: float) -> None:
    """Join a thread if it actually started, swallowing startup/shutdown races."""
    if thread is None:
        return
    try:
        if thread.is_alive():
            thread.join(timeout=timeout)
    except RuntimeError:
        # Thread may not have fully started yet when stop() races with start().
        return

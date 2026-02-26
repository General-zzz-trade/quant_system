"""FundingCollector — periodically fetches funding rates."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FUNDING_INTERVAL_S = 8 * 60 * 60  # 8 hours


class FundingCollector:
    """Periodically fetches funding rate data and stores it.

    Parameters
    ----------
    fetch_fn:
        Callable that returns a list of funding-rate dicts.
    store_fn:
        Callable that receives the list of funding-rate dicts for persistence.
    interval:
        Seconds between fetches (default 8 hours).
    """

    def __init__(
        self,
        fetch_fn: Callable[[], List[Dict[str, Any]]],
        store_fn: Callable[[List[Dict[str, Any]]], None],
        *,
        interval: float = FUNDING_INTERVAL_S,
    ) -> None:
        self._fetch_fn = fetch_fn
        self._store_fn = store_fn
        self._interval = interval

        self._running = False
        self._timer: Optional[threading.Timer] = None
        self._last_active: Optional[float] = None

    # -- Collector protocol --------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._schedule_next(run_now=True)
        logger.info("FundingCollector started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("FundingCollector stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_active_ts(self) -> Optional[float]:
        return self._last_active

    # -- Internal ------------------------------------------------------------

    def _schedule_next(self, *, run_now: bool = False) -> None:
        if not self._running:
            return
        if run_now:
            self._run()
        else:
            self._timer = threading.Timer(self._interval, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        try:
            data = self._fetch_fn()
            if data:
                self._store_fn(data)
            self._last_active = time.monotonic()
            logger.debug("Fetched %d funding records", len(data) if data else 0)
        except Exception:
            logger.exception("FundingCollector fetch failed")
        finally:
            self._schedule_next()

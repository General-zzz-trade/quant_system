"""Webhook alert sink — POSTs alerts to an HTTP endpoint.

Suitable for Telegram bots, Discord webhooks, Slack incoming webhooks,
or any custom alerting endpoint.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from .base import Alert, Severity

logger = logging.getLogger(__name__)


@dataclass
class WebhookAlertSink:
    """POST alerts as JSON to a webhook URL.

    Parameters
    ----------
    url : str
        Target webhook URL.
    min_severity : Severity
        Only alerts at or above this level are sent (default: WARNING).
    retries : int
        Number of retry attempts on transient failure.
    timeout_seconds : float
        HTTP request timeout.
    headers : dict
        Extra HTTP headers (e.g., Authorization).
    """
    url: str
    min_severity: Severity = Severity.WARNING
    retries: int = 2
    timeout_seconds: float = 5.0
    headers: Dict[str, str] = field(default_factory=dict)

    # Simple circuit breaker state
    _consecutive_failures: int = field(default=0, init=False)
    _circuit_open_until: float = field(default=0.0, init=False)
    _CIRCUIT_THRESHOLD: int = 5
    _CIRCUIT_RESET_SECONDS: float = 60.0

    def emit(self, alert: Alert) -> None:
        # Severity filter
        levels = list(Severity)
        if levels.index(alert.severity) < levels.index(self.min_severity):
            return

        # Circuit breaker check
        if self._consecutive_failures >= self._CIRCUIT_THRESHOLD:
            if time.monotonic() < self._circuit_open_until:
                logger.debug("webhook circuit open, skipping alert: %s", alert.title)
                return
            # Try to close circuit
            self._consecutive_failures = 0

        payload = json.dumps(alert.to_dict()).encode("utf-8")
        req = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )

        last_err: Optional[Exception] = None
        for attempt in range(1 + self.retries):
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp.read()  # drain response
                self._consecutive_failures = 0
                return
            except (URLError, OSError) as exc:
                last_err = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2 ** attempt))  # exponential backoff

        # All retries exhausted
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._CIRCUIT_THRESHOLD:
            self._circuit_open_until = time.monotonic() + self._CIRCUIT_RESET_SECONDS
        logger.warning("webhook alert failed after %d attempts: %s", 1 + self.retries, last_err)

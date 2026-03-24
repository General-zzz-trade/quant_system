"""Module-level helpers for LiveRunner — extracted from live_runner.py.

Contains: _setup_systemd_notify, _start_optional, _install_sighup,
_start_user_stream, _check_timeouts, _reconcile_startup, _FillRecordingAdapter.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List

from execution.observability.incidents import timeout_to_alert

logger = logging.getLogger(__name__)


def _setup_systemd_notify():
    """Set up systemd watchdog notify. Returns notify function or None."""
    try:
        import socket
        _sd_addr = os.environ.get("NOTIFY_SOCKET")
        if _sd_addr:
            _sd_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            if _sd_addr.startswith("@"):
                _sd_addr = "\0" + _sd_addr[1:]

            def _sd_notify_fn(msg: str) -> None:
                try:
                    _sd_sock.sendto(msg.encode(), _sd_addr)
                except Exception as e:
                    logger.error("Failed to send systemd notify '%s': %s", msg, e, exc_info=True)

            _sd_notify_fn("READY=1")
            logger.info("Systemd notify: READY=1")
            return _sd_notify_fn
    except Exception as e:
        logger.error("Failed to initialize systemd watchdog: %s", e, exc_info=True)
    return None


def _start_optional(subsystem: Any) -> None:
    """Start a subsystem if it is not None."""
    if subsystem is not None:
        subsystem.start()


def _install_sighup(runner: Any) -> None:
    """Install SIGHUP handler for model hot-reload."""
    import signal as _signal

    def _sighup_handler(signum: int, frame: Any) -> None:
        logger.info("SIGHUP received -- scheduling model reload")
        runner._reload_models_pending = True

    try:
        if threading.current_thread() is threading.main_thread():
            _signal.signal(_signal.SIGHUP, _sighup_handler)
        else:
            logger.warning("Skipping LiveRunner SIGHUP handler: not running in main thread")
    except (OSError, AttributeError, ValueError) as e:
        logger.warning("Failed to install SIGHUP handler: %s", e)


def _start_user_stream(runner: Any) -> None:
    """Start user stream in a background thread."""
    if runner.user_stream is None:
        return

    def _user_stream_loop() -> None:
        try:
            runner.user_stream.connect()
            runner._record_user_stream_connect()
        except Exception:
            runner._record_user_stream_failure(kind="connect")
            logger.warning("User stream initial connect failed", exc_info=True)
            return
        _backoff = 1.0
        _MAX_BACKOFF = 60.0
        while runner._running:
            try:
                runner.user_stream.step()
                _backoff = 1.0
            except Exception:
                runner._record_user_stream_failure(kind="step")
                logger.warning("User stream step error, reconnecting in %.0fs",
                               _backoff, exc_info=True)
                time.sleep(_backoff)
                try:
                    runner.user_stream.connect()
                    runner._record_user_stream_connect()
                    _backoff = 1.0
                except Exception:
                    runner._record_user_stream_failure(kind="reconnect")
                    logger.warning("User stream reconnect failed", exc_info=True)
                    _backoff = min(_backoff * 2, _MAX_BACKOFF)

    t = threading.Thread(target=_user_stream_loop, daemon=True, name="user-stream")
    t.start()
    runner._user_stream_thread = t
    logger.info("User stream thread started")


def _check_timeouts(runner: Any) -> None:
    """Check for timed-out orders and emit alerts."""
    timed_out = runner.timeout_tracker.check_timeouts()
    if timed_out:
        logger.warning("Timed out orders: %s", timed_out)
        venue = str(getattr(getattr(runner, "_config", None), "venue", ""))
        timeout_sec = float(getattr(runner.timeout_tracker, "timeout_sec", 0.0))
        for order_id in timed_out:
            try:
                runner._emit_execution_incident(
                    timeout_to_alert(
                        venue=venue, symbol="*",
                        order_id=str(order_id), timeout_sec=timeout_sec,
                    )
                )
            except Exception:
                logger.exception("timeout alert emit failed for order=%s", order_id)


def _reconcile_startup(
    local_view: Dict[str, Any],
    venue_state: Dict[str, Any],
    symbols: tuple[str, ...],
) -> List[str]:
    """Compare local state against exchange state. Returns list of mismatch descriptions."""
    mismatches: List[str] = []
    venue_positions = venue_state.get("positions", {})
    local_positions = local_view.get("positions", {})

    for sym in symbols:
        local_pos = local_positions.get(sym)
        venue_pos = venue_positions.get(sym)
        local_qty = float(getattr(local_pos, "qty", 0) if local_pos else 0)
        venue_qty = float(venue_pos.get("qty", 0) if isinstance(venue_pos, dict) else 0)
        if abs(local_qty - venue_qty) > 1e-8:
            mismatches.append(f"{sym} position: local={local_qty}, venue={venue_qty}")

    local_account = local_view.get("account")
    local_balance = float(getattr(local_account, "balance", 0) if local_account else 0)
    venue_balance = float(venue_state.get("balance", 0))
    if abs(local_balance - venue_balance) > 0.01:
        mismatches.append(f"Balance: local={local_balance:.2f}, venue={venue_balance:.2f}")
    return mismatches


class _FillRecordingAdapter:
    """Thin wrapper that intercepts fill events from send_order results."""

    def __init__(self, inner: Any, on_fill: Callable[[Any], None]) -> None:
        self._inner = inner
        self._on_fill = on_fill

    def send_order(self, order_event: Any) -> list:
        results = list(self._inner.send_order(order_event))
        for ev in results:
            et = getattr(getattr(ev, "event_type", None), "value", "")
            if "fill" in str(et).lower():
                self._on_fill(ev)
        return results

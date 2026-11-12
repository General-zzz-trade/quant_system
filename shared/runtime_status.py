"""Atomic main-loop observations and independent, read-only health evaluation.

The runner publishes these observations itself. There is deliberately no timer
or worker thread: a stalled main loop must eventually lose its heartbeat.
Status is operational evidence, never permission to bypass execution gates.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Callable
import uuid

STATUS_NAME = "runtime-status.json"
MAX_STATUS_BYTES = 262_144
HEARTBEAT_MAX_AGE = 30.0
RECONCILIATION_MAX_AGE = 120.0
RISK_MAX_AGE = 300.0
_DATA_FIELDS = (
    "status", "ready", "mode", "bar_ts_ms", "interval_ms", "valid_warmup_bars",
    "required_warmup_bars", "open_interest", "ls_ratio", "oi_event_ts_ms",
    "ls_event_ts_ms", "received_at_ms",
)


def _number(value: Any, name: str, *, minimum: float = 0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid runtime status " + name)
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid runtime status " + name)
    return value


def _text(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value or value.strip() != value
            or len(value) > 256 or any(ord(char) < 32 for char in value)):
        raise ValueError("Invalid runtime status " + name)
    return value


def _uuid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Invalid runtime status " + name)
    return value


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Invalid runtime status " + name)
    return value


def _flag(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError("Invalid runtime status " + name)
    return value


def _path(path: Path) -> Path:
    result = Path(path).absolute()
    if ".." in result.parts or any(item.is_symlink() for item in (result, *result.parents)):
        raise ValueError("Runtime status path cannot contain traversal or symbolic links")
    return result


def strict_json(raw: str) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate runtime status JSON key")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError("Non-finite runtime status JSON number")

    def floating(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite runtime status JSON number")
        return number

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    return _object(result, "document")


def read_json(path: Path) -> dict:
    """Read one bounded regular file without following a final symbolic link."""
    path = _path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STATUS_BYTES:
            raise ValueError("Runtime status must be a bounded regular file")
        raw = stream.read(MAX_STATUS_BYTES + 1)
    if len(raw) > MAX_STATUS_BYTES:
        raise ValueError("Runtime status is too large")
    return strict_json(raw.decode("utf-8"))


def _release(value: Any) -> dict:
    result = _object(value, "release")
    if not result:
        return {}
    patterns = {"revision": r"[0-9a-f]{40}", "image_id": r"sha256:[0-9a-f]{64}",
                "input_digest": r"[0-9a-f]{64}"}
    if set(result) != set(patterns):
        raise ValueError("Runtime release identity must be complete")
    for key, pattern in patterns.items():
        if not isinstance(result[key], str) or not re.fullmatch(pattern, result[key]):
            raise ValueError("Invalid runtime release " + key)
    return dict(result)


def _fresh(timestamp: Any, now: float, maximum: float, name: str) -> bool:
    observed = _number(timestamp, name, minimum=0.000001)
    return 0 <= now - observed <= maximum


def _process_identity(pid: int) -> dict:
    """Linux process incarnation; PID 1 alone is reusable after container restart."""
    raw = Path(f"/proc/{pid}/stat").read_text()
    # comm is parenthesized and may contain spaces or parentheses. The fields
    # after its final ')' begin with field 3; starttime is field 22.
    fields = raw[raw.rfind(")") + 2:].split()
    ticks = int(fields[19])
    boot = uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()).hex
    namespace = os.readlink(f"/proc/{pid}/ns/pid")
    return {"process_start_ticks": ticks, "system_boot_id": boot, "pid_namespace": namespace}


def _data_ready(row: Any, now: float) -> bool:
    row = _object(row, "data observation")
    ready = _flag(row.get("ready"), "data ready")
    if not ready:
        return False
    if row.get("mode") != "live" or row.get("status") != "ready":
        return False
    required = _integer(row.get("required_warmup_bars"), "required warmup", minimum=1)
    if _integer(row.get("valid_warmup_bars"), "valid warmup") < required:
        return False
    interval = _integer(row.get("interval_ms"), "bar interval", minimum=1)
    bar = _integer(row.get("bar_ts_ms"), "bar timestamp", minimum=1)
    if not 0 <= now * 1000 - bar <= interval + 300_000:
        return False
    receipt = _integer(row.get("received_at_ms"), "data receipt", minimum=1)
    if not 0 <= now * 1000 - receipt <= 120_000:
        return False
    for name, key in (("open_interest", "oi_event_ts_ms"), ("ls_ratio", "ls_event_ts_ms")):
        _number(row.get(name), name, minimum=0.000000000001)
        observed = _integer(row.get(key), key, minimum=1)
        if not 0 <= bar - observed <= 3_900_000:
            return False
    conditions = _object(row.get("external_conditions"), "external data conditions")
    for name, condition in conditions.items():
        _text(name, "external condition name")
        condition = _object(condition, "external condition")
        if not _flag(condition.get("ready"), "external condition ready"):
            return False
    return True


def evaluate_status(status: dict, *, now: float | None = None,
                    expected_scope: str | None = None, expected_state_id: str | None = None,
                    expected_release: dict | None = None, check_pid: bool = False) -> dict:
    """Recompute health from evidence; advertised readiness flags are ignored."""
    now = _number(time.time() if now is None else now, "clock", minimum=0.000001)
    status = _object(status, "document")
    if _integer(status.get("version"), "version", minimum=1) != 1:
        raise ValueError("Unsupported runtime status version")
    scope = _text(status.get("scope"), "scope")
    state_id = _uuid(status.get("state_id"), "state identity")
    _uuid(status.get("boot_id"), "boot identity")
    pid = _integer(status.get("pid"), "process id", minimum=1)
    process_start = _integer(status.get("process_start_ticks"), "process start ticks", minimum=1)
    system_boot = _uuid(status.get("system_boot_id"), "system boot identity")
    namespace = status.get("pid_namespace")
    if not isinstance(namespace, str) or not re.fullmatch(r"pid:\[[0-9]+\]", namespace):
        raise ValueError("Invalid runtime PID namespace identity")
    if expected_scope is not None and scope != expected_scope:
        raise ValueError("Runtime status scope differs from configured state")
    if expected_state_id is not None and state_id != expected_state_id:
        raise ValueError("Runtime status identity differs from configured state")
    mode, phase = status.get("mode"), status.get("phase")
    if mode not in {"paper", "shadow", "execution"} or phase not in {"starting", "running", "stopping", "stopped"}:
        raise ValueError("Invalid runtime phase or mode")
    manifest_digest = status.get("manifest_digest")
    if mode == "shadow" and (not isinstance(manifest_digest, str)
                             or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)):
        raise ValueError("Shadow runtime requires its actual loaded manifest digest")
    runners = status.get("runner_keys")
    if not isinstance(runners, list) or not runners or len(runners) > 128:
        raise ValueError("Runtime status needs the configured runner set")
    for key in runners:
        _text(key, "runner key")
    if len(set(runners)) != len(runners):
        raise ValueError("Runtime status runner set contains duplicates")
    execution_runners = status.get("execution_runner_keys")
    if (not isinstance(execution_runners, list)
            or any(not isinstance(key, str) for key in execution_runners)
            or len(set(execution_runners)) != len(execution_runners)
            or not set(execution_runners) <= set(runners)
            or (mode == "execution" and not execution_runners)
            or (mode in {"paper", "shadow"} and execution_runners)):
        raise ValueError("Runtime status needs an explicit execution runner set")
    release = _release(status.get("release"))
    if expected_release is not None and release != _release(expected_release):
        raise ValueError("Runtime status release identity differs from expected release")
    started = _number(status.get("started_at"), "start timestamp", minimum=0.000001)
    heartbeat = _number(status.get("heartbeat_at"), "heartbeat timestamp", minimum=0.000001)
    if heartbeat < started:
        raise ValueError("Runtime heartbeat precedes process start")
    reasons = []
    fresh = 0 <= now - heartbeat <= HEARTBEAT_MAX_AGE
    if not fresh:
        reasons.append("heartbeat_stale_or_future")
    process_alive = True
    if check_pid:
        try:
            os.kill(pid, 0)
            process_alive = _process_identity(pid) == {
                "process_start_ticks": process_start, "system_boot_id": system_boot, "pid_namespace": namespace,
            }
        except (OSError, ValueError, IndexError, OverflowError):
            process_alive = False
        if not process_alive:
            reasons.append("runtime_process_unavailable")
    live = fresh and process_alive and phase in {"starting", "running"}
    if phase != "running":
        reasons.append("runtime_not_running")
    data = _object(status.get("data"), "data observations")
    data_ready = set(data) == set(runners) and all(_data_ready(row, now) for row in data.values())
    if not data_ready:
        reasons.append("data_not_ready")
    ready = live and phase == "running" and data_ready
    eligible = False
    execution = status.get("execution")
    if mode in {"paper", "shadow"}:
        if execution is not None:
            raise ValueError("Paper or shadow runtime cannot assert execution observations")
    elif execution is None:
        ready = False
        reasons.append("execution_observations_unavailable")
    else:
        execution = _object(execution, "execution observations")
        modules = _object(execution.get("modules"), "execution modules")
        reconciles = _object(execution.get("reconciliations"), "execution reconciliations")
        execution_ready = set(modules) == set(execution_runners) and set(reconciles) == set(execution_runners)
        for key in execution_runners:
            if key not in modules or key not in reconciles:
                execution_ready = False
                continue
            module_ready = _flag(modules[key], "module readiness")
            record = _object(reconciles[key], "execution reconciliation")
            reconciled = _flag(record.get("ready"), "execution reconciliation ready")
            checked = record.get("checked_at")
            recent = checked is not None and _fresh(checked, now, RECONCILIATION_MAX_AGE, "reconciliation time")
            execution_ready = execution_ready and module_ready and reconciled and recent
        risk = _object(execution.get("risk"), "portfolio risk")
        risk_ready = _flag(risk.get("ready"), "portfolio risk ready")
        age = risk.get("age_seconds")
        risk_limit = min(RISK_MAX_AGE, _number(risk.get("max_age_seconds"), "portfolio risk age limit", minimum=0.001))
        captured = _number(risk.get("captured_at"), "portfolio risk capture time", minimum=0.000001)
        risk_recent = (age is not None and 0 <= now - captured
                       and 0 <= _number(age, "portfolio risk age") + now - captured <= risk_limit)
        wallet = _object(execution.get("wallet"), "wallet")
        wallet_ready = _flag(wallet.get("ready"), "wallet ready")
        wallet_limit = min(RECONCILIATION_MAX_AGE,
                           _number(wallet.get("max_age_seconds"), "wallet age limit", minimum=0.001))
        wallet_checked, through = wallet.get("checked_at"), wallet.get("verified_through_ms")
        wallet_recent = (wallet_checked is not None and through is not None
                         and _fresh(wallet_checked, now, wallet_limit, "wallet check time")
                         and _fresh(_integer(through, "wallet watermark", minimum=1) / 1000,
                                    now, wallet_limit, "wallet watermark"))
        journal = _object(execution.get("journal"), "journal counts")
        clean_journal = all(_integer(journal.get(key), "journal " + key) == 0
                            for key in ("unresolved", "unknown", "unconsumed"))
        if not execution_ready:
            reasons.append("execution_reconciliation_not_ready")
        if not (risk_ready and risk_recent):
            reasons.append("portfolio_risk_not_ready")
        if not (wallet_ready and wallet_recent):
            reasons.append("wallet_not_ready")
        if not clean_journal:
            reasons.append("execution_journal_not_clear")
        eligible = ready and execution_ready and risk_ready and risk_recent and wallet_ready and wallet_recent
        eligible = bool(eligible and clean_journal)
        ready = eligible
    return {"version": 1, "live": live, "ready": ready, "execution_eligible": eligible,
            "mode": mode, "phase": phase, "reasons": reasons, "scope": scope,
            "state_id": state_id, "boot_id": status["boot_id"], "heartbeat_at": heartbeat,
            "release": release, "manifest_digest": manifest_digest}


class RuntimeStatusWriter:
    """Called synchronously by the owner of the existing runtime writer lease."""

    def __init__(self, path: Path, *, scope: str, state_id: str, mode: str,
                 runner_keys: list[str], release: dict | None = None,
                 execution_runner_keys: list[str] | None = None,
                 manifest_digest: str | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = _path(path)
        if not self.path.parent.is_dir():
            raise ValueError("Runtime status directory must already exist")
        self.clock = clock
        execution_keys = execution_runner_keys
        if execution_keys is None:
            execution_keys = runner_keys if mode == "execution" else []
        self.identity = {"version": 1, "scope": scope, "state_id": state_id, "mode": mode,
                         "runner_keys": list(runner_keys), "release": _release(release or {}),
                         "execution_runner_keys": list(execution_keys),
                         "manifest_digest": manifest_digest,
                         "boot_id": uuid.uuid4().hex, "pid": os.getpid(), "started_at": clock(),
                         **_process_identity(os.getpid())}

    def publish(self, phase: str, *, data: dict | None = None, execution: dict | None = None) -> dict:
        observations = {}
        for key, row in (data or {}).items():
            observations[key] = {field: row[field] for field in _DATA_FIELDS if field in row}
            observations[key]["external_conditions"] = {
                name: {"ready": condition.get("ready")}
                for name, condition in row.get("external_conditions", {}).items()
            }
        sanitized_execution = None
        if execution is not None:
            sanitized_execution = {
                "modules": dict(execution.get("modules", {})),
                "reconciliations": {
                    key: {name: row.get(name) for name in ("ready", "checked_at")}
                    for key, row in execution.get("reconciliations", {}).items()
                },
                "risk": {key: execution.get("risk", {}).get(key)
                         for key in ("ready", "age_seconds", "captured_at", "max_age_seconds")},
                "wallet": {key: execution.get("wallet", {}).get(key)
                           for key in ("ready", "checked_at", "verified_through_ms", "max_age_seconds")},
                "journal": {key: execution.get("journal", {}).get(key)
                            for key in ("unresolved", "unknown", "unconsumed")},
            }
        status = {**self.identity, "phase": phase, "heartbeat_at": self.clock(),
                  "data": observations, "execution": sanitized_execution}
        # Validate before replacing the last complete observation. A failed
        # publication cannot keep that observation fresh.
        result = evaluate_status(status, now=status["heartbeat_at"])
        raw = json.dumps(status, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(raw) > MAX_STATUS_BYTES:
            raise ValueError("Runtime status is too large")
        path = _path(self.path)
        descriptor, temporary = tempfile.mkstemp(prefix=".runtime-status-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return result

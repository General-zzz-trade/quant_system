"""ADR-0016 point 4 (S8): three runtime units, a failure alert for each, hourly state backups.

- Paper (``binance-target``): as it was.
- Testnet execution: ``Restart=on-failure`` after 60 s; ``StartLimitIntervalSec=3600`` and ``StartLimitBurst=3`` in
  ``[Unit]`` (in ``[Service]`` systemd ignores them); the network is fixed to the testnet in the command itself,
  whatever ``.env`` says, so a unit that restarts by itself never reaches mainnet.
- Mainnet execution: ``Restart=no``, no ``[Install]`` (a person starts it; it never starts at boot).
- Every execution and backup unit sends ``OnFailure=`` to ``quant-alert@``.
Host layout (tech lead, 2026-11-16): one network per host; state and backups under /var/lib/quant.
"""
from __future__ import annotations

from pathlib import Path

UNITS = Path(__file__).resolve().parents[3] / "infra" / "systemd"


def _unit(name: str) -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for raw in (UNITS / name).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("["):
            current = line.strip("[]")
            sections.setdefault(current, [])
            continue
        key, _, value = line.partition("=")
        assert current is not None, f"{name}: {line} outside a section"
        sections[current].append((key.strip(), value.strip()))
    return sections


def _get(unit, section, key):
    values = [v for k, v in unit.get(section, []) if k == key]
    return values[-1] if values else None


def _execution_units():
    return [p.name for p in sorted(UNITS.glob("*.service")) if "--enable-execution" in p.read_text()]


def test_the_testnet_unit_restarts_a_limited_number_of_times_and_only_on_the_testnet():
    unit = _unit("binance-target-testnet.service")
    assert _get(unit, "Service", "Restart") == "on-failure" and _get(unit, "Service", "RestartSec") == "60"
    assert _get(unit, "Unit", "StartLimitIntervalSec") == "3600" and _get(unit, "Unit", "StartLimitBurst") == "3"
    assert _get(unit, "Service", "StartLimitIntervalSec") is None and _get(unit, "Service", "StartLimitBurst") is None
    command = _get(unit, "Service", "ExecStart")
    assert command.startswith("/usr/bin/env BINANCE_TESTNET=1 ")
    assert "--enable-execution" in command and "--state-dir /var/lib/quant/state/testnet" in command
    assert _get(unit, "Unit", "OnFailure") == "quant-alert@%n.service"


def test_the_mainnet_unit_never_restarts_and_never_starts_at_boot():
    unit = _unit("binance-target-mainnet.service")
    assert _get(unit, "Service", "Restart") == "no"
    assert "Install" not in unit
    command = _get(unit, "Service", "ExecStart")
    assert command.startswith("/usr/bin/env BINANCE_TESTNET=0 ")
    assert "--enable-execution" in command and "--state-dir /var/lib/quant/state/mainnet" in command
    assert _get(unit, "Unit", "OnFailure") == "quant-alert@%n.service"


def test_the_paper_unit_is_as_it_was():
    unit = _unit("binance-target.service")
    command = _get(unit, "Service", "ExecStart")
    assert "--dry-run" in command and "--enable-execution" not in command


def test_an_execution_unit_that_restarts_by_itself_is_fixed_to_the_testnet():
    units = _execution_units()
    assert sorted(units) == ["binance-target-mainnet.service", "binance-target-testnet.service"]
    for name in units:
        unit = _unit(name)
        if (_get(unit, "Service", "Restart") or "no") != "no":
            assert _get(unit, "Service", "ExecStart").startswith("/usr/bin/env BINANCE_TESTNET=1 "), name


def test_every_failure_alert_goes_to_a_unit_that_exists():
    assert (UNITS / "quant-alert@.service").is_file()
    alert = _unit("quant-alert@.service")
    assert _get(alert, "Service", "Type") == "oneshot"
    assert _get(alert, "Service", "ExecStart").endswith("-m trading.infrastructure.unit_alert %i")
    for path in sorted(UNITS.glob("*.service")):
        target = _get(_unit(path.name), "Unit", "OnFailure")
        if target is not None:
            assert target == "quant-alert@%n.service", path.name
    assert "OnFailure" not in dict(alert.get("Unit", []))                # an alert never alerts on itself


def test_the_hourly_backup_of_a_network_s_state():
    # Enabled per network as state-backup@testnet.timer (the instance names the state and backup directories).
    service = _unit("state-backup@.service")
    command = _get(service, "Service", "ExecStart")
    assert command.endswith("-m trading.infrastructure.state_backup --state-dir /var/lib/quant/state/%i "
                            "--backup-dir /var/lib/quant/backups/%i")
    assert _get(service, "Service", "Type") == "oneshot"
    assert _get(service, "Unit", "OnFailure") == "quant-alert@%n.service"
    timer = _unit("state-backup@.timer")
    assert _get(timer, "Timer", "OnCalendar") == "hourly" and _get(timer, "Install", "WantedBy") == "timers.target"


def test_the_execution_units_run_as_the_service_user_from_the_checkout():
    for name in _execution_units() + ["quant-alert@.service", "state-backup@.service"]:
        unit = _unit(name)
        assert _get(unit, "Service", "User") == "ubuntu", name
        assert _get(unit, "Service", "WorkingDirectory") == "/quant_system", name

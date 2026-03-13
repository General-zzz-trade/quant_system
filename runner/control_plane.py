from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True, slots=True)
class OperatorControlRequest:
    command: str
    reason: str = "manual_control"
    source: str = "operator"

    @classmethod
    def from_any(cls, control: Any) -> "OperatorControlRequest":
        if isinstance(control, cls):
            return control
        if isinstance(control, Mapping):
            return cls(
                command=str(control.get("command", "")).strip().lower(),
                reason=str(control.get("reason", "")).strip() or "manual_control",
                source=str(control.get("source", "")).strip() or "operator",
            )
        return cls(
            command=str(getattr(control, "command", "")).strip().lower(),
            reason=str(getattr(control, "reason", "")).strip() or "manual_control",
            source=str(getattr(control, "source", "")).strip() or "operator",
        )


@dataclass(frozen=True, slots=True)
class OperatorControlResult:
    accepted: bool
    command: str
    outcome: str
    reason: str
    source: str
    status: Optional[dict[str, Any]] = None
    detail: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OperatorControlPlane:
    runner: Any
    ALLOWED_COMMANDS = ("halt", "reduce_only", "resume", "flush", "shutdown")

    def execute(self, control: Any) -> OperatorControlResult:
        req = OperatorControlRequest.from_any(control)
        validation_error = self.validate_request(req)
        if validation_error is not None:
            return OperatorControlResult(
                accepted=False,
                command=req.command,
                outcome="rejected",
                reason=req.reason,
                source=req.source,
                status=self.runner.operator_status() if hasattr(self.runner, "operator_status") else None,
                error=validation_error["error"],
                error_code=validation_error["error_code"],
            )

        try:
            raw = self.runner.apply_control(req)
        except Exception as exc:
            return OperatorControlResult(
                accepted=False,
                command=req.command,
                outcome="rejected",
                reason=req.reason,
                source=req.source,
                status=self.runner.operator_status() if hasattr(self.runner, "operator_status") else None,
                error=str(exc),
                error_code="runtime_error",
            )

        return OperatorControlResult(
            accepted=True,
            command=req.command,
            outcome=self._derive_outcome(req.command, raw),
            reason=req.reason,
            source=req.source,
            status=self.runner.operator_status() if hasattr(self.runner, "operator_status") else None,
            detail=self._detail(raw),
        )

    @classmethod
    def validate_request(cls, req: OperatorControlRequest) -> Optional[dict[str, str]]:
        if not req.command:
            return {"error": "missing command", "error_code": "missing_command"}
        if req.command not in cls.ALLOWED_COMMANDS:
            return {
                "error": f"unsupported control command: {req.command}",
                "error_code": "invalid_command",
            }
        if not req.reason.strip():
            return {"error": "missing reason", "error_code": "missing_reason"}
        return None

    @classmethod
    def available_commands(cls) -> tuple[str, ...]:
        return cls.ALLOWED_COMMANDS

    @staticmethod
    def _derive_outcome(command: str, raw: Any) -> str:
        if command in {"halt", "reduce_only"} and hasattr(raw, "mode"):
            return str(raw.mode.value)
        if command == "resume":
            return "cleared" if bool(raw) else "noop"
        if command == "flush":
            if raw is None:
                return "unavailable"
            if hasattr(raw, "ok"):
                return "ok" if bool(raw.ok) else "drift"
            if isinstance(raw, Mapping):
                return "ok" if bool(raw.get("ok")) else "drift"
            return "completed"
        if command == "shutdown":
            return "stopping"
        return "completed"

    @staticmethod
    def _detail(raw: Any) -> Optional[dict[str, Any]]:
        if raw is None:
            return None
        if hasattr(raw, "__dataclass_fields__"):
            return dict(asdict(raw))
        if isinstance(raw, Mapping):
            return dict(raw)
        if isinstance(raw, bool):
            return {"value": raw}
        if hasattr(raw, "__dict__"):
            return dict(vars(raw))
        return {"value": raw}

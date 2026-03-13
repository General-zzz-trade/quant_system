from __future__ import annotations

from typing import Any

from execution.observability.alerts import build_execution_alert
from monitoring.alerts.base import Alert, Severity


def timeout_to_alert(
    *,
    venue: str,
    symbol: str,
    order_id: str,
    timeout_sec: float,
    source: str = "execution:timeout",
) -> Alert:
    return build_execution_alert(
        title="execution-timeout",
        message=f"{symbol} order {order_id} timed out after {timeout_sec:.1f}s on {venue}",
        severity=Severity.WARNING,
        source=source,
        category="execution_timeout",
        routing_key=f"{venue}:{symbol}:timeout",
        meta={
            "venue": venue,
            "symbol": symbol,
            "order_id": order_id,
            "timeout_sec": timeout_sec,
        },
    )


def reconcile_report_to_alert(report: Any, *, source: str = "execution:reconcile") -> Alert:
    venue = str(getattr(report, "venue", ""))
    drifts = tuple(getattr(report, "all_drifts", ()) or ())
    symbols = sorted({str(getattr(d, "symbol", "")) for d in drifts if getattr(d, "symbol", "")})
    should_halt = bool(getattr(report, "should_halt", False))
    drift_count = len(drifts)
    severity = Severity.ERROR if should_halt else Severity.WARNING
    scope = "critical" if should_halt else "warning"
    symbol_part = ",".join(symbols) if symbols else "*"
    return build_execution_alert(
        title="execution-reconcile-drift",
        message=f"{venue} reconcile detected {drift_count} drift(s)",
        severity=severity,
        source=source,
        category="execution_reconcile",
        routing_key=f"{venue}:{symbol_part}:{scope}",
        meta={
            "venue": venue,
            "drift_count": drift_count,
            "should_halt": should_halt,
            "symbols": symbols,
            "severity_scope": scope,
        },
    )


def synthetic_fill_to_alert(fill: Any, *, source: str = "execution:synthetic_fill") -> Alert:
    venue = str(getattr(fill, "venue", ""))
    symbol = str(getattr(fill, "symbol", ""))
    fill_id = str(getattr(fill, "fill_id", ""))
    order_id = str(getattr(fill, "order_id", ""))
    qty = str(getattr(fill, "qty", ""))
    side = str(getattr(fill, "side", ""))
    return build_execution_alert(
        title="execution-synthetic-fill",
        message=f"{symbol} synthetic fill {fill_id} qty={qty} side={side}",
        severity=Severity.INFO,
        source=source,
        category="execution_fill",
        routing_key=f"{venue}:{symbol}:synthetic_fill",
        meta={
            "venue": venue,
            "symbol": symbol,
            "fill_id": fill_id,
            "order_id": order_id,
            "qty": qty,
            "side": side,
            "synthetic": True,
        },
    )

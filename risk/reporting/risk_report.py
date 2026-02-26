"""Risk report generation — produces structured risk reports.

Generates daily risk summaries combining position data, VaR, stress tests,
and concentration metrics.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PositionRisk:
    """Risk summary for a single position."""
    symbol: str
    qty: float
    market_value: float
    weight: float
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class RiskSummary:
    """Overall portfolio risk summary."""
    report_ts: float
    total_equity: float
    total_exposure: float
    leverage: float
    position_count: int
    positions: tuple[PositionRisk, ...]
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    max_drawdown: Optional[float] = None
    hhi: Optional[float] = None
    stress_worst_drawdown: Optional[float] = None
    alerts: tuple[str, ...] = ()


class RiskReportGenerator:
    """Generates structured risk reports.

    Collects position, VaR, stress test, and concentration data into
    a unified report format. Supports JSON output and file persistence.
    """

    def __init__(self, *, output_dir: Optional[str] = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        equity: float,
        positions: Dict[str, Dict[str, float]],
        var_95: Optional[float] = None,
        var_99: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        hhi: Optional[float] = None,
        stress_worst_drawdown: Optional[float] = None,
        alerts: Optional[List[str]] = None,
    ) -> RiskSummary:
        """Generate a risk summary report.

        Args:
            equity: Total portfolio equity.
            positions: Dict of symbol → {qty, price, pnl}.
            var_95: Portfolio VaR at 95% confidence.
            var_99: Portfolio VaR at 99% confidence.
            max_drawdown: Current max drawdown fraction.
            hhi: Herfindahl-Hirschman concentration index.
            stress_worst_drawdown: Worst drawdown from stress tests.
            alerts: List of active risk alerts.
        """
        pos_risks: List[PositionRisk] = []
        total_exposure = 0.0

        for symbol, data in positions.items():
            qty = data.get("qty", 0.0)
            price = data.get("price", 0.0)
            pnl = data.get("pnl", 0.0)
            mv = abs(qty * price)
            total_exposure += mv
            pos_risks.append(PositionRisk(
                symbol=symbol,
                qty=qty,
                market_value=mv,
                weight=0.0,  # filled below
                unrealized_pnl=pnl,
            ))

        # Compute weights
        if total_exposure > 0:
            pos_risks = [
                PositionRisk(
                    symbol=p.symbol,
                    qty=p.qty,
                    market_value=p.market_value,
                    weight=p.market_value / total_exposure,
                    unrealized_pnl=p.unrealized_pnl,
                )
                for p in pos_risks
            ]

        leverage = total_exposure / equity if equity > 0 else 0.0

        summary = RiskSummary(
            report_ts=time.time(),
            total_equity=equity,
            total_exposure=total_exposure,
            leverage=leverage,
            position_count=len(pos_risks),
            positions=tuple(pos_risks),
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            hhi=hhi,
            stress_worst_drawdown=stress_worst_drawdown,
            alerts=tuple(alerts or []),
        )

        if self._output_dir:
            self._save(summary)

        return summary

    def _save(self, summary: RiskSummary) -> Path:
        """Save report to JSON file."""
        assert self._output_dir is not None
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime(summary.report_ts))
        path = self._output_dir / f"risk_report_{ts_str}.json"
        data = _summary_to_dict(summary)
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Risk report saved: %s", path)
        return path

    def format_text(self, summary: RiskSummary) -> str:
        """Format report as human-readable text."""
        lines = [
            "=" * 60,
            "RISK REPORT",
            "=" * 60,
            f"Equity:       ${summary.total_equity:,.2f}",
            f"Exposure:     ${summary.total_exposure:,.2f}",
            f"Leverage:     {summary.leverage:.2f}x",
            f"Positions:    {summary.position_count}",
        ]

        if summary.var_95 is not None:
            lines.append(f"VaR (95%):    ${summary.var_95:,.2f}")
        if summary.var_99 is not None:
            lines.append(f"VaR (99%):    ${summary.var_99:,.2f}")
        if summary.max_drawdown is not None:
            lines.append(f"Max DD:       {summary.max_drawdown:.2%}")
        if summary.hhi is not None:
            lines.append(f"HHI:          {summary.hhi:.4f}")

        lines.append("-" * 60)
        lines.append(f"{'Symbol':<12} {'Qty':>10} {'Value':>12} {'Weight':>8} {'PnL':>12}")
        lines.append("-" * 60)
        for p in summary.positions:
            lines.append(
                f"{p.symbol:<12} {p.qty:>10.4f} ${p.market_value:>11,.2f} "
                f"{p.weight:>7.1%} ${p.unrealized_pnl:>11,.2f}"
            )

        if summary.alerts:
            lines.append("-" * 60)
            lines.append("ALERTS:")
            for alert in summary.alerts:
                lines.append(f"  ! {alert}")

        lines.append("=" * 60)
        return "\n".join(lines)


def _summary_to_dict(summary: RiskSummary) -> Dict[str, Any]:
    """Convert RiskSummary to a JSON-serializable dict."""
    return {
        "report_ts": summary.report_ts,
        "total_equity": summary.total_equity,
        "total_exposure": summary.total_exposure,
        "leverage": summary.leverage,
        "position_count": summary.position_count,
        "positions": [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "market_value": p.market_value,
                "weight": p.weight,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in summary.positions
        ],
        "var_95": summary.var_95,
        "var_99": summary.var_99,
        "max_drawdown": summary.max_drawdown,
        "hhi": summary.hhi,
        "stress_worst_drawdown": summary.stress_worst_drawdown,
        "alerts": list(summary.alerts),
    }

"""P&L attribution analysis — cost, returns, and reporting."""
from attribution.pnl import PnLBreakdown, compute_pnl
from attribution.cost import CostBreakdown, compute_cost_attribution
from attribution.report import AttributionReport, build_report

__all__ = [
    "PnLBreakdown",
    "compute_pnl",
    "CostBreakdown",
    "compute_cost_attribution",
    "AttributionReport",
    "build_report",
]

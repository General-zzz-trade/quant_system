"""Storage — data loaders for backtest and analysis.

Provides typed record loaders that read CSV/DB artifacts produced by the
ingestion layer and return structured Python objects.
"""
from data.loaders.funding_rate import FundingRecord, load_funding_csv, funding_schedule_for_bars

__all__ = [
    "FundingRecord",
    "load_funding_csv",
    "funding_schedule_for_bars",
]

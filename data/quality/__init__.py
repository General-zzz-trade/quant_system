"""Quality — bar validation, gap detection, and live event filtering.

Modules:
  validators      BarValidator (OHLC consistency, time continuity, anomaly detection)
  gaps            GapDetector + GapFiller (missing-bar detection and interpolation)
  live_validator  LiveBarValidator (lightweight live-event gate)
"""
from data.quality.gaps import Gap, GapReport, GapDetector, GapFiller
from data.quality.validators import ValidationResult, BarValidator
from data.quality.live_validator import LiveBarValidator

__all__ = [
    "Gap",
    "GapReport",
    "GapDetector",
    "GapFiller",
    "ValidationResult",
    "BarValidator",
    "LiveBarValidator",
]

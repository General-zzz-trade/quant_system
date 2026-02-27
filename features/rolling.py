from __future__ import annotations

try:
    from features._quant_rolling import RollingWindow
except ImportError:
    from features._rolling_py import RollingWindow

from features._rolling_py import rolling_apply

__all__ = ["RollingWindow", "rolling_apply"]

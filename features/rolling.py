from __future__ import annotations

try:
    from _quant_hotpath import RollingWindow
except ImportError:
    from features._rolling_py import RollingWindow

from features._rolling_py import rolling_apply

__all__ = ["RollingWindow", "rolling_apply"]

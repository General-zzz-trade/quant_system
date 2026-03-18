#!/usr/bin/env python3
"""Compatibility alias for ``scripts.backtesting.backtest_engine``.

This module intentionally resolves to the implementation module object itself,
so mutable module globals such as ``MODEL_DIR`` stay in sync for legacy callers.
"""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("scripts.backtesting.backtest_engine", run_name="__main__")
else:
    from scripts.backtesting import backtest_engine as _impl

    sys.modules[__name__] = _impl

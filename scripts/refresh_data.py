#!/usr/bin/env python3
"""Compatibility entrypoint for ``scripts.data.refresh_data``."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("scripts.data.refresh_data", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.data.refresh_data")

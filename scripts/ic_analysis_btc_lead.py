#!/usr/bin/env python3
"""Compatibility entrypoint for ``scripts.research.ic_analysis_btc_lead``."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("scripts.research.ic_analysis_btc_lead", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.research.ic_analysis_btc_lead")

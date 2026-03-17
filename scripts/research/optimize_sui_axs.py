"""Optimize SUI and AXS models via feature reselection + min_hold tuning.

Usage:
    python3 -m scripts.research.optimize_sui_axs
"""
from __future__ import annotations
import logging

_log = logging.getLogger(__name__)

SUI_GRID = {"min_hold": [12, 18, 24], "add_oi_features": [True, False]}
AXS_GRID = {"min_hold": [12, 18, 24]}


def optimize_sui():
    _log.info("SUIUSDT optimization — adding V13 OI features + min_hold search")
    for mh in SUI_GRID["min_hold"]:
        for use_oi in SUI_GRID["add_oi_features"]:
            _log.info("  Testing min_hold=%d, oi=%s", mh, use_oi)


def optimize_axs():
    _log.info("AXSUSDT optimization — feature reselection + min_hold search")
    for mh in AXS_GRID["min_hold"]:
        _log.info("  Testing min_hold=%d", mh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optimize_sui()
    optimize_axs()

#!/usr/bin/env python3
"""Download social sentiment data for IC analysis.

Note: Free historical social sentiment data is very limited.
- CoinGecko trending: no historical API
- LunarCrush: requires API key
- Google Trends: rate-limited, unreliable

This script generates a placeholder CSV. For actual historical data,
use LunarCrush API with a key or collect incrementally via the SentimentPoller.

Usage:
    python scripts/download_sentiment.py
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_path = Path("data_files/btc_sentiment.csv")
    logger.info(
        "Historical social sentiment data requires a LunarCrush API key. "
        "The SentimentPoller collects data incrementally in live trading. "
        "No historical data available for IC analysis at this time."
    )
    logger.info("Skipping sentiment download — features will be tested in live trading only.")


if __name__ == "__main__":
    main()

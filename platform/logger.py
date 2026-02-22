from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(level)
    return log

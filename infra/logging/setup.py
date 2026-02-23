from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(*, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("quant_system")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        if log_file:
            p = Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(p, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger

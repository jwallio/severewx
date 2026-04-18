"""Logging helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("severewx")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger

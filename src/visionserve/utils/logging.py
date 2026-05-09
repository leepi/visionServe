"""Logging utility — clean structured output."""
from __future__ import annotations

import logging
import sys

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str = "visionserve", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_FORMATTER)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

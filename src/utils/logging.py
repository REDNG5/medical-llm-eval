"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import ensure_dir


def get_logger(name: str, log_file: str | None = None, level: str = "INFO") -> logging.Logger:
    """Create a configured logger with stream and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


"""Deterministic seed handling."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Set deterministic seed for supported standard libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


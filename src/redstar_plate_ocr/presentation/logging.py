"""Logging configuration for CLI."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    output_dir: Path | None = None,
    verbose: int = 0,
) -> None:
    """Configure logging with file and console handlers."""
    root = logging.getLogger()
    root.handlers.clear()

    console_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        console_levels.get(verbose, logging.DEBUG),
    )
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            output_dir / "train.log",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    root.setLevel(logging.DEBUG)

"""Logging setup for the AutoML package."""

from __future__ import annotations

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure standard-library logging for application entry points."""

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""

    return logging.getLogger(name)

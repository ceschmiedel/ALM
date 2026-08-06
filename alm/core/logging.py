"""Logging setup shared by the CLI, the API server and embedded use."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Install a single stderr handler at the requested level.

    Libraries should not configure logging on import; this is called
    explicitly by the CLI and the API entry points.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = (level or os.getenv("ALM_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, resolved, logging.INFO))

    # These are chatty and rarely useful at INFO during federation runs.
    for noisy in ("httpx", "httpcore", "urllib3", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

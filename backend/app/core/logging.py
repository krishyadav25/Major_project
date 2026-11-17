"""Structured logging configuration for Adapted Reasoning RAG."""

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """Configure the root logger with a consistent structured format.

    Uses the log level configured in application settings and writes to stdout.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"

    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if configure_logging is called multiple times
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress verbose loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance configured with the specified name."""
    return logging.getLogger(name)

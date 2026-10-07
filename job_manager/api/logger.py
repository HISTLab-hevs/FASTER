"""Application logger with optional rotating file handler.

Provides a singleton logger named ``job_manager`` that writes to
the console and, optionally, to a rotating log file configured
via ``config.yaml → log_file``.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from api.config import Config

_config = Config()


def get_logger() -> logging.Logger:
    """Return the configured application logger.

    On the first call the logger is set up with:

    - A :class:`~logging.StreamHandler` for console output.
    - An optional :class:`~logging.handlers.RotatingFileHandler`
      (max 5 MB, 3 backups) if ``log_file`` is set in the config.

    Subsequent calls return the same logger instance without adding
    duplicate handlers.

    Returns:
        The ``job_manager`` :class:`logging.Logger`.
    """
    logger = logging.getLogger("job_manager")

    if logger.handlers:
        return logger

    log_level_str = _config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_file = _config.get("log_file")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os


_configured = False


def _ensure_base_logger() -> None:
    global _configured
    if _configured:
        return
    level = os.getenv("ORCH_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    _configured = True


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    _ensure_base_logger()
    logger = logging.getLogger(name)
    # Do not duplicate handlers if already set
    if log_file and not any(
        isinstance(h, RotatingFileHandler) for h in logger.handlers
    ):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

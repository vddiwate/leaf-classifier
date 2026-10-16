"""Logging and observability framework with rotating file handlers and console output."""

import functools
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional


class ColorConsoleFormatter(logging.Formatter):
    """Custom formatter adding color to console logs for rapid debugging."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[1;91m"
    RESET = "\033[0m"

    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        orig_levelname = record.levelname
        record.levelname = f"{color}{orig_levelname:<8}{self.RESET}"
        formatted = super().format(record)
        record.levelname = orig_levelname  # Restore clean levelname for file handler!
        return formatted


_LOGGER_INITIALIZED = False


def setup_logger(
    name: str = "leafy_cv",
    log_dir: str = "logs",
    log_file_name: str = "leafy_cv.log",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per log file
    backup_count: int = 5,
) -> logging.Logger:
    """Configures and returns the central logger instance with a SINGLE unified log file.

    All events (DEBUG, INFO, WARNING, ERROR, CRITICAL) are recorded sequentially in
    one chronological log file so that any failure can be traced back to the exact
    steps that preceded it.
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)  # Root logger captures DEBUG and above

    # Avoid duplicate handlers if setup_logger is invoked multiple times
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 1. Console Formatter (Colored, concise)
    console_fmt = ColorConsoleFormatter(
        fmt="%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 2. Unified Rotating File Handler (Single log file for everything: INFO + ERROR)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    unified_log_file = log_path / log_file_name
    file_handler = RotatingFileHandler(
        filename=str(unified_log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    # The file captures DEBUG and above so full context is preserved when an error occurs
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    _LOGGER_INITIALIZED = True
    return logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """Retrieves a namespaced logger for any module in the pipeline.

    Example:
        logger = get_logger("preprocessing")
        # Logs as 'leafy_cv.preprocessing'
    """
    base_logger = logging.getLogger("leafy_cv")
    if not base_logger.handlers:
        setup_logger()

    if module_name:
        return logging.getLogger(f"leafy_cv.{module_name}")
    return base_logger


def timed_stage(stage_name: str) -> Callable:
    """Decorator to measure and log the latency of any pipeline function/stage."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger("profiler")
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug(f"Stage '{stage_name}' completed in {elapsed_ms:.2f} ms")
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.error(
                    f"Stage '{stage_name}' failed after {elapsed_ms:.2f} ms: {e}",
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator

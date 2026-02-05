"""
Logging configuration for Market Monitor.

Provides centralized logging setup with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Default format strings
DEFAULT_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DETAILED_FORMAT = '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s'
SIMPLE_FORMAT = '[%(levelname)s] %(message)s'


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    detailed: bool = False,
    quiet: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path for file logging
        format_string: Custom format string (overrides detailed/quiet)
        detailed: Use detailed format with file/line info
        quiet: Minimal output (only warnings and errors)
    """
    # Determine format
    if format_string:
        fmt = format_string
    elif detailed:
        fmt = DETAILED_FORMAT
    elif quiet:
        fmt = SIMPLE_FORMAT
    else:
        fmt = DEFAULT_FORMAT

    # Adjust level for quiet mode
    if quiet:
        level = logging.WARNING

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt))
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all, filter by handlers
        format=fmt,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set level for our package
    logging.getLogger('market_monitor').setLevel(level)

    # Suppress noisy third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('openpyxl').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting analysis...")
    """
    # If name starts with market_monitor, use as-is
    # Otherwise, prefix with market_monitor
    if name.startswith('market_monitor'):
        return logging.getLogger(name)
    return logging.getLogger(f'market_monitor.{name}')


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance
        exc: Exception to log
        context: Optional context description
    """
    if context:
        logger.error(f"{context}: {exc}", exc_info=True)
    else:
        logger.error(str(exc), exc_info=True)


class LogContext:
    """
    Context manager for logging operation start/end with timing.

    Example:
        with LogContext(logger, "Loading data"):
            load_data()
        # Logs: "Loading data..." and "Loading data completed in 2.5s"
    """

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> 'LogContext':
        self.start_time = datetime.now()
        self.logger.log(self.level, f"{self.operation}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.log(self.level, f"{self.operation} completed in {elapsed:.2f}s")
        else:
            self.logger.error(f"{self.operation} failed after {elapsed:.2f}s: {exc_val}")

        return False  # Don't suppress exceptions


class ProgressLogger:
    """
    Logger for progress updates that overwrites the same line.

    Example:
        progress = ProgressLogger(logger, total=100)
        for i in range(100):
            do_work(i)
            progress.update(i + 1)
        progress.finish()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        description: str = "Processing",
        update_interval: int = 10
    ):
        self.logger = logger
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.current = 0

    def update(self, current: int) -> None:
        """Update progress."""
        self.current = current
        if current % self.update_interval == 0 or current == self.total:
            pct = (current / self.total) * 100
            print(f"\r  {self.description}... {current}/{self.total} ({pct:.0f}%)", end='')
            sys.stdout.flush()

    def finish(self, message: Optional[str] = None) -> None:
        """Complete progress logging."""
        print()  # New line
        if message:
            self.logger.info(message)
        else:
            self.logger.info(f"{self.description} completed ({self.total} items)")

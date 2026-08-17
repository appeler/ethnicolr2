"""Structured logging configuration for ethnicolr2."""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def __init__(self, include_fields: list[str] | None = None):
        super().__init__()
        self.include_fields = include_fields or [
            "timestamp",
            "level",
            "logger",
            "message",
            "module",
            "function",
        ]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry: dict[str, Any] = {}

        # Basic fields
        if "timestamp" in self.include_fields:
            log_entry["timestamp"] = self.formatTime(record)
        if "level" in self.include_fields:
            log_entry["level"] = record.levelname
        if "logger" in self.include_fields:
            log_entry["logger"] = record.name
        if "message" in self.include_fields:
            log_entry["message"] = record.getMessage()
        if "module" in self.include_fields:
            log_entry["module"] = record.module
        if "function" in self.include_fields:
            log_entry["function"] = record.funcName
        if "line" in self.include_fields:
            log_entry["line"] = record.lineno

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields from the log call
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class EthnicolrLogger:
    """Centralized logger for ethnicolr2 with structured output and performance tracking."""

    _loggers: dict[str, logging.Logger] = {}
    _configured: bool = False

    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        format_type: str = "structured",  # "structured" or "simple"
        log_file: str | None = None,
        include_performance: bool = True,
    ) -> None:
        """Setup logging configuration for the entire package.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_type: "structured" for JSON output, "simple" for human-readable
            log_file: Optional file path for log output
            include_performance: Whether to include performance metrics in logs
        """
        if cls._configured:
            return

        # Get level from environment if not specified
        log_level = os.getenv("ETHNICOLR_LOG_LEVEL", level).upper()

        # Configure root logger
        root_logger = logging.getLogger("ethnicolr2")
        root_logger.setLevel(getattr(logging, log_level))

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        match format_type:
            case "structured":
                console_handler.setFormatter(StructuredFormatter())
            case _:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )

        root_logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)

        cls._configured = True

        # Log initial setup
        logger = cls.get_logger("setup")
        logger.info(
            "Logging configured",
            extra={
                "level": log_level,
                "format_type": format_type,
                "log_file": log_file,
                "include_performance": include_performance,
            },
        )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance for the given module.

        Args:
            name: Logger name (typically module name)

        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.setup()

        full_name = f"ethnicolr2.{name}"

        if full_name not in cls._loggers:
            logger = logging.getLogger(full_name)
            cls._loggers[full_name] = logger

        return cls._loggers[full_name]


class PerformanceTracker:
    """Context manager for tracking performance metrics."""

    def __init__(
        self, operation: str, logger: logging.Logger | None = None, **kwargs: Any
    ):
        self.operation = operation
        self.logger = logger or EthnicolrLogger.get_logger("performance")
        self.start_time: float | None = None
        self.metadata: dict[str, Any] = kwargs

    def __enter__(self) -> "PerformanceTracker":
        self.start_time = time.perf_counter()
        self.logger.debug(
            f"Started {self.operation}",
            extra={
                "operation": self.operation,
                "event": "start",
                **self.metadata,
            },
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.start_time is None:
            return

        duration = time.perf_counter() - self.start_time

        if exc_type:
            self.logger.error(
                f"Failed {self.operation}",
                extra={
                    "operation": self.operation,
                    "event": "error",
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__ if exc_type else None,
                    **self.metadata,
                },
            )
        else:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    "operation": self.operation,
                    "event": "complete",
                    "duration_seconds": duration,
                    **self.metadata,
                },
            )


def performance_logged(
    operation: str, logger: logging.Logger | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to automatically log function performance.

    Args:
        operation: Description of the operation being performed
        logger: Optional logger instance
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with PerformanceTracker(operation, logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Auto-configure logging on import if environment variable is set
if os.getenv("ETHNICOLR_AUTO_SETUP_LOGGING", "false").lower() == "true":
    EthnicolrLogger.setup()

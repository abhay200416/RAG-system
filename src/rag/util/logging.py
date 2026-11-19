"""
Comprehensive logging configuration for the RAG project.

Provides structured logging with support for:
- Console and file output
- JSON and human-readable formats
- Environment-based configuration
- Contextual logging with correlation IDs
"""

import logging
import logging.config
import json
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from contextvars import ContextVar

from .config import get_settings

# Context variable for correlation ID (useful for request tracing)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationFilter(logging.Filter):
    """Adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "-"
        return True


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, "correlation_id", "-"),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_data["stack_info"] = self.formatStack(record.stack_info)

        if self.include_extra:
            extra_fields = {
                k: v
                for k, v in record.__dict__.items()
                if k
                not in {
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "msg",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "correlation_id",
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Human-readable colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        correlation_id = getattr(record, "correlation_id", "-")
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        log_line = (
            f"{color}{timestamp}{reset} | "
            f"{color}{record.levelname:<8}{reset} | "
            f"\033[36m{record.name}{reset} | "
            f"cid:{correlation_id} | "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def get_log_level() -> int:
    """Get log level from settings."""
    settings = get_settings()
    level_str = settings.log_level.upper()
    return getattr(logging, level_str, logging.INFO)


def get_log_format() -> str:
    """Get log format from settings."""
    settings = get_settings()
    return settings.log_format.lower()


def get_log_file() -> Optional[str]:
    """Get log file path from settings."""
    settings = get_settings()
    return settings.log_file


def setup_logging(
    level: Optional[int] = None,
    format_type: Optional[str] = None,
    log_file: Optional[str] = None,
    json_output: bool = False,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (default: from RAG_LOG_LEVEL env or INFO)
        format_type: 'console' for colored, 'json' for JSON (default: from RAG_LOG_FORMAT env or 'console')
        log_file: Path to log file (default: from RAG_LOG_FILE env or None)
        json_output: Force JSON output to console (overrides format_type)
    """
    level = level or get_log_level()
    format_type = format_type or get_log_format()
    log_file = log_file or get_log_file()

    if json_output:
        format_type = "json"

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Add correlation filter to all handlers
    correlation_filter = CorrelationFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.addFilter(correlation_filter)

    if format_type == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())

    root_logger.addHandler(console_handler)

    # File handler (if configured)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.addFilter(correlation_filter)
        file_handler.setFormatter(JSONFormatter(include_extra=True))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id_var.get()


class LoggingContext:
    """Context manager for temporary correlation ID."""

    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.previous_id: Optional[str] = None

    def __enter__(self):
        self.previous_id = get_correlation_id()
        set_correlation_id(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_correlation_id(self.previous_id)


# Initialize logging on module import
setup_logging()

# Export commonly used items
__all__ = [
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "LoggingContext",
    "CorrelationFilter",
    "JSONFormatter",
    "ColoredFormatter",
]
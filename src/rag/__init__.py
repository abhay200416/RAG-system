"""
RAG System - Main entry point.

This module demonstrates proper logging usage throughout the application.
"""

import uuid
from rag.util.logging import get_logger, LoggingContext, set_correlation_id, get_correlation_id
from rag.util.config import get_settings

# Get a logger for this module
logger = get_logger(__name__)


def main() -> None:
    """Main entry point demonstrating logging features."""
    settings = get_settings()

    logger.info("Starting RAG System", extra={"version": settings.app_version})

    # Demonstrate correlation ID for request tracing
    correlation_id = str(uuid.uuid4())[:8]
    with LoggingContext(correlation_id):
        logger.info("Processing request with correlation ID")

        # Simulate some work
        _demonstrate_log_levels()
        _demonstrate_structured_logging()
        _demonstrate_exception_logging()

    logger.info("RAG System shutdown complete")


def _demonstrate_log_levels() -> None:
    """Show different log levels."""
    logger.debug("Debug message - detailed diagnostic information")
    logger.info("Info message - general operational information")
    logger.warning("Warning message - something unexpected happened")
    logger.error("Error message - serious problem occurred")
    logger.critical("Critical message - program may be unable to continue")


def _demonstrate_structured_logging() -> None:
    """Show structured logging with extra fields."""
    logger.info(
        "User action recorded",
        extra={
            "user_id": "user_123",
            "action": "document_upload",
            "document_count": 5,
            "total_size_mb": 12.5,
        }
    )

    # Log with context
    logger.info("Vector store query executed", extra={
        "query": "machine learning",
        "top_k": 5,
        "results_found": 3,
        "latency_ms": 42,
    })


def _demonstrate_exception_logging() -> None:
    """Show exception logging with traceback."""
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.exception(
            "Mathematical error occurred",
            extra={"operation": "division", "operand_a": 10, "operand_b": 0}
        )


def demonstrate_contextual_logging() -> None:
    """Demonstrate setting correlation ID manually."""
    # Set correlation ID for a request handler
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    set_correlation_id(request_id)

    logger.info("Handling HTTP request", extra={
        "method": "POST",
        "path": "/api/query",
        "client_ip": "192.168.1.100",
    })

    # Clear correlation ID when done
    set_correlation_id(None)


if __name__ == "__main__":
    main()
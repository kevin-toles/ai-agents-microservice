"""Structured logging configuration with JSON format.

Implements structured logging per AGENT_FUNCTIONS_ARCHITECTURE.md and
CODING_PATTERNS_ANALYSIS.md requirements.

Features:
- JSON-formatted log output for production
- Human-readable format for development
- Request correlation IDs
- Service context (Kitchen Brigade role)

Reference: WBS-AGT2 AC-2.2
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from src.core.config import get_settings


def add_service_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service context to all log entries.
    
    Kitchen Brigade Architecture:
        ai-agents is the Expeditor - coordinates pipeline execution.
    
    Args:
        logger: The wrapped logger object.
        method_name: The name of the method called on the logger.
        event_dict: The event dictionary to process.
    
    Returns:
        Updated event dictionary with service context.
    """
    settings = get_settings()
    event_dict["service"] = settings.service_name
    event_dict["environment"] = settings.environment
    event_dict["role"] = "Expeditor"  # Kitchen Brigade role
    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application.
    
    In development: Human-readable colored output
    In production: JSON-formatted structured logs
    
    Pattern: Structured Logging with Context
    Reference: CODING_PATTERNS_ANALYSIS.md
    """
    settings = get_settings()
    
    # Determine if we should use JSON format
    use_json = settings.environment in ("production", "staging")
    
    # Common processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_service_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if use_json:
        # Production: JSON output
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: Human-readable colored output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level.upper()),
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Optional logger name (module name recommended).
    
    Returns:
        Configured structlog BoundLogger instance.
    
    Example:
        ```python
        from src.core.logging import get_logger
        
        logger = get_logger(__name__)
        logger.info("Processing request", request_id="abc123", function="extract_structure")
        ```
    """
    return structlog.get_logger(name)


# Convenience type alias
Logger = structlog.BoundLogger

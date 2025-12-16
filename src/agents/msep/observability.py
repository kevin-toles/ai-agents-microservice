"""MSEP Observability.

WBS: MSE-4.5 - Observability
Structured logging and timing metrics for MSEP.

Reference Documents:
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-4.5

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any


# Configure logging
logger = logging.getLogger(__name__)


class MSEPObserver:
    """Observer for MSEP metrics and logging.

    Provides structured logging with correlation IDs,
    timing metrics, and error rate tracking.
    """

    def __init__(self, correlation_id: str | None = None) -> None:
        """Initialize observer.

        Args:
            correlation_id: Optional correlation ID (generated if not provided)
        """
        self._correlation_id = correlation_id or str(uuid.uuid4())
        self._timing_metrics: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}

    @property
    def correlation_id(self) -> str:
        """Get correlation ID."""
        return self._correlation_id

    def log_enrichment_start(self, corpus_size: int) -> dict[str, Any]:
        """Log start of enrichment.

        Args:
            corpus_size: Number of documents in corpus

        Returns:
            Structured log entry
        """
        log_entry: dict[str, Any] = {
            "event": "enrichment_start",
            "correlation_id": self._correlation_id,
            "corpus_size": corpus_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"MSEP enrichment start: {log_entry}")
        return log_entry

    def log_enrichment_complete(
        self,
        processing_time_ms: float,
        total_chapters: int,
    ) -> dict[str, Any]:
        """Log completion of enrichment.

        Args:
            processing_time_ms: Total processing time in milliseconds
            total_chapters: Number of chapters processed

        Returns:
            Structured log entry
        """
        log_entry: dict[str, Any] = {
            "event": "enrichment_complete",
            "correlation_id": self._correlation_id,
            "processing_time_ms": processing_time_ms,
            "total_chapters": total_chapters,
            "timing_metrics": self._timing_metrics.copy(),
            "error_counts": self._error_counts.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"MSEP enrichment complete: {log_entry}")
        return log_entry

    def log_service_call(
        self,
        service: str,
        endpoint: str,
        duration_ms: float,
        success: bool,
    ) -> dict[str, Any]:
        """Log a service call.

        Args:
            service: Service name
            endpoint: API endpoint called
            duration_ms: Call duration in milliseconds
            success: Whether call succeeded

        Returns:
            Structured log entry
        """
        log_entry: dict[str, Any] = {
            "event": "service_call",
            "correlation_id": self._correlation_id,
            "service": service,
            "endpoint": endpoint,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if success:
            self.record_success(service)
        logger.debug(f"MSEP service call: {log_entry}")
        return log_entry

    def log_service_error(
        self,
        service: str,
        endpoint: str,
        error: str,
    ) -> dict[str, Any]:
        """Log a service error.

        Args:
            service: Service name
            endpoint: API endpoint called
            error: Error message

        Returns:
            Structured log entry
        """
        self.record_error(service, error)

        log_entry: dict[str, Any] = {
            "event": "service_error",
            "correlation_id": self._correlation_id,
            "service": service,
            "endpoint": endpoint,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(f"MSEP service error: {log_entry}")
        return log_entry

    def record_timing(self, operation: str, duration_ms: float) -> None:
        """Record timing metric.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
        """
        self._timing_metrics[operation] = duration_ms

    def get_timing_metrics(self) -> dict[str, float]:
        """Get all recorded timing metrics.

        Returns:
            Dict of operation -> duration_ms
        """
        return self._timing_metrics.copy()

    def record_error(self, service: str, error: str) -> None:
        """Record error for service.

        Args:
            service: Service name
            error: Error message
        """
        self._error_counts[service] = self._error_counts.get(service, 0) + 1

    def record_success(self, service: str) -> None:
        """Record success for service.

        Args:
            service: Service name
        """
        self._success_counts[service] = self._success_counts.get(service, 0) + 1

    def get_error_counts(self) -> dict[str, int]:
        """Get error counts per service.

        Returns:
            Dict of service -> error count
        """
        return self._error_counts.copy()

    def get_success_counts(self) -> dict[str, int]:
        """Get success counts per service.

        Returns:
            Dict of service -> success count
        """
        return self._success_counts.copy()

    def get_error_rate(self, service: str) -> float:
        """Get error rate for service.

        Args:
            service: Service name

        Returns:
            Error rate (0.0 to 1.0)
        """
        errors = self._error_counts.get(service, 0)
        successes = self._success_counts.get(service, 0)
        total = errors + successes

        if total == 0:
            return 0.0

        return errors / total

"""MSEP Exception Hierarchy.

WBS: MSE-2.4 - Exception Hierarchy
Defines MSEPError base and derived exceptions.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #7: Exception shadowing - all exceptions use unique names (no TimeoutError, ConnectionError)
- #2.2: Full type annotations
- #13: Exception chaining via __cause__
"""

from __future__ import annotations


class MSEPError(Exception):
    """Base exception for all MSEP-related errors.

    All MSEP exceptions inherit from this class to enable
    catching any MSEP error with a single except clause.

    Attributes:
        message: Human-readable error description.
        cause: Original exception that caused this error (optional).
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize MSEP error.

        Args:
            message: Human-readable error description.
            cause: Original exception that caused this error.
        """
        self.message = message
        self.cause = cause
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class EnrichmentTimeoutError(MSEPError):
    """Raised when an enrichment operation times out.

    NOT named TimeoutError to avoid shadowing builtins.TimeoutError
    per CODING_PATTERNS_ANALYSIS.md anti-pattern #7.

    Attributes:
        message: Human-readable error description.
        cause: Original exception that caused this error (optional).
        service: Name of the service that timed out (optional).
        timeout_seconds: Configured timeout value in seconds (optional).
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        service: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Initialize enrichment timeout error.

        Args:
            message: Human-readable error description.
            cause: Original exception that caused this error.
            service: Name of the service that timed out.
            timeout_seconds: Configured timeout value in seconds.
        """
        super().__init__(message, cause)
        self.service = service
        self.timeout_seconds = timeout_seconds


class ServiceUnavailableError(MSEPError):
    """Raised when a required service is unavailable.

    NOT named ConnectionError to avoid shadowing builtins.ConnectionError
    per CODING_PATTERNS_ANALYSIS.md anti-pattern #7.

    Attributes:
        message: Human-readable error description.
        cause: Original exception that caused this error (optional).
        service: Name of the unavailable service (optional).
        url: URL that was unreachable (optional).
    """

    def __init__(
        self,
        message: str,
        cause: Exception | None = None,
        service: str | None = None,
        url: str | None = None,
    ) -> None:
        """Initialize service unavailable error.

        Args:
            message: Human-readable error description.
            cause: Original exception that caused this error.
            service: Name of the unavailable service.
            url: URL that was unreachable.
        """
        super().__init__(message, cause)
        self.service = service
        self.url = url

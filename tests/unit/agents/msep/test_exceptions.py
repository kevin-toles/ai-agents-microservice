"""Unit tests for MSEP exceptions.

WBS: MSE-2.4 - Exception Hierarchy
TDD Phase: RED (tests written BEFORE implementation)

Acceptance Criteria Coverage:
- AC-2.4.1: MSEPError is base exception
- AC-2.4.2: EnrichmentTimeoutError, ServiceUnavailableError inherit from MSEPError
- AC-2.4.3: No exception shadows Python builtins (per #7)
- AC-2.4.4: Each exception has message: str and optional cause: Exception

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #7: Exception shadowing - all exceptions prefixed with MSEP or unique names
- #2.2: Full type annotations
"""

from __future__ import annotations

import builtins

import pytest


# =============================================================================
# AC-2.4.1: MSEPError Base Exception Tests
# =============================================================================


class TestMSEPError:
    """Tests for MSEPError base exception."""

    def test_msep_error_is_exception(self) -> None:
        """AC-2.4.1: MSEPError should inherit from Exception."""
        from src.agents.msep.exceptions import MSEPError

        assert issubclass(MSEPError, Exception)

    def test_msep_error_can_be_raised(self) -> None:
        """AC-2.4.1: MSEPError should be raisable."""
        from src.agents.msep.exceptions import MSEPError

        with pytest.raises(MSEPError):
            raise MSEPError("Test error")

    def test_msep_error_caught_as_exception(self) -> None:
        """AC-2.4.1: MSEPError should be catchable as Exception."""
        from src.agents.msep.exceptions import MSEPError

        with pytest.raises(Exception):
            raise MSEPError("Test error")

    def test_msep_error_has_message(self) -> None:
        """AC-2.4.4: MSEPError should have message attribute."""
        from src.agents.msep.exceptions import MSEPError

        error = MSEPError("Test error message")

        assert error.message == "Test error message"

    def test_msep_error_message_in_str(self) -> None:
        """AC-2.4.4: MSEPError str should contain message."""
        from src.agents.msep.exceptions import MSEPError

        error = MSEPError("Test error message")

        assert "Test error message" in str(error)

    def test_msep_error_optional_cause(self) -> None:
        """AC-2.4.4: MSEPError should accept optional cause."""
        from src.agents.msep.exceptions import MSEPError

        original = ValueError("Original error")
        error = MSEPError("Wrapper error", cause=original)

        assert error.cause is original

    def test_msep_error_cause_defaults_to_none(self) -> None:
        """AC-2.4.4: MSEPError cause should default to None."""
        from src.agents.msep.exceptions import MSEPError

        error = MSEPError("Test error")

        assert error.cause is None


# =============================================================================
# AC-2.4.2: Derived Exception Tests
# =============================================================================


class TestEnrichmentTimeoutError:
    """Tests for EnrichmentTimeoutError exception."""

    def test_enrichment_timeout_inherits_msep_error(self) -> None:
        """AC-2.4.2: EnrichmentTimeoutError should inherit from MSEPError."""
        from src.agents.msep.exceptions import MSEPError, EnrichmentTimeoutError

        assert issubclass(EnrichmentTimeoutError, MSEPError)

    def test_enrichment_timeout_can_be_raised(self) -> None:
        """AC-2.4.2: EnrichmentTimeoutError should be raisable."""
        from src.agents.msep.exceptions import EnrichmentTimeoutError

        with pytest.raises(EnrichmentTimeoutError):
            raise EnrichmentTimeoutError("Timeout occurred")

    def test_enrichment_timeout_caught_as_msep_error(self) -> None:
        """AC-2.4.2: EnrichmentTimeoutError should be catchable as MSEPError."""
        from src.agents.msep.exceptions import MSEPError, EnrichmentTimeoutError

        with pytest.raises(MSEPError):
            raise EnrichmentTimeoutError("Timeout occurred")

    def test_enrichment_timeout_has_message(self) -> None:
        """AC-2.4.4: EnrichmentTimeoutError should have message attribute."""
        from src.agents.msep.exceptions import EnrichmentTimeoutError

        error = EnrichmentTimeoutError("Timeout after 30s")

        assert error.message == "Timeout after 30s"

    def test_enrichment_timeout_has_service_field(self) -> None:
        """EnrichmentTimeoutError should have optional service field."""
        from src.agents.msep.exceptions import EnrichmentTimeoutError

        error = EnrichmentTimeoutError(
            "Timeout waiting for response",
            service="code-orchestrator",
        )

        assert error.service == "code-orchestrator"

    def test_enrichment_timeout_has_timeout_seconds_field(self) -> None:
        """EnrichmentTimeoutError should have optional timeout_seconds field."""
        from src.agents.msep.exceptions import EnrichmentTimeoutError

        error = EnrichmentTimeoutError(
            "Timeout waiting for response",
            timeout_seconds=30.0,
        )

        assert error.timeout_seconds == 30.0


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError exception."""

    def test_service_unavailable_inherits_msep_error(self) -> None:
        """AC-2.4.2: ServiceUnavailableError should inherit from MSEPError."""
        from src.agents.msep.exceptions import MSEPError, ServiceUnavailableError

        assert issubclass(ServiceUnavailableError, MSEPError)

    def test_service_unavailable_can_be_raised(self) -> None:
        """AC-2.4.2: ServiceUnavailableError should be raisable."""
        from src.agents.msep.exceptions import ServiceUnavailableError

        with pytest.raises(ServiceUnavailableError):
            raise ServiceUnavailableError("Service down")

    def test_service_unavailable_caught_as_msep_error(self) -> None:
        """AC-2.4.2: ServiceUnavailableError should be catchable as MSEPError."""
        from src.agents.msep.exceptions import MSEPError, ServiceUnavailableError

        with pytest.raises(MSEPError):
            raise ServiceUnavailableError("Service down")

    def test_service_unavailable_has_message(self) -> None:
        """AC-2.4.4: ServiceUnavailableError should have message attribute."""
        from src.agents.msep.exceptions import ServiceUnavailableError

        error = ServiceUnavailableError("Service is unavailable")

        assert error.message == "Service is unavailable"

    def test_service_unavailable_has_service_field(self) -> None:
        """ServiceUnavailableError should have service field."""
        from src.agents.msep.exceptions import ServiceUnavailableError

        error = ServiceUnavailableError(
            "Cannot connect to service",
            service="semantic-search",
        )

        assert error.service == "semantic-search"

    def test_service_unavailable_has_url_field(self) -> None:
        """ServiceUnavailableError should have optional url field."""
        from src.agents.msep.exceptions import ServiceUnavailableError

        error = ServiceUnavailableError(
            "Cannot connect to service",
            service="code-orchestrator",
            url="http://localhost:8083",
        )

        assert error.url == "http://localhost:8083"


# =============================================================================
# AC-2.4.3: No Builtin Shadowing Tests
# =============================================================================


class TestNoBuiltinShadowing:
    """Tests to ensure no exception shadows Python builtins (per #7)."""

    def test_no_timeout_error_shadowing(self) -> None:
        """AC-2.4.3: No exception named TimeoutError should exist."""
        from src.agents.msep import exceptions

        # Check no direct TimeoutError class exists (would shadow builtins.TimeoutError)
        assert not hasattr(exceptions, "TimeoutError")

    def test_no_connection_error_shadowing(self) -> None:
        """AC-2.4.3: No exception named ConnectionError should exist."""
        from src.agents.msep import exceptions

        # Check no direct ConnectionError class exists (would shadow builtins.ConnectionError)
        assert not hasattr(exceptions, "ConnectionError")

    def test_no_value_error_shadowing(self) -> None:
        """AC-2.4.3: No exception named ValueError should exist."""
        from src.agents.msep import exceptions

        # Check no direct ValueError class exists (would shadow builtins.ValueError)
        assert not hasattr(exceptions, "ValueError")

    def test_enrichment_timeout_is_not_timeout_error(self) -> None:
        """AC-2.4.3: EnrichmentTimeoutError should NOT be builtins.TimeoutError."""
        from src.agents.msep.exceptions import EnrichmentTimeoutError

        # Our custom exception is distinct from the builtin
        assert EnrichmentTimeoutError is not builtins.TimeoutError
        assert not issubclass(EnrichmentTimeoutError, builtins.TimeoutError)


# =============================================================================
# Exception Chaining Tests
# =============================================================================


class TestExceptionChaining:
    """Tests for proper exception chaining."""

    def test_msep_error_chains_cause(self) -> None:
        """MSEPError should chain cause via __cause__."""
        from src.agents.msep.exceptions import MSEPError

        original = ValueError("Original error")
        error = MSEPError("Wrapper", cause=original)

        assert error.__cause__ is original

    def test_enrichment_timeout_chains_cause(self) -> None:
        """EnrichmentTimeoutError should chain cause via __cause__."""
        from src.agents.msep.exceptions import EnrichmentTimeoutError
        import asyncio

        original = asyncio.TimeoutError()
        error = EnrichmentTimeoutError("Timeout", cause=original)

        assert error.__cause__ is original

    def test_service_unavailable_chains_cause(self) -> None:
        """ServiceUnavailableError should chain cause via __cause__."""
        from src.agents.msep.exceptions import ServiceUnavailableError

        original = ConnectionRefusedError()
        error = ServiceUnavailableError("Cannot connect", cause=original)

        assert error.__cause__ is original

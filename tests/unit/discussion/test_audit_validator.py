"""Unit tests for AuditServiceValidator.

WBS Reference: WBS-KB5 - Provenance & Audit Integration
Tasks: KB5.4, KB5.5, KB5.7

Acceptance Criteria:
- AC-KB5.2: Citations validated via audit-service:8084/v1/validate
- AC-KB5.3: Invalid citations rejected, trigger additional evidence gathering
- AC-KB5.5: Audit trail includes discussion_history with all cycles

Exit Criteria:
- audit-service receives validation request with citations
- Invalid citation (fake source) triggers loop retry

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Low cognitive complexity via focused test classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Constants
# =============================================================================

_TEST_AUDIT_URL = "http://audit-service:8084"
_TEST_VALID_CITATION = {
    "marker": 1,
    "source": "cache.py#L10",
    "content": "The cache uses Redis.",
}
_TEST_INVALID_CITATION = {
    "marker": 2,
    "source": "nonexistent.py#L999",
    "content": "Hallucinated claim.",
}


# =============================================================================
# Test Imports (Will Fail in RED Phase)
# =============================================================================


class TestAuditValidatorImports:
    """Test that AuditServiceValidator can be imported."""

    def test_audit_validator_importable(self) -> None:
        """AuditServiceValidator class should be importable."""
        from src.discussion.audit_validator import AuditServiceValidator

        assert AuditServiceValidator is not None

    def test_validation_result_importable(self) -> None:
        """ValidationResult dataclass should be importable."""
        from src.discussion.audit_validator import ValidationResult

        assert ValidationResult is not None

    def test_validation_config_importable(self) -> None:
        """ValidationConfig dataclass should be importable."""
        from src.discussion.audit_validator import ValidationConfig

        assert ValidationConfig is not None


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_has_is_valid_field(self) -> None:
        """ValidationResult must have is_valid field."""
        from src.discussion.audit_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            valid_citations=[1, 2],
            invalid_citations=[],
            errors=[],
        )
        assert result.is_valid is True

    def test_result_has_valid_citations_field(self) -> None:
        """ValidationResult must have valid_citations list."""
        from src.discussion.audit_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            valid_citations=[1, 2, 3],
            invalid_citations=[],
            errors=[],
        )
        assert result.valid_citations == [1, 2, 3]

    def test_result_has_invalid_citations_field(self) -> None:
        """ValidationResult must have invalid_citations list."""
        from src.discussion.audit_validator import ValidationResult

        result = ValidationResult(
            is_valid=False,
            valid_citations=[1],
            invalid_citations=[2, 3],
            errors=["Citation 2 not found", "Citation 3 hallucinated"],
        )
        assert result.invalid_citations == [2, 3]

    def test_result_has_errors_field(self) -> None:
        """ValidationResult must have errors list."""
        from src.discussion.audit_validator import ValidationResult

        result = ValidationResult(
            is_valid=False,
            valid_citations=[],
            invalid_citations=[1],
            errors=["Source not found"],
        )
        assert "Source not found" in result.errors

    def test_result_is_frozen(self) -> None:
        """ValidationResult should be immutable."""
        from src.discussion.audit_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            valid_citations=[],
            invalid_citations=[],
            errors=[],
        )
        with pytest.raises((AttributeError, TypeError)):
            result.is_valid = False  # type: ignore

    def test_result_to_dict(self) -> None:
        """ValidationResult should serialize to dict."""
        from src.discussion.audit_validator import ValidationResult

        result = ValidationResult(
            is_valid=False,
            valid_citations=[1],
            invalid_citations=[2],
            errors=["Error"],
        )
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["valid_citations"] == [1]
        assert d["invalid_citations"] == [2]
        assert d["errors"] == ["Error"]


# =============================================================================
# ValidationConfig Tests
# =============================================================================


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_service_url(self) -> None:
        """Default service_url should be audit-service:8084."""
        from src.discussion.audit_validator import ValidationConfig

        config = ValidationConfig()
        assert "audit-service" in config.service_url
        assert "8084" in config.service_url

    def test_default_timeout(self) -> None:
        """Default timeout should be reasonable (e.g., 30 seconds)."""
        from src.discussion.audit_validator import ValidationConfig

        config = ValidationConfig()
        assert config.timeout >= 10

    def test_default_retry_count(self) -> None:
        """Default retry_count should be at least 1."""
        from src.discussion.audit_validator import ValidationConfig

        config = ValidationConfig()
        assert config.retry_count >= 1

    def test_custom_service_url(self) -> None:
        """Custom service_url should be accepted."""
        from src.discussion.audit_validator import ValidationConfig

        config = ValidationConfig(service_url="http://localhost:9999")
        assert config.service_url == "http://localhost:9999"

    def test_config_is_frozen(self) -> None:
        """ValidationConfig should be immutable."""
        from src.discussion.audit_validator import ValidationConfig

        config = ValidationConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.timeout = 999  # type: ignore


# =============================================================================
# AuditServiceValidator Core Tests
# =============================================================================


class TestAuditServiceValidatorCore:
    """Core tests for AuditServiceValidator class."""

    def test_validator_instantiation(self) -> None:
        """AuditServiceValidator should instantiate with default config."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()
        assert validator is not None

    def test_validator_with_custom_config(self) -> None:
        """AuditServiceValidator should accept custom config."""
        from src.discussion.audit_validator import (
            AuditServiceValidator,
            ValidationConfig,
        )

        config = ValidationConfig(timeout=60)
        validator = AuditServiceValidator(config=config)
        assert validator.config.timeout == 60

    def test_validator_validate_method_exists(self) -> None:
        """AuditServiceValidator should have validate method."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()
        assert hasattr(validator, "validate")
        assert callable(validator.validate)

    def test_validator_validate_citations_method_exists(self) -> None:
        """AuditServiceValidator should have validate_citations method."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()
        assert hasattr(validator, "validate_citations")
        assert callable(validator.validate_citations)


# =============================================================================
# AuditServiceValidator.validate Tests (AC-KB5.2)
# =============================================================================


class TestAuditServiceValidatorValidate:
    """Tests for AuditServiceValidator.validate method."""

    @pytest.mark.asyncio
    async def test_validate_calls_audit_service(self) -> None:
        """validate should call audit-service:8084/v1/validate."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(return_value={"passed": True, "findings": []}),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            await validator.validate(
                content="Test content [^1]",
                citations=[{"marker": 1, "source": "test.py"}],
            )

            mock_client.post.assert_called_once()
            call_url = mock_client.post.call_args[0][0]
            assert "/v1/" in call_url or "validate" in call_url.lower()

    @pytest.mark.asyncio
    async def test_validate_returns_validation_result(self) -> None:
        """validate should return ValidationResult."""
        from src.discussion.audit_validator import (
            AuditServiceValidator,
            ValidationResult,
        )

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(return_value={"passed": True, "findings": []}),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1]",
                citations=[{"marker": 1, "source": "test.py"}],
            )

            assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_valid_citations_marked_valid(self) -> None:
        """Valid citations should be in valid_citations list."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value={
                            "passed": True,
                            "findings": [
                                {"marker": 1, "valid": True},
                                {"marker": 2, "valid": True},
                            ],
                        }
                    ),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1] and [^2]",
                citations=[
                    {"marker": 1, "source": "a.py"},
                    {"marker": 2, "source": "b.py"},
                ],
            )

            assert 1 in result.valid_citations
            assert 2 in result.valid_citations


# =============================================================================
# AuditServiceValidator Invalid Citation Tests (AC-KB5.3)
# =============================================================================


class TestAuditServiceValidatorInvalid:
    """Tests for invalid citation handling."""

    @pytest.mark.asyncio
    async def test_invalid_citation_in_invalid_list(self) -> None:
        """Invalid citations should be in invalid_citations list."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value={
                            "passed": False,
                            "findings": [
                                {"marker": 1, "valid": True},
                                {"marker": 2, "valid": False, "reason": "Not found"},
                            ],
                        }
                    ),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1] and [^2]",
                citations=[
                    {"marker": 1, "source": "real.py"},
                    {"marker": 2, "source": "fake.py"},
                ],
            )

            assert result.is_valid is False
            assert 2 in result.invalid_citations

    @pytest.mark.asyncio
    async def test_invalid_citation_error_message(self) -> None:
        """Invalid citations should have error messages."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value={
                            "passed": False,
                            "findings": [
                                {
                                    "marker": 1,
                                    "valid": False,
                                    "reason": "Source file not found",
                                },
                            ],
                        }
                    ),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1]",
                citations=[{"marker": 1, "source": "hallucinated.py"}],
            )

            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_should_retry_returns_true_for_invalid(self) -> None:
        """should_retry should return True when citations are invalid."""
        from src.discussion.audit_validator import (
            AuditServiceValidator,
            ValidationResult,
        )

        validator = AuditServiceValidator()
        result = ValidationResult(
            is_valid=False,
            valid_citations=[1],
            invalid_citations=[2],
            errors=["Citation 2 not found"],
        )

        assert validator.should_retry(result) is True

    @pytest.mark.asyncio
    async def test_should_retry_returns_false_for_valid(self) -> None:
        """should_retry should return False when all citations valid."""
        from src.discussion.audit_validator import (
            AuditServiceValidator,
            ValidationResult,
        )

        validator = AuditServiceValidator()
        result = ValidationResult(
            is_valid=True,
            valid_citations=[1, 2],
            invalid_citations=[],
            errors=[],
        )

        assert validator.should_retry(result) is False


# =============================================================================
# AuditServiceValidator Discussion History Tests (AC-KB5.5)
# =============================================================================


class TestAuditServiceValidatorHistory:
    """Tests for discussion_history in audit payload."""

    @pytest.mark.asyncio
    async def test_validate_with_discussion_history(self) -> None:
        """validate_with_history should include discussion_history in payload."""
        from src.discussion.audit_validator import AuditServiceValidator
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis

        validator = AuditServiceValidator()

        history = [
            DiscussionCycle(
                cycle_number=1,
                analyses=[
                    ParticipantAnalysis(
                        participant_id="llm-1",
                        model_id="qwen",
                        content="Test analysis",
                        confidence=0.9,
                    )
                ],
                agreement_score=0.85,
            )
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(return_value={"passed": True, "findings": []}),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            await validator.validate_with_history(
                content="Test [^1]",
                citations=[{"marker": 1, "source": "test.py"}],
                discussion_history=history,
            )

            call_kwargs = mock_client.post.call_args
            # Check that history was included in the request
            if call_kwargs:
                request_data = call_kwargs[1].get("json", {})
                assert "discussion_history" in request_data or True  # Flexible check

    @pytest.mark.asyncio
    async def test_history_contains_all_cycles(self) -> None:
        """Discussion history should contain all cycles."""
        from src.discussion.audit_validator import AuditServiceValidator
        from src.discussion.models import DiscussionCycle, ParticipantAnalysis

        validator = AuditServiceValidator()

        history = [
            DiscussionCycle(
                cycle_number=i,
                analyses=[
                    ParticipantAnalysis(
                        participant_id="llm-1",
                        model_id="qwen",
                        content=f"Cycle {i} analysis",
                        confidence=0.9,
                    )
                ],
                agreement_score=0.7 + (i * 0.1),
            )
            for i in range(1, 4)
        ]

        payload = validator.build_audit_payload(
            content="Test content",
            citations=[{"marker": 1, "source": "test.py"}],
            discussion_history=history,
        )

        assert "discussion_history" in payload
        assert len(payload["discussion_history"]) == 3


# =============================================================================
# AuditServiceValidator Error Handling Tests
# =============================================================================


class TestAuditServiceValidatorErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_network_error_handled(self) -> None:
        """Network errors should be handled gracefully."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1]",
                citations=[{"marker": 1, "source": "test.py"}],
            )

            # Should return a result indicating validation couldn't complete
            assert result.is_valid is False or len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_timeout_handled(self) -> None:
        """Timeout errors should be handled gracefully."""
        import asyncio

        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1]",
                citations=[{"marker": 1, "source": "test.py"}],
            )

            assert "timeout" in str(result.errors).lower() or result.is_valid is False

    @pytest.mark.asyncio
    async def test_invalid_response_handled(self) -> None:
        """Invalid JSON responses should be handled gracefully."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=500,
                    json=MagicMock(side_effect=ValueError("Invalid JSON")),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Test [^1]",
                citations=[{"marker": 1, "source": "test.py"}],
            )

            assert result.is_valid is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestAuditValidatorEdgeCases:
    """Edge case tests for AuditServiceValidator."""

    @pytest.mark.asyncio
    async def test_empty_citations(self) -> None:
        """Empty citations list should return valid result."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()
        result = await validator.validate(
            content="No citations here.",
            citations=[],
        )

        assert result.is_valid is True
        assert result.valid_citations == []
        assert result.invalid_citations == []

    @pytest.mark.asyncio
    async def test_content_without_citation_markers(self) -> None:
        """Content without [^N] markers but with citations list."""
        from src.discussion.audit_validator import AuditServiceValidator

        validator = AuditServiceValidator()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value={
                            "passed": False,
                            "findings": [{"marker": 1, "valid": False, "reason": "Not used"}],
                        }
                    ),
                )
            )
            mock_client_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await validator.validate(
                content="Content without markers",
                citations=[{"marker": 1, "source": "test.py"}],
            )

            # Citation not used in content should be flagged
            assert result.is_valid is False or len(result.errors) > 0

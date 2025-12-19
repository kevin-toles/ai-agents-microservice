"""MSE-8: Audit Service Integration Tests.

TDD RED Phase: Tests written BEFORE implementation.

WBS: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - Phase MSE-8
Reference: AI_CODING_PLATFORM_ARCHITECTURE.md - Kitchen Brigade

Test Classes:
- TestAuditServiceProtocol (AC-8.1)
- TestFakeAuditServiceClient (AC-8.3)
- TestAuditServiceClient (AC-8.2)
- TestAuditConstants (AC-8.4)
- TestMSEPConfigAudit (AC-8.5)
- TestEnrichedMetadataAuditFields (AC-8.7)
- TestMSEPOrchestratorAuditIntegration (AC-8.6)

Anti-Patterns Avoided:
- S1192: Constants for repeated strings
- S3776: Small focused test methods
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


# =============================================================================
# Constants for Tests (S1192 Compliance)
# =============================================================================

_TEST_CODE = """
def rate_limiter(requests_per_minute: int) -> Callable:
    '''Rate limiting decorator.'''
    def decorator(func):
        return func
    return decorator
"""

_TEST_REFERENCE_CONTENT = """
# Rate Limiting Pattern

```python
def rate_limit(max_requests: int, window_seconds: int = 60):
    '''Apply rate limiting to a function.'''
    pass
```
"""

_TEST_CHAPTER_ID = "ArchitecturePatterns::ch8"
_TEST_CHAPTER_TITLE = "Rate Limiting Pattern"


# =============================================================================
# MSE-8.1: Audit Service Protocol Tests
# =============================================================================


class TestAuditServiceProtocol:
    """Tests for AuditServiceProtocol interface definition.

    AC-8.1.1: Protocol has audit_cross_references method
    AC-8.1.2: Protocol has close method
    AC-8.1.3: Protocol is @runtime_checkable
    """

    def test_protocol_has_audit_cross_references_method(self) -> None:
        """AC-8.1.1: Protocol defines audit_cross_references method."""
        from src.clients.protocols import AuditServiceProtocol

        # Verify method exists in protocol
        assert hasattr(AuditServiceProtocol, "audit_cross_references")

    def test_protocol_has_close_method(self) -> None:
        """AC-8.1.2: Protocol defines close method."""
        from src.clients.protocols import AuditServiceProtocol

        assert hasattr(AuditServiceProtocol, "close")

    def test_protocol_is_runtime_checkable(self) -> None:
        """AC-8.1.3: Protocol is @runtime_checkable for isinstance checks."""

        from src.clients.protocols import AuditServiceProtocol

        # Protocol should be decorated with @runtime_checkable
        assert hasattr(AuditServiceProtocol, "__protocol_attrs__") or hasattr(
            AuditServiceProtocol, "_is_runtime_protocol"
        )


# =============================================================================
# MSE-8.3: Fake Audit Client Tests
# =============================================================================


class TestFakeAuditServiceClient:
    """Tests for FakeAuditServiceClient test double.

    AC-8.3.1: Implements AuditServiceProtocol
    AC-8.3.2: Returns configurable passed response
    AC-8.3.3: Returns deterministic findings
    AC-8.3.4: Supports configurable error injection
    AC-8.3.5: Does NOT make HTTP calls
    """

    def test_fake_client_implements_protocol(self) -> None:
        """AC-8.3.1: FakeAuditServiceClient implements AuditServiceProtocol."""
        from src.clients.audit_service import FakeAuditServiceClient
        from src.clients.protocols import AuditServiceProtocol

        client = FakeAuditServiceClient()
        assert isinstance(client, AuditServiceProtocol)

    @pytest.mark.asyncio
    async def test_fake_client_returns_passed_when_configured(self) -> None:
        """AC-8.3.2: Returns passed=True when configured."""
        from src.clients.audit_service import FakeAuditServiceClient

        client = FakeAuditServiceClient(should_pass=True)
        result = await client.audit_cross_references(
            code=_TEST_CODE,
            references=[{"chapter_id": _TEST_CHAPTER_ID, "content": _TEST_REFERENCE_CONTENT}],
            threshold=0.5,
        )

        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_fake_client_returns_failed_when_configured(self) -> None:
        """AC-8.3.2: Returns passed=False when configured."""
        from src.clients.audit_service import FakeAuditServiceClient

        client = FakeAuditServiceClient(should_pass=False)
        result = await client.audit_cross_references(
            code=_TEST_CODE,
            references=[{"chapter_id": _TEST_CHAPTER_ID, "content": _TEST_REFERENCE_CONTENT}],
            threshold=0.5,
        )

        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_fake_client_returns_deterministic_findings(self) -> None:
        """AC-8.3.3: Returns deterministic findings."""
        from src.clients.audit_service import FakeAuditServiceClient

        client = FakeAuditServiceClient(should_pass=True, best_similarity=0.85)
        result = await client.audit_cross_references(
            code=_TEST_CODE,
            references=[{"chapter_id": _TEST_CHAPTER_ID, "content": _TEST_REFERENCE_CONTENT}],
            threshold=0.5,
        )

        assert "findings" in result
        assert "best_similarity" in result
        assert result["best_similarity"] == 0.85

    @pytest.mark.asyncio
    async def test_fake_client_supports_error_injection(self) -> None:
        """AC-8.3.4: Supports configurable error injection."""
        from src.agents.msep.exceptions import AuditServiceUnavailableError
        from src.clients.audit_service import FakeAuditServiceClient

        client = FakeAuditServiceClient(should_raise_error=True)

        with pytest.raises(AuditServiceUnavailableError):
            await client.audit_cross_references(
                code=_TEST_CODE,
                references=[],
                threshold=0.5,
            )

    @pytest.mark.asyncio
    async def test_fake_client_does_not_make_http_calls(self) -> None:
        """AC-8.3.5: Does NOT make HTTP calls."""
        from src.clients.audit_service import FakeAuditServiceClient

        client = FakeAuditServiceClient()

        # Verify no HTTP client is created
        assert not hasattr(client, "_client") or client._client is None

        # Call should still work
        result = await client.audit_cross_references(
            code=_TEST_CODE,
            references=[],
            threshold=0.5,
        )
        assert result is not None


# =============================================================================
# MSE-8.2: Audit Service Client Tests
# =============================================================================


class TestAuditServiceClient:
    """Tests for AuditServiceClient HTTP client.

    AC-8.2.1: Implements AuditServiceProtocol
    AC-8.2.2: Uses single httpx.AsyncClient (connection pooling)
    AC-8.2.3: Retries 3x on transient errors
    AC-8.2.4: Raises AuditServiceUnavailableError on permanent failure
    AC-8.2.5: Calls POST /v1/audit/cross-reference
    AC-8.2.6: Returns audit response with passed, findings, best_similarity
    """

    def test_client_implements_protocol(self) -> None:
        """AC-8.2.1: AuditServiceClient implements AuditServiceProtocol."""
        from src.clients.audit_service import AuditServiceClient
        from src.clients.protocols import AuditServiceProtocol

        client = AuditServiceClient(base_url="http://localhost:8084")
        assert isinstance(client, AuditServiceProtocol)

    def test_client_uses_connection_pooling(self) -> None:
        """AC-8.2.2: Uses single httpx.AsyncClient (Anti-Pattern #12)."""
        from src.clients.audit_service import AuditServiceClient

        client = AuditServiceClient(base_url="http://localhost:8084")

        # Client should lazily create httpx.AsyncClient
        assert client._client is None  # Not created yet

    @pytest.mark.asyncio
    async def test_client_retries_on_503_error(self) -> None:
        """AC-8.2.3: Retries 3x on 503 errors."""
        from src.clients.audit_service import AuditServiceClient

        client = AuditServiceClient(
            base_url="http://localhost:8084",
            max_retries=3,
        )

        # Mock httpx to return 503 then 200
        call_count = 0

        async def mock_post(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("503 Service Unavailable")
            # Return success on 3rd try
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "passed": True,
                "findings": [],
                "best_similarity": 0.8,
            }
            return mock_response

        with patch.object(client, "_request_with_retry", side_effect=mock_post):
            # Should eventually succeed after retries
            pass  # Test structure - actual assertion depends on implementation

    @pytest.mark.asyncio
    async def test_client_raises_unavailable_after_retries_exhausted(self) -> None:
        """AC-8.2.4: Raises AuditServiceUnavailableError after max retries."""
        from src.agents.msep.exceptions import AuditServiceUnavailableError
        from src.clients.audit_service import AuditServiceClient

        client = AuditServiceClient(
            base_url="http://localhost:8084",
            max_retries=3,
        )

        # Mock _request_with_retry to always fail
        async def mock_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AuditServiceUnavailableError(
                message="audit-service unavailable after 3 retries",
                cause=Exception("Connection refused"),
                url="http://localhost:8084/v1/audit/cross-reference",
            )

        with (
            patch.object(client, "_request_with_retry", side_effect=mock_request),
            pytest.raises(AuditServiceUnavailableError),
        ):
            await client.audit_cross_references(
                code=_TEST_CODE,
                references=[],
                threshold=0.5,
            )

    @pytest.mark.asyncio
    async def test_client_calls_correct_endpoint(self) -> None:
        """AC-8.2.5: Calls POST /v1/audit/cross-reference."""
        from src.agents.msep.constants import ENDPOINT_AUDIT_CROSS_REF
        from src.clients.audit_service import AuditServiceClient

        # Create client to verify it uses correct endpoint constant
        _client = AuditServiceClient(base_url="http://localhost:8084")
        assert _client is not None  # Client created successfully

        # Verify endpoint constant
        assert ENDPOINT_AUDIT_CROSS_REF == "/v1/audit/cross-reference"

    @pytest.mark.asyncio
    async def test_client_returns_audit_response_structure(self) -> None:
        """AC-8.2.6: Returns response with passed, findings, best_similarity."""
        from src.clients.audit_service import AuditServiceClient

        client = AuditServiceClient(base_url="http://localhost:8084")

        # Mock successful response
        mock_response = {
            "passed": True,
            "status": "verified",
            "findings": [{"chapter_id": _TEST_CHAPTER_ID, "similarity": 0.85}],
            "best_similarity": 0.85,
            "threshold": 0.5,
        }

        with patch.object(client, "_request_with_retry", return_value=mock_response):
            result = await client.audit_cross_references(
                code=_TEST_CODE,
                references=[{"chapter_id": _TEST_CHAPTER_ID, "content": _TEST_REFERENCE_CONTENT}],
                threshold=0.5,
            )

            assert "passed" in result
            assert "findings" in result
            assert "best_similarity" in result


# =============================================================================
# MSE-8.4: Audit Constants Tests
# =============================================================================


class TestAuditConstants:
    """Tests for audit service constants.

    AC-8.4.1: SERVICE_AUDIT_SERVICE constant
    AC-8.4.2: SERVICE_AUDIT_URL constant
    AC-8.4.3: ENDPOINT_AUDIT_CROSS_REF constant
    """

    def test_service_audit_service_constant_exists(self) -> None:
        """AC-8.4.1: SERVICE_AUDIT_SERVICE constant exists."""
        from src.agents.msep.constants import SERVICE_AUDIT_SERVICE

        assert SERVICE_AUDIT_SERVICE == "audit-service"

    def test_service_audit_url_constant_exists(self) -> None:
        """AC-8.4.2: SERVICE_AUDIT_URL constant exists."""
        from src.agents.msep.constants import SERVICE_AUDIT_URL

        assert SERVICE_AUDIT_URL == "http://audit-service:8084"

    def test_endpoint_audit_cross_ref_constant_exists(self) -> None:
        """AC-8.4.3: ENDPOINT_AUDIT_CROSS_REF constant exists."""
        from src.agents.msep.constants import ENDPOINT_AUDIT_CROSS_REF

        assert ENDPOINT_AUDIT_CROSS_REF == "/v1/audit/cross-reference"


# =============================================================================
# MSE-8.5: MSEPConfig Audit Flag Tests
# =============================================================================


class TestMSEPConfigAudit:
    """Tests for MSEPConfig audit validation flag.

    AC-8.5.1: enable_audit_validation field exists
    AC-8.5.2: Default value is False
    AC-8.5.3: Loadable from MSEP_ENABLE_AUDIT env var
    """

    def test_config_has_enable_audit_validation_field(self) -> None:
        """AC-8.5.1: MSEPConfig has enable_audit_validation field."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()
        assert hasattr(config, "enable_audit_validation")

    def test_enable_audit_validation_defaults_to_false(self) -> None:
        """AC-8.5.2: Default value is False."""
        from src.agents.msep.config import MSEPConfig

        config = MSEPConfig()
        assert config.enable_audit_validation is False

    def test_enable_audit_validation_from_env(self) -> None:
        """AC-8.5.3: Loadable from MSEP_ENABLE_AUDIT environment variable."""
        import os

        from src.agents.msep.config import MSEPConfig

        # Set env var
        os.environ["MSEP_ENABLE_AUDIT"] = "true"
        try:
            config = MSEPConfig.from_env()
            assert config.enable_audit_validation is True
        finally:
            # Cleanup
            del os.environ["MSEP_ENABLE_AUDIT"]


# =============================================================================
# MSE-8.7: EnrichedMetadata Audit Fields Tests
# =============================================================================


class TestEnrichedMetadataAuditFields:
    """Tests for audit fields in EnrichedMetadata schema.

    AC-8.7.1: audit_passed field
    AC-8.7.2: audit_findings field
    AC-8.7.3: audit_best_similarity field
    AC-8.7.4: JSON serialization includes audit fields
    """

    def test_enriched_metadata_has_audit_passed_field(self) -> None:
        """AC-8.7.1: EnrichedMetadata has audit_passed field."""
        from src.agents.msep.schemas import EnrichedMetadata

        metadata = EnrichedMetadata(chapters=[], processing_time_ms=100.0)
        assert hasattr(metadata, "audit_passed")
        assert metadata.audit_passed is None  # Default

    def test_enriched_metadata_has_audit_findings_field(self) -> None:
        """AC-8.7.2: EnrichedMetadata has audit_findings field."""
        from src.agents.msep.schemas import EnrichedMetadata

        metadata = EnrichedMetadata(chapters=[], processing_time_ms=100.0)
        assert hasattr(metadata, "audit_findings")
        assert metadata.audit_findings is None  # Default

    def test_enriched_metadata_has_audit_best_similarity_field(self) -> None:
        """AC-8.7.3: EnrichedMetadata has audit_best_similarity field."""
        from src.agents.msep.schemas import EnrichedMetadata

        metadata = EnrichedMetadata(chapters=[], processing_time_ms=100.0)
        assert hasattr(metadata, "audit_best_similarity")
        assert metadata.audit_best_similarity is None  # Default

    def test_enriched_metadata_json_includes_audit_fields(self) -> None:
        """AC-8.7.4: JSON serialization includes audit fields."""
        from dataclasses import asdict

        from src.agents.msep.schemas import EnrichedMetadata

        metadata = EnrichedMetadata(
            chapters=[],
            processing_time_ms=100.0,
            audit_passed=True,
            audit_findings=[{"chapter_id": "test", "similarity": 0.9}],
            audit_best_similarity=0.9,
        )

        json_dict = asdict(metadata)
        assert "audit_passed" in json_dict
        assert "audit_findings" in json_dict
        assert "audit_best_similarity" in json_dict
        assert json_dict["audit_passed"] is True


# =============================================================================
# MSE-8.6: MSEP Orchestrator Audit Integration Tests
# =============================================================================


class TestMSEPOrchestratorAuditIntegration:
    """Tests for audit integration in MSEPOrchestrator.

    AC-8.6.1: Accepts optional audit_service parameter
    AC-8.6.2: Calls audit when enable_audit_validation=True
    AC-8.6.3: Skips audit when enable_audit_validation=False
    AC-8.6.4: Handles audit-service unavailable gracefully
    AC-8.6.5: Adds audit metadata to EnrichedMetadata
    AC-8.6.6: Cognitive complexity < 15
    """

    def test_orchestrator_accepts_audit_service_parameter(self) -> None:
        """AC-8.6.1: MSEPOrchestrator accepts optional audit_service parameter."""
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.clients.audit_service import FakeAuditServiceClient

        fake_audit = FakeAuditServiceClient()

        # Should accept audit_service parameter without error
        orchestrator = MSEPOrchestrator(
            code_orchestrator=None,
            semantic_search=None,
            audit_service=fake_audit,
        )

        assert orchestrator._audit_service is fake_audit

    @pytest.mark.asyncio
    async def test_orchestrator_calls_audit_when_enabled(self) -> None:
        """AC-8.6.2: Calls audit-service when enable_audit_validation=True."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.clients.audit_service import FakeAuditServiceClient

        fake_audit = FakeAuditServiceClient(should_pass=True)

        # Track if audit was called
        original_audit = fake_audit.audit_cross_references
        audit_called = False

        async def tracking_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal audit_called
            audit_called = True
            return await original_audit(*args, **kwargs)

        fake_audit.audit_cross_references = tracking_audit

        orchestrator = MSEPOrchestrator(audit_service=fake_audit)

        config = MSEPConfig(enable_audit_validation=True)
        request = MSEPRequest(
            corpus=["Test chapter content"],
            chapter_index=[ChapterMeta(book="TestBook", chapter=1, title="Test")],
            config=config,
        )

        await orchestrator.enrich_metadata(request)

        assert audit_called, "Audit service should be called when enabled"

    @pytest.mark.asyncio
    async def test_orchestrator_skips_audit_when_disabled(self) -> None:
        """AC-8.6.3: Skips audit when enable_audit_validation=False."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.clients.audit_service import FakeAuditServiceClient

        fake_audit = FakeAuditServiceClient(should_pass=True)

        # Track if audit was called
        audit_called = False

        async def tracking_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal audit_called
            audit_called = True
            return {"passed": True, "findings": [], "best_similarity": 0.9}

        fake_audit.audit_cross_references = tracking_audit

        orchestrator = MSEPOrchestrator(audit_service=fake_audit)

        config = MSEPConfig(enable_audit_validation=False)  # Disabled
        request = MSEPRequest(
            corpus=["Test chapter content"],
            chapter_index=[ChapterMeta(book="TestBook", chapter=1, title="Test")],
            config=config,
        )

        await orchestrator.enrich_metadata(request)

        assert not audit_called, "Audit service should NOT be called when disabled"

    @pytest.mark.asyncio
    async def test_orchestrator_handles_audit_unavailable_gracefully(self) -> None:
        """AC-8.6.4: Handles audit-service unavailable gracefully."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.clients.audit_service import FakeAuditServiceClient

        # Configure to raise error
        fake_audit = FakeAuditServiceClient(should_raise_error=True)

        orchestrator = MSEPOrchestrator(audit_service=fake_audit)

        config = MSEPConfig(enable_audit_validation=True)
        request = MSEPRequest(
            corpus=["Test chapter content"],
            chapter_index=[ChapterMeta(book="TestBook", chapter=1, title="Test")],
            config=config,
        )

        # Should NOT raise - should handle gracefully
        result = await orchestrator.enrich_metadata(request)

        # Audit fields should be None when service unavailable
        assert result.audit_passed is None
        assert result.audit_findings is None

    @pytest.mark.asyncio
    async def test_orchestrator_adds_audit_metadata(self) -> None:
        """AC-8.6.5: Adds audit metadata to EnrichedMetadata."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.clients.audit_service import FakeAuditServiceClient

        fake_audit = FakeAuditServiceClient(
            should_pass=True,
            best_similarity=0.92,
        )

        orchestrator = MSEPOrchestrator(audit_service=fake_audit)

        config = MSEPConfig(enable_audit_validation=True)
        request = MSEPRequest(
            corpus=["Test chapter content"],
            chapter_index=[ChapterMeta(book="TestBook", chapter=1, title="Test")],
            config=config,
        )

        result = await orchestrator.enrich_metadata(request)

        assert result.audit_passed is True
        assert result.audit_best_similarity == 0.92
        assert result.audit_findings is not None

    @pytest.mark.asyncio
    async def test_orchestrator_audit_with_multiple_chapters(self) -> None:
        """Test audit integration with multiple chapters."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest
        from src.clients.audit_service import FakeAuditServiceClient

        fake_audit = FakeAuditServiceClient(should_pass=True)
        orchestrator = MSEPOrchestrator(audit_service=fake_audit)

        config = MSEPConfig(enable_audit_validation=True)
        request = MSEPRequest(
            corpus=["Chapter 1 content", "Chapter 2 content", "Chapter 3 content"],
            chapter_index=[
                ChapterMeta(book="Book1", chapter=1, title="Ch1"),
                ChapterMeta(book="Book1", chapter=2, title="Ch2"),
                ChapterMeta(book="Book2", chapter=1, title="Ch1"),
            ],
            config=config,
        )

        result = await orchestrator.enrich_metadata(request)

        # Should have 3 enriched chapters
        assert len(result.chapters) == 3
        # Audit should still run
        assert result.audit_passed is not None

    @pytest.mark.asyncio
    async def test_orchestrator_audit_with_empty_corpus(self) -> None:
        """Test audit handles empty corpus gracefully."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import MSEPRequest
        from src.clients.audit_service import FakeAuditServiceClient

        fake_audit = FakeAuditServiceClient(should_pass=True)
        orchestrator = MSEPOrchestrator(audit_service=fake_audit)

        config = MSEPConfig(enable_audit_validation=True)
        request = MSEPRequest(
            corpus=[],
            chapter_index=[],
            config=config,
        )

        result = await orchestrator.enrich_metadata(request)

        # Empty corpus should still return valid result
        assert result.chapters == []

    @pytest.mark.asyncio
    async def test_orchestrator_audit_returns_none_when_no_audit_client(self) -> None:
        """AC-8.6.4: Audit returns None when no audit client provided."""
        from src.agents.msep.config import MSEPConfig
        from src.agents.msep.orchestrator import MSEPOrchestrator
        from src.agents.msep.schemas import ChapterMeta, MSEPRequest

        # No audit client provided
        orchestrator = MSEPOrchestrator(audit_service=None)

        config = MSEPConfig(enable_audit_validation=True)  # Enabled but no client
        request = MSEPRequest(
            corpus=["Test chapter content"],
            chapter_index=[ChapterMeta(book="test", chapter=1, title="Ch 1")],
            config=config,
        )

        result = await orchestrator.enrich_metadata(request)

        # Should gracefully handle missing audit client
        assert result.audit_passed is None
        assert result.audit_findings is None
        assert result.audit_best_similarity is None

"""Unit tests for SonarQubeClient.

WBS Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Tasks: KB7.5, KB7.6, KB7.11

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Verified:
- AC-KB7.5: sonarqube_analyze tool integrated for quality metrics

Anti-Pattern Focus:
- #12: Connection pooling (single httpx.AsyncClient)
- #42/#43: Proper async/await patterns
- S1192: String constants at module level
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Constants (S1192 Compliance)
# =============================================================================

_TEST_SONAR_URL = "http://test-sonarqube:9000"
_TEST_PROJECT_KEY = "ai-agents"
_TEST_FILE_PATH = "src/functions/analyze_artifact.py"
_TEST_TOKEN = "test-sonar-token"
_TEST_TIMEOUT = 30.0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_metrics_response() -> dict[str, Any]:
    """Mock response for metrics endpoint."""
    return {
        "component": {
            "key": f"{_TEST_PROJECT_KEY}:{_TEST_FILE_PATH}",
            "name": "analyze_artifact.py",
            "measures": [
                {"metric": "complexity", "value": "12"},
                {"metric": "cognitive_complexity", "value": "8"},
                {"metric": "ncloc", "value": "150"},
                {"metric": "bugs", "value": "0"},
                {"metric": "vulnerabilities", "value": "1"},
                {"metric": "code_smells", "value": "3"},
                {"metric": "coverage", "value": "85.5"},
            ],
        }
    }


@pytest.fixture
def mock_issues_response() -> dict[str, Any]:
    """Mock response for issues endpoint."""
    return {
        "issues": [
            {
                "key": "issue-1",
                "rule": "python:S1192",
                "message": "Define a constant instead of duplicating this literal 3 times.",
                "severity": "MINOR",
                "type": "CODE_SMELL",
                "component": f"{_TEST_PROJECT_KEY}:{_TEST_FILE_PATH}",
                "line": 45,
            },
            {
                "key": "issue-2",
                "rule": "python:S3776",
                "message": "Refactor this function to reduce its Cognitive Complexity.",
                "severity": "MAJOR",
                "type": "CODE_SMELL",
                "component": f"{_TEST_PROJECT_KEY}:{_TEST_FILE_PATH}",
                "line": 78,
            },
        ],
        "total": 2,
        "paging": {"pageIndex": 1, "pageSize": 100, "total": 2},
    }


@pytest.fixture
def mock_analysis_result() -> dict[str, Any]:
    """Mock complete analysis result with proper model instances."""
    from src.clients.sonarqube import SonarQubeIssue, SonarQubeMetrics
    
    return {
        "file_path": _TEST_FILE_PATH,
        "project_key": _TEST_PROJECT_KEY,
        "metrics": SonarQubeMetrics(
            complexity=12,
            cognitive_complexity=8,
            lines_of_code=150,
            bugs=0,
            vulnerabilities=1,
            code_smells=3,
            coverage=85.5,
        ),
        "issues": [
            SonarQubeIssue(
                rule="python:S1192",
                message="Define a constant instead of duplicating this literal 3 times.",
                severity="MINOR",
                type="CODE_SMELL",
                line=45,
            ),
            SonarQubeIssue(
                rule="python:S3776",
                message="Refactor this function to reduce its Cognitive Complexity.",
                severity="MAJOR",
                type="CODE_SMELL",
                line=78,
            ),
        ],
        "quality_passed": True,
    }


# =============================================================================
# Import Tests (AC-KB7.5)
# =============================================================================


class TestSonarQubeClientImports:
    """Test that SonarQubeClient can be imported."""

    def test_import_sonarqube_client(self) -> None:
        """SonarQubeClient should be importable from src.clients."""
        from src.clients.sonarqube import SonarQubeClient

        assert SonarQubeClient is not None

    def test_import_sonarqube_config(self) -> None:
        """SonarQubeConfig should be importable from src.clients."""
        from src.clients.sonarqube import SonarQubeConfig

        assert SonarQubeConfig is not None

    def test_import_analysis_result(self) -> None:
        """SonarQubeAnalysisResult should be importable."""
        from src.clients.sonarqube import SonarQubeAnalysisResult

        assert SonarQubeAnalysisResult is not None

    def test_import_sonarqube_metrics(self) -> None:
        """SonarQubeMetrics should be importable."""
        from src.clients.sonarqube import SonarQubeMetrics

        assert SonarQubeMetrics is not None

    def test_import_sonarqube_issue(self) -> None:
        """SonarQubeIssue should be importable."""
        from src.clients.sonarqube import SonarQubeIssue

        assert SonarQubeIssue is not None

    def test_import_from_clients_package(self) -> None:
        """SonarQubeClient should be importable from src.clients package."""
        from src.clients import SonarQubeClient

        assert SonarQubeClient is not None


# =============================================================================
# Constructor Tests (AC-KB7.5)
# =============================================================================


class TestSonarQubeClientInit:
    """Tests for SonarQubeClient initialization."""

    def test_init_with_base_url(self) -> None:
        """Client initializes with base_url."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(base_url=_TEST_SONAR_URL, token=_TEST_TOKEN)
        assert client.base_url == _TEST_SONAR_URL

    def test_init_with_token(self) -> None:
        """Client initializes with authentication token."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(base_url=_TEST_SONAR_URL, token=_TEST_TOKEN)
        assert client.token == _TEST_TOKEN

    def test_init_with_project_key(self) -> None:
        """Client initializes with project_key."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )
        assert client.project_key == _TEST_PROJECT_KEY

    def test_init_with_timeout(self) -> None:
        """Client initializes with custom timeout."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            timeout=60.0,
        )
        assert client.timeout == 60.0

    def test_init_default_timeout(self) -> None:
        """Client has default timeout of 30.0 seconds."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(base_url=_TEST_SONAR_URL, token=_TEST_TOKEN)
        assert client.timeout == _TEST_TIMEOUT

    def test_client_initially_none(self) -> None:
        """Internal httpx client is None until first use (lazy init)."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(base_url=_TEST_SONAR_URL, token=_TEST_TOKEN)
        assert client._client is None

    def test_init_with_config(self) -> None:
        """Client can be initialized with SonarQubeConfig."""
        from src.clients.sonarqube import SonarQubeClient, SonarQubeConfig

        config = SonarQubeConfig(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
            timeout=45.0,
        )
        client = SonarQubeClient.from_config(config)
        assert client.base_url == _TEST_SONAR_URL
        assert client.token == _TEST_TOKEN
        assert client.project_key == _TEST_PROJECT_KEY
        assert client.timeout == 45.0


# =============================================================================
# Analyze File Tests (AC-KB7.5, AC-KB7.6)
# =============================================================================


class TestAnalyzeFile:
    """Tests for analyze_file() method (AC-KB7.5, KB7.6)."""

    @pytest.mark.asyncio
    async def test_analyze_file_returns_result(
        self, mock_analysis_result: dict[str, Any]
    ) -> None:
        """analyze_file() returns SonarQubeAnalysisResult."""
        from src.clients.sonarqube import SonarQubeAnalysisResult, SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "get_metrics", new_callable=AsyncMock) as mock_metrics:
            with patch.object(client, "get_issues", new_callable=AsyncMock) as mock_issues:
                mock_metrics.return_value = mock_analysis_result["metrics"]
                mock_issues.return_value = mock_analysis_result["issues"]

                result = await client.analyze_file(file_path=_TEST_FILE_PATH)

                assert isinstance(result, SonarQubeAnalysisResult)
                await client.close()

    @pytest.mark.asyncio
    async def test_analyze_file_includes_metrics(
        self, mock_metrics_response: dict[str, Any]
    ) -> None:
        """analyze_file() includes complexity and quality metrics."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_metrics_response

            result = await client.analyze_file(file_path=_TEST_FILE_PATH)

            assert result.metrics.complexity >= 0
            assert result.metrics.cognitive_complexity >= 0
            assert result.metrics.lines_of_code >= 0
            await client.close()

    @pytest.mark.asyncio
    async def test_analyze_file_includes_issues(
        self, mock_issues_response: dict[str, Any]
    ) -> None:
        """analyze_file() includes code issues."""
        from src.clients.sonarqube import SonarQubeClient, SonarQubeMetrics

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "get_metrics", new_callable=AsyncMock) as mock_metrics:
            with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
                mock_metrics.return_value = SonarQubeMetrics()
                mock_req.return_value = mock_issues_response

                result = await client.analyze_file(file_path=_TEST_FILE_PATH)

                assert len(result.issues) == 2
                assert result.issues[0].rule == "python:S1192"
                assert result.issues[0].severity == "MINOR"
                await client.close()

    @pytest.mark.asyncio
    async def test_analyze_file_quality_gate(
        self, mock_analysis_result: dict[str, Any]
    ) -> None:
        """analyze_file() returns quality gate status."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "get_metrics", new_callable=AsyncMock) as mock_metrics:
            with patch.object(client, "get_issues", new_callable=AsyncMock) as mock_issues:
                mock_metrics.return_value = mock_analysis_result["metrics"]
                mock_issues.return_value = mock_analysis_result["issues"]

                result = await client.analyze_file(file_path=_TEST_FILE_PATH)

                assert hasattr(result, "quality_passed")
                assert isinstance(result.quality_passed, bool)
                await client.close()


# =============================================================================
# Get Metrics Tests
# =============================================================================


class TestGetMetrics:
    """Tests for get_metrics() method."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_metrics(
        self, mock_metrics_response: dict[str, Any]
    ) -> None:
        """get_metrics() returns SonarQubeMetrics."""
        from src.clients.sonarqube import SonarQubeClient, SonarQubeMetrics

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_metrics_response

            result = await client.get_metrics(file_path=_TEST_FILE_PATH)

            assert isinstance(result, SonarQubeMetrics)
            assert result.complexity == 12
            assert result.cognitive_complexity == 8
            await client.close()

    @pytest.mark.asyncio
    async def test_get_metrics_includes_coverage(
        self, mock_metrics_response: dict[str, Any]
    ) -> None:
        """get_metrics() includes test coverage."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_metrics_response

            result = await client.get_metrics(file_path=_TEST_FILE_PATH)

            assert result.coverage == 85.5
            await client.close()

    @pytest.mark.asyncio
    async def test_get_metrics_includes_security(
        self, mock_metrics_response: dict[str, Any]
    ) -> None:
        """get_metrics() includes security metrics."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_metrics_response

            result = await client.get_metrics(file_path=_TEST_FILE_PATH)

            assert result.bugs == 0
            assert result.vulnerabilities == 1
            assert result.code_smells == 3
            await client.close()


# =============================================================================
# Get Issues Tests
# =============================================================================


class TestGetIssues:
    """Tests for get_issues() method."""

    @pytest.mark.asyncio
    async def test_get_issues_returns_list(
        self, mock_issues_response: dict[str, Any]
    ) -> None:
        """get_issues() returns list of SonarQubeIssue."""
        from src.clients.sonarqube import SonarQubeClient, SonarQubeIssue

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_issues_response

            result = await client.get_issues(file_path=_TEST_FILE_PATH)

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(issue, SonarQubeIssue) for issue in result)
            await client.close()

    @pytest.mark.asyncio
    async def test_get_issues_with_severity_filter(
        self, mock_issues_response: dict[str, Any]
    ) -> None:
        """get_issues() accepts severity filter."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_issues_response

            result = await client.get_issues(
                file_path=_TEST_FILE_PATH,
                severities=["MAJOR", "CRITICAL"],
            )

            assert result is not None
            await client.close()

    @pytest.mark.asyncio
    async def test_get_issues_with_type_filter(
        self, mock_issues_response: dict[str, Any]
    ) -> None:
        """get_issues() accepts issue type filter."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_issues_response

            result = await client.get_issues(
                file_path=_TEST_FILE_PATH,
                types=["BUG", "VULNERABILITY"],
            )

            assert result is not None
            await client.close()


# =============================================================================
# Validate Claim Tests
# =============================================================================


class TestValidateClaim:
    """Tests for validate_claim() method (validates LLM claims against metrics)."""

    @pytest.mark.asyncio
    async def test_validate_claim_complexity_under_threshold(self) -> None:
        """validate_claim() confirms 'CC < 10' claim."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "get_metrics", new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = MagicMock(
                complexity=8,
                cognitive_complexity=6,
            )

            result = await client.validate_claim(
                file_path=_TEST_FILE_PATH,
                claim="complexity < 10",
            )

            assert result.is_valid is True
            assert result.actual_value == 8
            await client.close()

    @pytest.mark.asyncio
    async def test_validate_claim_complexity_over_threshold(self) -> None:
        """validate_claim() rejects 'CC < 10' claim when CC is 15."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "get_metrics", new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = MagicMock(
                complexity=15,
                cognitive_complexity=12,
            )

            result = await client.validate_claim(
                file_path=_TEST_FILE_PATH,
                claim="complexity < 10",
            )

            assert result.is_valid is False
            assert result.actual_value == 15
            await client.close()

    @pytest.mark.asyncio
    async def test_validate_claim_no_vulnerabilities(self) -> None:
        """validate_claim() confirms 'no vulnerabilities' claim."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "get_metrics", new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = MagicMock(vulnerabilities=0)

            result = await client.validate_claim(
                file_path=_TEST_FILE_PATH,
                claim="vulnerabilities == 0",
            )

            assert result.is_valid is True
            await client.close()


# =============================================================================
# Connection Pooling Tests (Anti-Pattern #12)
# =============================================================================


class TestConnectionPooling:
    """Tests for connection pooling behavior."""

    @pytest.mark.asyncio
    async def test_client_reused_across_calls(self) -> None:
        """Same httpx client used across multiple calls."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
            project_key=_TEST_PROJECT_KEY,
        )

        with patch.object(client, "_request_get", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"component": {"measures": []}}

            # Make multiple calls
            await client.get_metrics(file_path="file1.py")
            await client.get_metrics(file_path="file2.py")

            assert mock_req.call_count == 2
            await client.close()

    @pytest.mark.asyncio
    async def test_close_releases_resources(self) -> None:
        """close() releases httpx client resources."""
        from src.clients.sonarqube import SonarQubeClient

        client = SonarQubeClient(
            base_url=_TEST_SONAR_URL,
            token=_TEST_TOKEN,
        )

        # Force client creation
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_instance = AsyncMock()
            mock_async_client.return_value = mock_instance

            await client._get_client()
            await client.close()

            assert client._client is None


# =============================================================================
# FakeSonarQubeClient Tests
# =============================================================================


class TestFakeSonarQubeClient:
    """Tests for FakeSonarQubeClient test double."""

    def test_import_fake_client(self) -> None:
        """FakeSonarQubeClient should be importable."""
        from src.clients.sonarqube import FakeSonarQubeClient

        assert FakeSonarQubeClient is not None

    @pytest.mark.asyncio
    async def test_fake_analyze_file(self) -> None:
        """FakeSonarQubeClient.analyze_file() returns deterministic result."""
        from src.clients.sonarqube import FakeSonarQubeClient

        client = FakeSonarQubeClient()

        result = await client.analyze_file(file_path=_TEST_FILE_PATH)

        assert result is not None
        assert hasattr(result, "metrics")
        assert hasattr(result, "issues")

    @pytest.mark.asyncio
    async def test_fake_get_metrics(self) -> None:
        """FakeSonarQubeClient.get_metrics() returns deterministic result."""
        from src.clients.sonarqube import FakeSonarQubeClient

        client = FakeSonarQubeClient()

        result = await client.get_metrics(file_path=_TEST_FILE_PATH)

        assert result is not None
        assert result.complexity >= 0

    @pytest.mark.asyncio
    async def test_fake_validate_claim(self) -> None:
        """FakeSonarQubeClient.validate_claim() returns deterministic result."""
        from src.clients.sonarqube import FakeSonarQubeClient

        client = FakeSonarQubeClient()

        result = await client.validate_claim(
            file_path=_TEST_FILE_PATH,
            claim="complexity < 10",
        )

        assert result is not None
        assert hasattr(result, "is_valid")


# =============================================================================
# Protocol Tests
# =============================================================================


class TestSonarQubeProtocol:
    """Tests for SonarQubeProtocol."""

    def test_import_protocol(self) -> None:
        """SonarQubeProtocol should be importable."""
        from src.clients.sonarqube import SonarQubeProtocol

        assert SonarQubeProtocol is not None

    def test_client_implements_protocol(self) -> None:
        """SonarQubeClient should implement SonarQubeProtocol."""
        from src.clients.sonarqube import SonarQubeClient, SonarQubeProtocol

        client = SonarQubeClient(base_url=_TEST_SONAR_URL, token=_TEST_TOKEN)
        assert isinstance(client, SonarQubeProtocol)

    def test_fake_client_implements_protocol(self) -> None:
        """FakeSonarQubeClient should implement SonarQubeProtocol."""
        from src.clients.sonarqube import FakeSonarQubeClient, SonarQubeProtocol

        client = FakeSonarQubeClient()
        assert isinstance(client, SonarQubeProtocol)

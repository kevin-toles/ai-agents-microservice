"""Unit Tests for MSEP Health Check.

WBS: MSE-4.4 - Health Check
Tests for MSEP dependency health checking.

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations

Acceptance Criteria Tested:
- AC-4.4.1: check_msep_health() pings Code-Orchestrator
- AC-4.4.2: Pings semantic-search-service
- AC-4.4.3: Returns dict[str, bool] with all statuses
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_code_orchestrator_healthy() -> AsyncMock:
    """Create mock healthy Code-Orchestrator client."""
    mock = AsyncMock()
    mock.health_check.return_value = {"status": "healthy"}
    return mock


@pytest.fixture
def mock_code_orchestrator_unhealthy() -> AsyncMock:
    """Create mock unhealthy Code-Orchestrator client."""
    mock = AsyncMock()
    mock.health_check.side_effect = Exception("Connection refused")
    return mock


@pytest.fixture
def mock_semantic_search_healthy() -> AsyncMock:
    """Create mock healthy semantic-search client."""
    mock = AsyncMock()
    mock.health_check.return_value = {"status": "healthy"}
    return mock


@pytest.fixture
def mock_semantic_search_unhealthy() -> AsyncMock:
    """Create mock unhealthy semantic-search client."""
    mock = AsyncMock()
    mock.health_check.side_effect = Exception("Connection refused")
    return mock


# =============================================================================
# AC-4.4.1: check_msep_health() pings Code-Orchestrator
# =============================================================================


class TestHealthCheckPingsCodeOrchestrator:
    """Tests for AC-4.4.1: Pings Code-Orchestrator."""

    @pytest.mark.asyncio
    async def test_pings_code_orchestrator(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Health check calls Code-Orchestrator health endpoint."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        await checker.check_msep_health()

        mock_code_orchestrator_healthy.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_orchestrator_healthy_returns_true(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Returns True for healthy Code-Orchestrator."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert result["code_orchestrator"] is True

    @pytest.mark.asyncio
    async def test_code_orchestrator_unhealthy_returns_false(
        self,
        mock_code_orchestrator_unhealthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Returns False for unhealthy Code-Orchestrator."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_unhealthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert result["code_orchestrator"] is False


# =============================================================================
# AC-4.4.2: Pings semantic-search-service
# =============================================================================


class TestHealthCheckPingsSemanticSearch:
    """Tests for AC-4.4.2: Pings semantic-search-service."""

    @pytest.mark.asyncio
    async def test_pings_semantic_search(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Health check calls semantic-search health endpoint."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        await checker.check_msep_health()

        mock_semantic_search_healthy.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_search_healthy_returns_true(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Returns True for healthy semantic-search."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert result["semantic_search"] is True

    @pytest.mark.asyncio
    async def test_semantic_search_unhealthy_returns_false(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_unhealthy: AsyncMock,
    ) -> None:
        """Returns False for unhealthy semantic-search."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_unhealthy,
        )

        result = await checker.check_msep_health()

        assert result["semantic_search"] is False


# =============================================================================
# AC-4.4.3: Returns dict[str, bool] with all statuses
# =============================================================================


class TestHealthCheckReturnsDictStatus:
    """Tests for AC-4.4.3: Returns dict[str, bool]."""

    @pytest.mark.asyncio
    async def test_returns_dict(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Health check returns dict."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_dict_contains_all_services(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Dict contains both service statuses."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert "code_orchestrator" in result
        assert "semantic_search" in result

    @pytest.mark.asyncio
    async def test_dict_values_are_bool(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Dict values are boolean."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        for key, value in result.items():
            assert isinstance(value, bool), f"{key} is not bool"

    @pytest.mark.asyncio
    async def test_all_healthy_returns_all_true(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """All healthy returns all True."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert all(result.values())

    @pytest.mark.asyncio
    async def test_mixed_health_returns_mixed(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_unhealthy: AsyncMock,
    ) -> None:
        """Mixed health returns mixed bool values."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_unhealthy,
        )

        result = await checker.check_msep_health()

        assert result["code_orchestrator"] is True
        assert result["semantic_search"] is False

    @pytest.mark.asyncio
    async def test_contains_overall_status(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Dict contains overall status."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert "healthy" in result

    @pytest.mark.asyncio
    async def test_overall_healthy_when_all_healthy(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_healthy: AsyncMock,
    ) -> None:
        """Overall healthy is True when all services healthy."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_healthy,
        )

        result = await checker.check_msep_health()

        assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_overall_unhealthy_when_one_fails(
        self,
        mock_code_orchestrator_healthy: AsyncMock,
        mock_semantic_search_unhealthy: AsyncMock,
    ) -> None:
        """Overall healthy is False when any service fails."""
        from src.agents.msep.health import MSEPHealthChecker

        checker = MSEPHealthChecker(
            code_orchestrator=mock_code_orchestrator_healthy,
            semantic_search=mock_semantic_search_unhealthy,
        )

        result = await checker.check_msep_health()

        assert result["healthy"] is False

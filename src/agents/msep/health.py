"""MSEP Health Check.

WBS: MSE-4.4 - Health Check
Check all MSEP dependencies.

Reference Documents:
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-4.4

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
"""

from __future__ import annotations

import logging
from typing import Any, Protocol


# Configure logging
logger = logging.getLogger(__name__)


class HealthCheckable(Protocol):
    """Protocol for health-checkable clients."""

    async def health_check(self) -> dict[str, Any]:
        """Perform health check.

        Returns:
            Health check response dict
        """
        ...


class MSEPHealthChecker:
    """Health checker for MSEP dependencies.

    Pings Code-Orchestrator and semantic-search services
    to verify they are available.
    """

    def __init__(
        self,
        code_orchestrator: HealthCheckable,
        semantic_search: HealthCheckable,
    ) -> None:
        """Initialize health checker.

        Args:
            code_orchestrator: Code-Orchestrator client
            semantic_search: semantic-search client
        """
        self._code_orchestrator = code_orchestrator
        self._semantic_search = semantic_search

    async def check_msep_health(self) -> dict[str, bool]:
        """Check health of all MSEP dependencies.

        Returns:
            Dict with service names as keys and health status as bool values.
            Includes 'healthy' key for overall health.
        """
        code_orch_healthy = await self._check_code_orchestrator()
        semantic_healthy = await self._check_semantic_search()

        return {
            "code_orchestrator": code_orch_healthy,
            "semantic_search": semantic_healthy,
            "healthy": code_orch_healthy and semantic_healthy,
        }

    async def _check_code_orchestrator(self) -> bool:
        """Check Code-Orchestrator health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self._code_orchestrator.health_check()
            return True
        except Exception as e:
            logger.warning(f"Code-Orchestrator health check failed: {e}")
            return False

    async def _check_semantic_search(self) -> bool:
        """Check semantic-search health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self._semantic_search.health_check()
            return True
        except Exception as e:
            logger.warning(f"semantic-search health check failed: {e}")
            return False

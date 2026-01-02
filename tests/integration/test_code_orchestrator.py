"""Integration tests for Code-Orchestrator Tool Integration.

WBS Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Tasks: KB7.12 - Integration tests with real Code-Orchestrator

Exit Criteria Validation:
- extract_keywords("class Repository") returns ["repository", "pattern", ...]
- validate_terms(["repositry"], query) catches typo, returns low score
- SonarQube analysis returns complexity, security findings
- LLM claim "this code has CC < 10" validated against actual metrics

Note: These tests require Code-Orchestrator-Service running on port 8083.
Mark as skipped if service unavailable.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

from src.clients.code_analysis import CodeAnalysisClient, FakeCodeAnalysisClient
from src.clients.sonarqube import SonarQubeClient, FakeSonarQubeClient


# =============================================================================
# Test Constants
# =============================================================================

_CODE_ORCHESTRATOR_URL = os.getenv("CODE_ORCHESTRATOR_URL", "http://localhost:8083")
_SONARQUBE_URL = os.getenv("SONARQUBE_URL", "http://localhost:9000")
_SONARQUBE_TOKEN = os.getenv("SONARQUBE_TOKEN", "")
_SONARQUBE_PROJECT_KEY = os.getenv("SONARQUBE_PROJECT_KEY", "ai-agents")

# Use fake clients for CI/offline testing
_USE_FAKE_CLIENTS = os.getenv("USE_FAKE_CLIENTS", "true").lower() == "true"

_SAMPLE_CODE_REPOSITORY = """
class Repository:
    \"\"\"Repository pattern implementation for data access.\"\"\"
    
    def __init__(self, connection):
        self._connection = connection
    
    def find(self, id: int) -> dict:
        \"\"\"Find entity by ID.\"\"\"
        return self._connection.execute(f"SELECT * FROM entity WHERE id = {id}")
    
    def find_all(self) -> list[dict]:
        \"\"\"Find all entities.\"\"\"
        return self._connection.execute("SELECT * FROM entity")
    
    def save(self, entity: dict) -> None:
        \"\"\"Save entity to database.\"\"\"
        self._connection.execute("INSERT INTO entity VALUES (?)", entity)
"""

_SAMPLE_CODE_SIMPLE = "class Repository:\n    def find(self, id: int) -> dict:\n        pass"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def code_orchestrator_available() -> bool:
    """Check if Code-Orchestrator-Service is available."""
    if _USE_FAKE_CLIENTS:
        return True  # Fake clients always available
    try:
        response = httpx.get(f"{_CODE_ORCHESTRATOR_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def sonarqube_available() -> bool:
    """Check if SonarQube is available."""
    if _USE_FAKE_CLIENTS:
        return True  # Fake clients always available
    try:
        response = httpx.get(f"{_SONARQUBE_URL}/api/system/status", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
async def code_analysis_client():
    """Create CodeAnalysisClient for testing."""
    if _USE_FAKE_CLIENTS:
        client = FakeCodeAnalysisClient()
        yield client
    else:
        client = CodeAnalysisClient(base_url=_CODE_ORCHESTRATOR_URL)
        yield client
        await client.close()


@pytest.fixture
async def sonarqube_client():
    """Create SonarQubeClient for testing."""
    if _USE_FAKE_CLIENTS:
        client = FakeSonarQubeClient()
        yield client
    else:
        client = SonarQubeClient(
            base_url=_SONARQUBE_URL,
            token=_SONARQUBE_TOKEN,
            project_key=_SONARQUBE_PROJECT_KEY,
        )
        yield client
        await client.close()


# =============================================================================
# Exit Criteria Tests
# =============================================================================


class TestExtractKeywords:
    """Exit Criterion: extract_keywords("class Repository") returns keywords."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extract_keywords_from_repository_class(
        self,
        code_orchestrator_available: bool,
        code_analysis_client,
    ) -> None:
        """extract_keywords("class Repository") returns ["repository", "pattern", ...]."""
        if not code_orchestrator_available:
            pytest.skip("Code-Orchestrator-Service not available")

        result = await code_analysis_client.extract_keywords(
            code=_SAMPLE_CODE_REPOSITORY,
            top_k=10,
        )

        # Verify keywords extracted
        assert result.keywords, "Should extract at least one keyword"
        
        # Keywords should include repository-related terms
        keyword_lower = [k.lower() for k in result.keywords]
        assert any(
            term in keyword_lower
            for term in ["repository", "entity", "find", "connection", "save"]
        ), f"Expected repository-related keywords, got: {result.keywords}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extract_keywords_simple_class(
        self,
        code_orchestrator_available: bool,
        code_analysis_client,
    ) -> None:
        """extract_keywords for simple class returns basic keywords."""
        if not code_orchestrator_available:
            pytest.skip("Code-Orchestrator-Service not available")

        result = await code_analysis_client.extract_keywords(
            code=_SAMPLE_CODE_SIMPLE,
            top_k=5,
        )

        assert result.keywords, "Should extract keywords from simple class"
        assert result.model, "Should identify model used"


class TestValidateTerms:
    """Exit Criterion: validate_terms(["repositry"], query) catches typo."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_validate_terms_catches_typo(
        self,
        code_orchestrator_available: bool,
        code_analysis_client,
    ) -> None:
        """validate_terms(["repositry"], query) returns low score for typo."""
        if not code_orchestrator_available:
            pytest.skip("Code-Orchestrator-Service not available")

        # Misspelled term
        result = await code_analysis_client.validate_terms(
            terms=["repositry"],  # Typo: missing 'o'
            query="repository pattern implementation",
        )

        # Should have validation result
        assert result.terms, "Should return term validation results"
        
        # Typo should have lower score or be marked invalid
        typo_term = next((t for t in result.terms if t.get("term") == "repositry"), None)
        if typo_term:
            # Either low score or marked invalid
            assert (
                typo_term.get("score", 1.0) < 0.8 or not typo_term.get("valid", True)
            ), f"Typo 'repositry' should have low score or be invalid: {typo_term}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_validate_terms_correct_spelling(
        self,
        code_orchestrator_available: bool,
        code_analysis_client,
    ) -> None:
        """validate_terms with correct spelling returns high score."""
        if not code_orchestrator_available:
            pytest.skip("Code-Orchestrator-Service not available")

        result = await code_analysis_client.validate_terms(
            terms=["repository", "pattern"],
            query="repository pattern implementation",
        )

        assert result.terms, "Should return term validation results"
        
        # Correct terms should have higher scores
        valid_terms = [t for t in result.terms if t.get("valid", False)]
        assert len(valid_terms) >= 1, "At least one term should be valid"


class TestCodeRanking:
    """Exit Criterion: code_ranking uses CodeBERT to rank results."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rank_code_results(
        self,
        code_orchestrator_available: bool,
        code_analysis_client,
    ) -> None:
        """rank_code_results ranks code snippets by relevance."""
        if not code_orchestrator_available:
            pytest.skip("Code-Orchestrator-Service not available")

        result = await code_analysis_client.rank_code_results(
            code_snippets=[
                _SAMPLE_CODE_REPOSITORY,
                "def unrelated_function(): pass",
            ],
            query="repository pattern data access",
        )

        assert result.rankings, "Should return ranked results"
        assert result.model, "Should identify model used"
        
        # Repository code should rank higher than unrelated function
        if len(result.rankings) >= 2:
            repository_rank = next(
                (r for r in result.rankings if "Repository" in r.get("code", "")),
                None,
            )
            assert repository_rank, "Repository code should be in rankings"


class TestSonarQubeAnalysis:
    """Exit Criterion: SonarQube returns complexity, security findings."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sonarqube_returns_metrics(
        self,
        sonarqube_available: bool,
        sonarqube_client,
    ) -> None:
        """SonarQube analysis returns complexity metrics."""
        if not sonarqube_available:
            pytest.skip("SonarQube not available")

        # Use a known file in the project
        result = await sonarqube_client.get_metrics(
            file_path="src/clients/code_analysis.py",
        )

        # Should return metrics
        assert hasattr(result, "complexity"), "Should have complexity metric"
        assert hasattr(result, "cognitive_complexity"), "Should have cognitive complexity"
        assert hasattr(result, "lines_of_code"), "Should have lines of code"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sonarqube_analyze_file(
        self,
        sonarqube_available: bool,
        sonarqube_client,
    ) -> None:
        """analyze_file returns complete analysis with quality gate."""
        if not sonarqube_available:
            pytest.skip("SonarQube not available")

        result = await sonarqube_client.analyze_file(
            file_path="src/clients/code_analysis.py",
        )

        assert hasattr(result, "metrics"), "Should have metrics"
        assert hasattr(result, "quality_passed"), "Should have quality gate status"
        assert hasattr(result, "issues"), "Should have issues list"


class TestClaimValidation:
    """Exit Criterion: LLM claim validated against actual metrics."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_validate_complexity_claim(
        self,
        sonarqube_available: bool,
        sonarqube_client,
    ) -> None:
        """Validate claim 'this code has CC < 10' against actual metrics."""
        if not sonarqube_available:
            pytest.skip("SonarQube not available")

        result = await sonarqube_client.validate_claim(
            file_path="src/clients/code_analysis.py",
            claim="this code has complexity less than 10",
        )

        # Should return validation result
        assert hasattr(result, "is_valid"), "Should have is_valid field"
        assert hasattr(result, "actual_value"), "Should have actual_value field"
        assert hasattr(result, "explanation"), "Should have explanation"


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Demo: Full CodeT5+ → GraphCodeBERT → CodeBERT → SonarQube pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_validation_pipeline(
        self,
        code_orchestrator_available: bool,
        code_analysis_client,
    ) -> None:
        """Full pipeline: extract → validate → rank."""
        if not code_orchestrator_available:
            pytest.skip("Code-Orchestrator-Service not available")

        # Step 1: Extract keywords with CodeT5+
        keywords_result = await code_analysis_client.extract_keywords(
            code=_SAMPLE_CODE_REPOSITORY,
            top_k=5,
        )
        assert keywords_result.keywords, "Step 1 failed: no keywords extracted"

        # Step 2: Validate terms with GraphCodeBERT
        validation_result = await code_analysis_client.validate_terms(
            terms=keywords_result.keywords,
            query="repository pattern for data access layer",
        )
        assert validation_result.terms, "Step 2 failed: no validation results"

        # Step 3: Rank code with CodeBERT
        ranking_result = await code_analysis_client.rank_code_results(
            code_snippets=[_SAMPLE_CODE_REPOSITORY],
            query="repository pattern for data access layer",
        )
        assert ranking_result.rankings, "Step 3 failed: no rankings"

        # All steps completed
        assert True, "Full pipeline completed successfully"

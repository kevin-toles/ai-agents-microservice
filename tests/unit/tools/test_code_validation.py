"""Unit tests for CodeValidationTool.

WBS Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Tasks: KB7.7, KB7.11

TDD Phase: RED - Tests written before implementation.

Acceptance Criteria Verified:
- AC-KB7.6: Tools available to analyze_artifact and validate_against_spec agents
- AC-KB7.7: Validation failures from tools trigger discussion loop retry

Anti-Pattern Focus:
- S1192: String constants at module level
- S3776: Low cognitive complexity via composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


if TYPE_CHECKING:
    pass


# =============================================================================
# Test Constants (S1192 Compliance)
# =============================================================================

_TEST_CODE_SAMPLE = "class Repository:\n    def find(self, id: int) -> dict:\n        pass"
_TEST_QUERY = "repository pattern implementation"
_TEST_FILE_PATH = "src/functions/analyze_artifact.py"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_keyword_result() -> MagicMock:
    """Mock KeywordResult from CodeAnalysisClient."""
    result = MagicMock()
    result.keywords = ["repository", "pattern", "find"]
    result.scores = [0.95, 0.88, 0.75]
    result.model = "codet5p"
    return result


@pytest.fixture
def mock_validation_result() -> MagicMock:
    """Mock TermValidationResult from CodeAnalysisClient."""
    result = MagicMock()
    result.terms = [
        {"term": "repository", "score": 0.92, "valid": True},
        {"term": "pattern", "score": 0.85, "valid": True},
    ]
    result.model = "graphcodebert"
    result.query = _TEST_QUERY
    return result


@pytest.fixture
def mock_ranking_result() -> MagicMock:
    """Mock CodeRankingResult from CodeAnalysisClient."""
    result = MagicMock()
    result.rankings = [
        {"code": _TEST_CODE_SAMPLE, "score": 0.95, "rank": 1},
    ]
    result.model = "codebert"
    result.query = _TEST_QUERY
    return result


@pytest.fixture
def mock_sonar_result() -> MagicMock:
    """Mock SonarQubeAnalysisResult."""
    result = MagicMock()
    result.metrics = MagicMock(
        complexity=8,
        cognitive_complexity=6,
        lines_of_code=150,
        bugs=0,
        vulnerabilities=0,
        code_smells=2,
        coverage=85.5,
    )
    result.issues = []
    result.quality_passed = True
    result.file_path = _TEST_FILE_PATH
    return result


# =============================================================================
# Import Tests (AC-KB7.6)
# =============================================================================


class TestCodeValidationToolImports:
    """Test that CodeValidationTool can be imported."""

    def test_import_code_validation_tool(self) -> None:
        """CodeValidationTool should be importable from src.tools."""
        from src.tools.code_validation import CodeValidationTool

        assert CodeValidationTool is not None

    def test_import_validation_result(self) -> None:
        """CodeValidationResult should be importable."""
        from src.tools.code_validation import CodeValidationResult

        assert CodeValidationResult is not None

    def test_import_validation_step(self) -> None:
        """ValidationStep should be importable."""
        from src.tools.code_validation import ValidationStep

        assert ValidationStep is not None

    def test_import_from_tools_package(self) -> None:
        """CodeValidationTool should be importable from src.tools package."""
        from src.tools import CodeValidationTool

        assert CodeValidationTool is not None


# =============================================================================
# Constructor Tests
# =============================================================================


class TestCodeValidationToolInit:
    """Tests for CodeValidationTool initialization."""

    def test_init_with_clients(self) -> None:
        """Tool initializes with code_analysis_client and sonarqube_client."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = MagicMock()
        mock_sonar_client = MagicMock()

        tool = CodeValidationTool(
            code_analysis_client=mock_code_client,
            sonarqube_client=mock_sonar_client,
        )

        assert tool._code_analysis_client is mock_code_client
        assert tool._sonarqube_client is mock_sonar_client

    def test_init_without_sonarqube(self) -> None:
        """Tool can initialize without SonarQube client."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = MagicMock()

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        assert tool._code_analysis_client is mock_code_client
        assert tool._sonarqube_client is None

    def test_init_with_config(self) -> None:
        """Tool can initialize with configuration."""
        from src.tools.code_validation import CodeValidationTool, CodeValidationConfig

        config = CodeValidationConfig(
            keyword_threshold=0.7,
            validation_threshold=0.8,
            max_complexity=15,
            require_sonarqube=False,
        )

        tool = CodeValidationTool(
            code_analysis_client=MagicMock(),
            config=config,
        )

        assert tool.config.keyword_threshold == 0.7
        assert tool.config.validation_threshold == 0.8


# =============================================================================
# Validate Code Tests (AC-KB7.6)
# =============================================================================


class TestValidateCode:
    """Tests for validate_code() method."""

    @pytest.mark.asyncio
    async def test_validate_code_returns_result(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
    ) -> None:
        """validate_code() returns CodeValidationResult."""
        from src.tools.code_validation import CodeValidationResult, CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool.validate_code(
            code=_TEST_CODE_SAMPLE,
            query=_TEST_QUERY,
        )

        assert isinstance(result, CodeValidationResult)

    @pytest.mark.asyncio
    async def test_validate_code_extracts_keywords(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
    ) -> None:
        """validate_code() calls extract_keywords()."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        mock_code_client.extract_keywords.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_code_validates_terms(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
    ) -> None:
        """validate_code() calls validate_terms() with extracted keywords."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        mock_code_client.validate_terms.assert_called_once()
        call_args = mock_code_client.validate_terms.call_args
        assert call_args.kwargs.get("terms") == mock_keyword_result.keywords

    @pytest.mark.asyncio
    async def test_validate_code_ranks_results(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
    ) -> None:
        """validate_code() calls rank_code_results()."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        mock_code_client.rank_code_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_code_includes_sonarqube(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
        mock_sonar_result: MagicMock,
    ) -> None:
        """validate_code() includes SonarQube analysis when client provided."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        mock_sonar_client = AsyncMock()
        mock_sonar_client.analyze_file.return_value = mock_sonar_result

        tool = CodeValidationTool(
            code_analysis_client=mock_code_client,
            sonarqube_client=mock_sonar_client,
        )

        result = await tool.validate_code(
            code=_TEST_CODE_SAMPLE,
            query=_TEST_QUERY,
            file_path=_TEST_FILE_PATH,
        )

        assert result.sonarqube_result is not None
        mock_sonar_client.analyze_file.assert_called_once()


# =============================================================================
# Validation Steps Tests
# =============================================================================


class TestValidationSteps:
    """Tests for individual validation steps."""

    @pytest.mark.asyncio
    async def test_step_keyword_extraction(
        self, mock_keyword_result: MagicMock
    ) -> None:
        """Keyword extraction step produces correct result."""
        from src.tools.code_validation import CodeValidationTool, ValidationStep

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool._run_step(
            step=ValidationStep.KEYWORD_EXTRACTION,
            code=_TEST_CODE_SAMPLE,
            query=_TEST_QUERY,
        )

        assert result.step == ValidationStep.KEYWORD_EXTRACTION
        assert result.passed is True
        assert result.keywords == mock_keyword_result.keywords

    @pytest.mark.asyncio
    async def test_step_term_validation(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
    ) -> None:
        """Term validation step produces correct result."""
        from src.tools.code_validation import CodeValidationTool, ValidationStep

        mock_code_client = AsyncMock()
        mock_code_client.validate_terms.return_value = mock_validation_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool._run_step(
            step=ValidationStep.TERM_VALIDATION,
            terms=mock_keyword_result.keywords,
            query=_TEST_QUERY,
        )

        assert result.step == ValidationStep.TERM_VALIDATION
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_step_code_ranking(
        self, mock_ranking_result: MagicMock
    ) -> None:
        """Code ranking step produces correct result."""
        from src.tools.code_validation import CodeValidationTool, ValidationStep

        mock_code_client = AsyncMock()
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool._run_step(
            step=ValidationStep.CODE_RANKING,
            code_snippets=[_TEST_CODE_SAMPLE],
            query=_TEST_QUERY,
        )

        assert result.step == ValidationStep.CODE_RANKING
        assert result.passed is True
        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_step_sonarqube_analysis(
        self, mock_sonar_result: MagicMock
    ) -> None:
        """SonarQube analysis step produces correct result."""
        from src.tools.code_validation import CodeValidationTool, ValidationStep

        mock_sonar_client = AsyncMock()
        mock_sonar_client.analyze_file.return_value = mock_sonar_result

        tool = CodeValidationTool(
            code_analysis_client=AsyncMock(),
            sonarqube_client=mock_sonar_client,
        )

        result = await tool._run_step(
            step=ValidationStep.SONARQUBE_ANALYSIS,
            file_path=_TEST_FILE_PATH,
        )

        assert result.step == ValidationStep.SONARQUBE_ANALYSIS
        assert result.passed is True
        assert result.complexity == 8


# =============================================================================
# Validation Failure Tests (AC-KB7.7)
# =============================================================================


class TestValidationFailures:
    """Tests for validation failure handling (AC-KB7.7)."""

    @pytest.mark.asyncio
    async def test_keyword_extraction_failure(self) -> None:
        """Keyword extraction failure returns failed result."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = MagicMock(
            keywords=[],  # No keywords extracted
            scores=[],
            model="codet5p",
        )

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool.validate_code(code="", query=_TEST_QUERY)

        assert result.passed is False
        assert result.failure_reason is not None

    @pytest.mark.asyncio
    async def test_term_validation_failure(self) -> None:
        """Term validation failure returns failed result."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = MagicMock(
            keywords=["repositry"],  # typo
            scores=[0.5],
            model="codet5p",
        )
        mock_code_client.validate_terms.return_value = MagicMock(
            terms=[{"term": "repositry", "score": 0.2, "valid": False}],
            model="graphcodebert",
            query=_TEST_QUERY,
        )
        mock_code_client.rank_code_results.return_value = MagicMock(
            rankings=[],
            model="codebert",
            query=_TEST_QUERY,
        )

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        assert result.passed is False
        assert "term validation" in result.failure_reason.lower() or result.failure_reason

    @pytest.mark.asyncio
    async def test_sonarqube_failure(self, mock_keyword_result: MagicMock) -> None:
        """SonarQube quality gate failure returns failed result."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = MagicMock(
            terms=[{"term": "repository", "score": 0.9, "valid": True}],
            model="graphcodebert",
        )
        mock_code_client.rank_code_results.return_value = MagicMock(
            rankings=[{"code": _TEST_CODE_SAMPLE, "score": 0.9, "rank": 1}],
            model="codebert",
        )

        mock_sonar_client = AsyncMock()
        mock_sonar_client.analyze_file.return_value = MagicMock(
            metrics=MagicMock(
                complexity=25,  # High complexity
                cognitive_complexity=20,
                bugs=3,
                vulnerabilities=2,
            ),
            issues=[],
            quality_passed=False,
        )

        tool = CodeValidationTool(
            code_analysis_client=mock_code_client,
            sonarqube_client=mock_sonar_client,
        )

        result = await tool.validate_code(
            code=_TEST_CODE_SAMPLE,
            query=_TEST_QUERY,
            file_path=_TEST_FILE_PATH,
        )

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_failure_triggers_retry_flag(self) -> None:
        """Validation failure sets should_retry flag."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = MagicMock(
            keywords=[],
            scores=[],
            model="codet5p",
        )

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool.validate_code(code="", query=_TEST_QUERY)

        assert result.should_retry is True


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for full validation pipeline (AC-KB7.6 demo flow)."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
        mock_sonar_result: MagicMock,
    ) -> None:
        """Full pipeline: CodeT5+ → GraphCodeBERT → CodeBERT → SonarQube."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        mock_sonar_client = AsyncMock()
        mock_sonar_client.analyze_file.return_value = mock_sonar_result

        tool = CodeValidationTool(
            code_analysis_client=mock_code_client,
            sonarqube_client=mock_sonar_client,
        )

        result = await tool.validate_code(
            code=_TEST_CODE_SAMPLE,
            query=_TEST_QUERY,
            file_path=_TEST_FILE_PATH,
        )

        # All steps should have been called
        mock_code_client.extract_keywords.assert_called_once()
        mock_code_client.validate_terms.assert_called_once()
        mock_code_client.rank_code_results.assert_called_once()
        mock_sonar_client.analyze_file.assert_called_once()

        # Result should indicate success
        assert result.passed is True
        assert len(result.steps) == 4  # All 4 steps executed

    @pytest.mark.asyncio
    async def test_pipeline_step_order(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
        mock_sonar_result: MagicMock,
    ) -> None:
        """Pipeline executes steps in correct order."""
        from src.tools.code_validation import CodeValidationTool, ValidationStep

        call_order: list[str] = []

        mock_code_client = AsyncMock()

        async def track_keywords(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append("keywords")
            return mock_keyword_result

        async def track_validate(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append("validate")
            return mock_validation_result

        async def track_rank(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append("rank")
            return mock_ranking_result

        mock_code_client.extract_keywords.side_effect = track_keywords
        mock_code_client.validate_terms.side_effect = track_validate
        mock_code_client.rank_code_results.side_effect = track_rank

        mock_sonar_client = AsyncMock()

        async def track_sonar(*args: Any, **kwargs: Any) -> MagicMock:
            call_order.append("sonar")
            return mock_sonar_result

        mock_sonar_client.analyze_file.side_effect = track_sonar

        tool = CodeValidationTool(
            code_analysis_client=mock_code_client,
            sonarqube_client=mock_sonar_client,
        )

        await tool.validate_code(
            code=_TEST_CODE_SAMPLE,
            query=_TEST_QUERY,
            file_path=_TEST_FILE_PATH,
        )

        assert call_order == ["keywords", "validate", "rank", "sonar"]


# =============================================================================
# Integration with Agents Tests (AC-KB7.6)
# =============================================================================


class TestAgentIntegration:
    """Tests for integration with analyze_artifact and validate_against_spec agents."""

    def test_tool_has_agent_interface(self) -> None:
        """CodeValidationTool has interface compatible with agent functions."""
        from src.tools.code_validation import CodeValidationTool

        tool = CodeValidationTool(code_analysis_client=AsyncMock())

        # Should have async validate method
        assert hasattr(tool, "validate_code")
        assert callable(tool.validate_code)

    @pytest.mark.asyncio
    async def test_tool_result_serializable(
        self,
        mock_keyword_result: MagicMock,
        mock_validation_result: MagicMock,
        mock_ranking_result: MagicMock,
    ) -> None:
        """CodeValidationResult is JSON-serializable for agent handoff."""
        from src.tools.code_validation import CodeValidationTool

        mock_code_client = AsyncMock()
        mock_code_client.extract_keywords.return_value = mock_keyword_result
        mock_code_client.validate_terms.return_value = mock_validation_result
        mock_code_client.rank_code_results.return_value = mock_ranking_result

        tool = CodeValidationTool(code_analysis_client=mock_code_client)

        result = await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        # Should be serializable
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "passed" in result_dict
        assert "steps" in result_dict


# =============================================================================
# FakeCodeValidationTool Tests
# =============================================================================


class TestFakeCodeValidationTool:
    """Tests for FakeCodeValidationTool test double."""

    def test_import_fake_tool(self) -> None:
        """FakeCodeValidationTool should be importable."""
        from src.tools.code_validation import FakeCodeValidationTool

        assert FakeCodeValidationTool is not None

    @pytest.mark.asyncio
    async def test_fake_validate_code(self) -> None:
        """FakeCodeValidationTool.validate_code() returns deterministic result."""
        from src.tools.code_validation import FakeCodeValidationTool

        tool = FakeCodeValidationTool()

        result = await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        assert result is not None
        assert hasattr(result, "passed")
        assert hasattr(result, "steps")

    @pytest.mark.asyncio
    async def test_fake_deterministic_output(self) -> None:
        """FakeCodeValidationTool produces deterministic output for same input."""
        from src.tools.code_validation import FakeCodeValidationTool

        tool = FakeCodeValidationTool()

        result1 = await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)
        result2 = await tool.validate_code(code=_TEST_CODE_SAMPLE, query=_TEST_QUERY)

        assert result1.passed == result2.passed

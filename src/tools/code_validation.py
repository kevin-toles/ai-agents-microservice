"""Code Validation Tool for Agent Functions.

WBS Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Tasks: KB7.7, KB7.8, KB7.9, KB7.10

Acceptance Criteria:
- AC-KB7.6: Tools available to analyze_artifact and validate_against_spec agents
- AC-KB7.7: Validation failures from tools trigger discussion loop retry

Reference: KITCHEN_BRIGADE_ARCHITECTURE.md → Agent → Tool/Service Mapping
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Tool Protocols

Anti-Patterns Avoided:
- #12: Connection pooling (reuses clients with pooled connections)
- #42/#43: Proper async/await patterns
- S1192: String constants at module level
- S3776: Cognitive complexity < 15 via composition
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from src.clients.code_analysis import CodeAnalysisProtocol
    from src.clients.sonarqube import SonarQubeProtocol


logger = logging.getLogger(__name__)


# =============================================================================
# Module Constants (S1192 Compliance)
# =============================================================================

_DEFAULT_KEYWORD_THRESHOLD = 0.6
_DEFAULT_VALIDATION_THRESHOLD = 0.7
_DEFAULT_RANKING_THRESHOLD = 0.5
_DEFAULT_MAX_COMPLEXITY = 15

_FAILURE_NO_KEYWORDS = "Keyword extraction failed: no keywords extracted"
_FAILURE_TERM_VALIDATION = "Term validation failed: low validation score"
_FAILURE_CODE_RANKING = "Code ranking failed: low ranking score"
_FAILURE_SONARQUBE = "SonarQube analysis failed: quality gate not passed"


# =============================================================================
# Enums
# =============================================================================


class ValidationStep(str, Enum):
    """Steps in the code validation pipeline.
    
    Pipeline order:
    1. KEYWORD_EXTRACTION - CodeT5+ extracts keywords from code
    2. TERM_VALIDATION - GraphCodeBERT validates terms against query
    3. CODE_RANKING - CodeBERT ranks code against query
    4. SONARQUBE_ANALYSIS - SonarQube checks code quality
    """

    KEYWORD_EXTRACTION = "keyword_extraction"
    TERM_VALIDATION = "term_validation"
    CODE_RANKING = "code_ranking"
    SONARQUBE_ANALYSIS = "sonarqube_analysis"


# =============================================================================
# Configuration
# =============================================================================


class CodeValidationConfig(BaseModel):
    """Configuration for CodeValidationTool.
    
    Attributes:
        keyword_threshold: Minimum score for keyword extraction
        validation_threshold: Minimum score for term validation
        ranking_threshold: Minimum score for code ranking
        max_complexity: Maximum allowed cognitive complexity
        require_sonarqube: Whether SonarQube analysis is required
    """

    model_config = ConfigDict(frozen=True)

    keyword_threshold: float = Field(
        default=_DEFAULT_KEYWORD_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum score for keyword extraction",
    )
    validation_threshold: float = Field(
        default=_DEFAULT_VALIDATION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum score for term validation",
    )
    ranking_threshold: float = Field(
        default=_DEFAULT_RANKING_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum score for code ranking",
    )
    max_complexity: int = Field(
        default=_DEFAULT_MAX_COMPLEXITY,
        ge=1,
        description="Maximum allowed cognitive complexity",
    )
    require_sonarqube: bool = Field(
        default=False,
        description="Whether SonarQube analysis is required",
    )


# =============================================================================
# Result Models
# =============================================================================


class StepResult(BaseModel):
    """Result from a single validation step.
    
    Attributes:
        step: Which validation step produced this result
        passed: Whether the step passed
        score: Confidence/quality score (0-1)
        keywords: Extracted keywords (for KEYWORD_EXTRACTION step)
        complexity: Code complexity (for SONARQUBE_ANALYSIS step)
        details: Additional step-specific details
    """

    model_config = ConfigDict(frozen=True)

    step: ValidationStep = Field(
        description="Which validation step",
    )
    passed: bool = Field(
        default=True,
        description="Whether step passed",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        description="Confidence/quality score",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords",
    )
    complexity: int = Field(
        default=0,
        ge=0,
        description="Code complexity",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details",
    )


class CodeValidationResult(BaseModel):
    """Complete result from code validation pipeline.
    
    Attributes:
        passed: Whether all validation steps passed
        steps: Results from each validation step
        failure_reason: Reason for failure (if any)
        should_retry: Whether the agent should retry
        sonarqube_result: SonarQube analysis result (if performed)
        keywords: Extracted keywords from code
        validation_score: Overall validation score
    """

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(
        default=False,
        description="Whether validation passed",
    )
    steps: list[StepResult] = Field(
        default_factory=list,
        description="Results from each step",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Reason for failure",
    )
    should_retry: bool = Field(
        default=False,
        description="Whether agent should retry",
    )
    sonarqube_result: Any | None = Field(
        default=None,
        description="SonarQube analysis result",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords",
    )
    validation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall validation score",
    )


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class CodeValidationProtocol(Protocol):
    """Protocol for code validation operations.
    
    Defines interface for code validation tools.
    """

    async def validate_code(
        self,
        code: str,
        query: str,
        file_path: str | None = None,
    ) -> CodeValidationResult:
        """Validate code against query and quality standards."""
        ...


# =============================================================================
# CodeValidationTool
# =============================================================================


class CodeValidationTool(CodeValidationProtocol):
    """Tool for validating code using ML models and static analysis.
    
    AC-KB7.6: Tools available to analyze_artifact and validate_against_spec agents
    AC-KB7.7: Validation failures from tools trigger discussion loop retry
    
    Pipeline:
    1. CodeT5+ extracts keywords from code
    2. GraphCodeBERT validates terms against query
    3. CodeBERT ranks code relevance
    4. SonarQube checks code quality (optional)
    
    Example:
        >>> from src.clients.code_analysis import CodeAnalysisClient
        >>> from src.clients.sonarqube import SonarQubeClient
        >>> 
        >>> code_client = CodeAnalysisClient(base_url="http://localhost:8083")
        >>> sonar_client = SonarQubeClient(
        ...     base_url="http://localhost:9000",
        ...     token="your-token",
        ...     project_key="ai-agents",
        ... )
        >>> 
        >>> tool = CodeValidationTool(
        ...     code_analysis_client=code_client,
        ...     sonarqube_client=sonar_client,
        ... )
        >>> 
        >>> result = await tool.validate_code(
        ...     code="class Repository: pass",
        ...     query="repository pattern",
        ...     file_path="src/repository.py",
        ... )
        >>> print(result.passed)
        True
    """

    def __init__(
        self,
        code_analysis_client: CodeAnalysisProtocol,
        sonarqube_client: SonarQubeProtocol | None = None,
        config: CodeValidationConfig | None = None,
    ) -> None:
        """Initialize the code validation tool.
        
        Args:
            code_analysis_client: Client for CodeT5+/GraphCodeBERT/CodeBERT
            sonarqube_client: Optional client for SonarQube analysis
            config: Configuration for validation thresholds
        """
        self._code_analysis_client = code_analysis_client
        self._sonarqube_client = sonarqube_client
        self.config = config or CodeValidationConfig()

    async def validate_code(
        self,
        code: str,
        query: str,
        file_path: str | None = None,
    ) -> CodeValidationResult:
        """Validate code against query and quality standards.
        
        AC-KB7.6: Execute full validation pipeline
        AC-KB7.7: Set should_retry flag on failures
        
        Args:
            code: Source code to validate
            query: Query describing expected functionality
            file_path: Optional file path for SonarQube analysis
            
        Returns:
            CodeValidationResult with validation status and details
        """
        steps: list[StepResult] = []
        keywords: list[str] = []
        passed = True
        failure_reason: str | None = None
        sonarqube_result: Any | None = None

        # Step 1: Keyword Extraction (CodeT5+)
        keyword_result = await self._run_step(
            step=ValidationStep.KEYWORD_EXTRACTION,
            code=code,
            query=query,
        )
        steps.append(keyword_result)
        keywords = keyword_result.keywords

        if not keyword_result.passed:
            passed = False
            failure_reason = _FAILURE_NO_KEYWORDS

        # Step 2: Term Validation (GraphCodeBERT)
        if passed and keywords:
            validation_result = await self._run_step(
                step=ValidationStep.TERM_VALIDATION,
                terms=keywords,
                query=query,
            )
            steps.append(validation_result)

            if not validation_result.passed:
                passed = False
                failure_reason = _FAILURE_TERM_VALIDATION

        # Step 3: Code Ranking (CodeBERT)
        if passed:
            ranking_result = await self._run_step(
                step=ValidationStep.CODE_RANKING,
                code_snippets=[code],
                query=query,
            )
            steps.append(ranking_result)

            if not ranking_result.passed:
                passed = False
                failure_reason = _FAILURE_CODE_RANKING

        # Step 4: SonarQube Analysis (optional)
        if self._sonarqube_client is not None and file_path is not None:
            sonar_step_result = await self._run_step(
                step=ValidationStep.SONARQUBE_ANALYSIS,
                file_path=file_path,
            )
            steps.append(sonar_step_result)
            sonarqube_result = sonar_step_result.details.get("result")

            if not sonar_step_result.passed:
                passed = False
                failure_reason = _FAILURE_SONARQUBE

        # Calculate overall validation score
        validation_score = self._calculate_validation_score(steps)

        return CodeValidationResult(
            passed=passed,
            steps=steps,
            failure_reason=failure_reason,
            should_retry=not passed,  # AC-KB7.7: Trigger retry on failure
            sonarqube_result=sonarqube_result,
            keywords=keywords,
            validation_score=validation_score,
        )

    async def _run_step(
        self,
        step: ValidationStep,
        **kwargs: Any,
    ) -> StepResult:
        """Run a single validation step.
        
        Args:
            step: Which validation step to run
            **kwargs: Step-specific arguments
            
        Returns:
            StepResult with step outcome
        """
        if step == ValidationStep.KEYWORD_EXTRACTION:
            return await self._step_keyword_extraction(
                code=kwargs.get("code", ""),
                query=kwargs.get("query", ""),
            )
        elif step == ValidationStep.TERM_VALIDATION:
            return await self._step_term_validation(
                terms=kwargs.get("terms", []),
                query=kwargs.get("query", ""),
            )
        elif step == ValidationStep.CODE_RANKING:
            return await self._step_code_ranking(
                code_snippets=kwargs.get("code_snippets", []),
                query=kwargs.get("query", ""),
            )
        elif step == ValidationStep.SONARQUBE_ANALYSIS:
            return await self._step_sonarqube_analysis(
                file_path=kwargs.get("file_path", ""),
            )
        else:
            return StepResult(step=step, passed=False)

    async def _step_keyword_extraction(
        self,
        code: str,
        query: str,
    ) -> StepResult:
        """Execute keyword extraction step using CodeT5+.
        
        Args:
            code: Source code
            query: Query context
            
        Returns:
            StepResult with extracted keywords
        """
        try:
            result = await self._code_analysis_client.extract_keywords(
                code=code,
                top_k=10,
            )

            keywords = result.keywords if hasattr(result, "keywords") else []
            scores = result.scores if hasattr(result, "scores") else []
            avg_score = sum(scores) / len(scores) if scores else 0.0

            passed = (
                len(keywords) > 0 and avg_score >= self.config.keyword_threshold
            )

            return StepResult(
                step=ValidationStep.KEYWORD_EXTRACTION,
                passed=passed,
                score=avg_score,
                keywords=keywords,
                details={"model": getattr(result, "model", "codet5p")},
            )
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return StepResult(
                step=ValidationStep.KEYWORD_EXTRACTION,
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

    async def _step_term_validation(
        self,
        terms: list[str],
        query: str,
    ) -> StepResult:
        """Execute term validation step using GraphCodeBERT.
        
        Args:
            terms: Terms to validate
            query: Query context
            
        Returns:
            StepResult with validation status
        """
        try:
            result = await self._code_analysis_client.validate_terms(
                terms=terms,
                query=query,
            )

            terms_data = result.terms if hasattr(result, "terms") else []
            valid_terms = [t for t in terms_data if t.get("valid", False)]
            avg_score = (
                sum(t.get("score", 0) for t in terms_data) / len(terms_data)
                if terms_data
                else 0.0
            )

            passed = (
                len(valid_terms) > 0 and avg_score >= self.config.validation_threshold
            )

            return StepResult(
                step=ValidationStep.TERM_VALIDATION,
                passed=passed,
                score=avg_score,
                details={
                    "valid_terms": len(valid_terms),
                    "total_terms": len(terms_data),
                    "model": getattr(result, "model", "graphcodebert"),
                },
            )
        except Exception as e:
            logger.warning(f"Term validation failed: {e}")
            return StepResult(
                step=ValidationStep.TERM_VALIDATION,
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

    async def _step_code_ranking(
        self,
        code_snippets: list[str],
        query: str,
    ) -> StepResult:
        """Execute code ranking step using CodeBERT.
        
        Args:
            code_snippets: Code samples to rank
            query: Query context
            
        Returns:
            StepResult with ranking score
        """
        try:
            result = await self._code_analysis_client.rank_code_results(
                code_snippets=code_snippets,
                query=query,
            )

            rankings = result.rankings if hasattr(result, "rankings") else []
            top_score = rankings[0].get("score", 0.0) if rankings else 0.0

            passed = top_score >= self.config.ranking_threshold

            return StepResult(
                step=ValidationStep.CODE_RANKING,
                passed=passed,
                score=top_score,
                details={
                    "rankings": rankings,
                    "model": getattr(result, "model", "codebert"),
                },
            )
        except Exception as e:
            logger.warning(f"Code ranking failed: {e}")
            return StepResult(
                step=ValidationStep.CODE_RANKING,
                passed=False,
                score=0.0,
                details={"error": str(e)},
            )

    async def _step_sonarqube_analysis(
        self,
        file_path: str,
    ) -> StepResult:
        """Execute SonarQube analysis step.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            StepResult with quality metrics
        """
        if self._sonarqube_client is None:
            return StepResult(
                step=ValidationStep.SONARQUBE_ANALYSIS,
                passed=True,
                details={"skipped": True},
            )

        try:
            result = await self._sonarqube_client.analyze_file(file_path=file_path)

            complexity = (
                result.metrics.complexity
                if hasattr(result, "metrics") and hasattr(result.metrics, "complexity")
                else 0
            )
            quality_passed = (
                result.quality_passed if hasattr(result, "quality_passed") else True
            )

            passed = quality_passed and complexity <= self.config.max_complexity

            return StepResult(
                step=ValidationStep.SONARQUBE_ANALYSIS,
                passed=passed,
                complexity=complexity,
                details={"result": result},
            )
        except Exception as e:
            logger.warning(f"SonarQube analysis failed: {e}")
            return StepResult(
                step=ValidationStep.SONARQUBE_ANALYSIS,
                passed=False,
                details={"error": str(e)},
            )

    def _calculate_validation_score(
        self,
        steps: list[StepResult],
    ) -> float:
        """Calculate overall validation score from step results.
        
        Args:
            steps: List of step results
            
        Returns:
            Average score across all steps
        """
        if not steps:
            return 0.0

        scores = [step.score for step in steps if step.score > 0]
        return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# FakeCodeValidationTool (Test Double)
# =============================================================================


class FakeCodeValidationTool(CodeValidationProtocol):
    """Fake implementation for testing (AC-KB7.11).
    
    Provides deterministic results for unit testing without
    requiring actual ML model or SonarQube connections.
    
    Example:
        >>> tool = FakeCodeValidationTool()
        >>> result = await tool.validate_code(
        ...     code="class Test: pass",
        ...     query="test class",
        ... )
        >>> assert result.passed is True
    """

    def __init__(
        self,
        default_passed: bool = True,
        default_keywords: list[str] | None = None,
        default_score: float = 0.85,
    ) -> None:
        """Initialize the fake tool.
        
        Args:
            default_passed: Default pass/fail status
            default_keywords: Default keywords to return
            default_score: Default validation score
        """
        self._default_passed = default_passed
        self._default_keywords = default_keywords or ["class", "method", "pattern"]
        self._default_score = default_score

    async def validate_code(
        self,
        code: str,
        query: str,
        file_path: str | None = None,
    ) -> CodeValidationResult:
        """Return deterministic validation result.
        
        Args:
            code: Source code (used for deterministic hashing)
            query: Query (used for deterministic hashing)
            file_path: Optional file path
            
        Returns:
            Deterministic CodeValidationResult
        """
        # Generate deterministic result based on input hash
        input_hash = hashlib.md5(f"{code}{query}".encode()).hexdigest()
        deterministic_passed = self._default_passed

        # Create step results
        steps = [
            StepResult(
                step=ValidationStep.KEYWORD_EXTRACTION,
                passed=True,
                score=self._default_score,
                keywords=self._default_keywords,
            ),
            StepResult(
                step=ValidationStep.TERM_VALIDATION,
                passed=True,
                score=self._default_score,
            ),
            StepResult(
                step=ValidationStep.CODE_RANKING,
                passed=True,
                score=self._default_score,
            ),
        ]

        if file_path:
            steps.append(
                StepResult(
                    step=ValidationStep.SONARQUBE_ANALYSIS,
                    passed=True,
                    complexity=8,
                )
            )

        return CodeValidationResult(
            passed=deterministic_passed,
            steps=steps,
            failure_reason=None if deterministic_passed else "Fake failure",
            should_retry=not deterministic_passed,
            keywords=self._default_keywords,
            validation_score=self._default_score,
        )

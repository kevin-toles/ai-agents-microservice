"""SonarQube Client for Code Quality Analysis.

WBS Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Tasks: KB7.5, KB7.6

Acceptance Criteria:
- AC-KB7.5: sonarqube_analyze tool integrated for quality metrics

Reference: KITCHEN_BRIGADE_ARCHITECTURE.md → Agent → Tool/Service Mapping

Anti-Patterns Avoided:
- #12: Connection pooling (single httpx.AsyncClient)
- #42/#43: Proper async/await patterns
- S1192: String constants at module level
- S3776: Cognitive complexity < 15 via helper methods
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Module Constants (S1192 Compliance)
# =============================================================================

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 3
_RETRY_BACKOFF_FACTOR = 0.5

_ENDPOINT_MEASURES = "/api/measures/component"
_ENDPOINT_ISSUES = "/api/issues/search"

_METRIC_KEYS = "complexity,cognitive_complexity,ncloc,bugs,vulnerabilities,code_smells,coverage"

_ERROR_SERVICE_UNAVAILABLE = "SonarQube service unavailable after {retries} retries"


# =============================================================================
# Configuration
# =============================================================================


class SonarQubeConfig(BaseModel):
    """Configuration for SonarQubeClient.
    
    Attributes:
        base_url: Base URL for SonarQube server
        token: Authentication token
        project_key: Default project key
        timeout: Request timeout in seconds
    """

    model_config = ConfigDict(frozen=True)

    base_url: str = Field(
        default="http://localhost:9000",
        description="Base URL for SonarQube server",
    )
    token: str = Field(
        default="",
        description="SonarQube authentication token",
    )
    project_key: str = Field(
        default="",
        description="Default project key",
    )
    timeout: float = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1.0,
        description="Request timeout in seconds",
    )


# =============================================================================
# Result Models
# =============================================================================


class SonarQubeMetrics(BaseModel):
    """Quality metrics from SonarQube.
    
    Attributes:
        complexity: Cyclomatic complexity
        cognitive_complexity: Cognitive complexity
        lines_of_code: Number of lines of code
        bugs: Number of bugs
        vulnerabilities: Number of security vulnerabilities
        code_smells: Number of code smells
        coverage: Test coverage percentage
    """

    model_config = ConfigDict(frozen=True)

    complexity: int = Field(
        default=0,
        ge=0,
        description="Cyclomatic complexity",
    )
    cognitive_complexity: int = Field(
        default=0,
        ge=0,
        description="Cognitive complexity",
    )
    lines_of_code: int = Field(
        default=0,
        ge=0,
        description="Number of lines of code",
    )
    bugs: int = Field(
        default=0,
        ge=0,
        description="Number of bugs",
    )
    vulnerabilities: int = Field(
        default=0,
        ge=0,
        description="Number of security vulnerabilities",
    )
    code_smells: int = Field(
        default=0,
        ge=0,
        description="Number of code smells",
    )
    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Test coverage percentage",
    )


class SonarQubeIssue(BaseModel):
    """A code issue from SonarQube.
    
    Attributes:
        rule: Rule identifier (e.g., "python:S1192")
        message: Issue description
        severity: Issue severity (INFO, MINOR, MAJOR, CRITICAL, BLOCKER)
        type: Issue type (BUG, VULNERABILITY, CODE_SMELL)
        line: Line number where issue occurs
    """

    model_config = ConfigDict(frozen=True)

    rule: str = Field(
        default="",
        description="Rule identifier",
    )
    message: str = Field(
        default="",
        description="Issue description",
    )
    severity: str = Field(
        default="INFO",
        description="Issue severity",
    )
    type: str = Field(
        default="CODE_SMELL",
        description="Issue type",
    )
    line: int = Field(
        default=0,
        ge=0,
        description="Line number",
    )


class SonarQubeAnalysisResult(BaseModel):
    """Complete analysis result from SonarQube.
    
    Attributes:
        file_path: Path to analyzed file
        project_key: SonarQube project key
        metrics: Quality metrics
        issues: List of code issues
        quality_passed: Whether quality gate passed
    """

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(
        default="",
        description="Path to analyzed file",
    )
    project_key: str = Field(
        default="",
        description="SonarQube project key",
    )
    metrics: SonarQubeMetrics = Field(
        default_factory=SonarQubeMetrics,
        description="Quality metrics",
    )
    issues: list[SonarQubeIssue] = Field(
        default_factory=list,
        description="List of code issues",
    )
    quality_passed: bool = Field(
        default=True,
        description="Whether quality gate passed",
    )


class ClaimValidationResult(BaseModel):
    """Result of claim validation against metrics.
    
    Attributes:
        is_valid: Whether the claim is valid
        actual_value: Actual metric value
        claim: Original claim
        explanation: Why claim is valid/invalid
    """

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(
        default=False,
        description="Whether the claim is valid",
    )
    actual_value: float | int = Field(
        default=0,
        description="Actual metric value",
    )
    claim: str = Field(
        default="",
        description="Original claim",
    )
    explanation: str = Field(
        default="",
        description="Validation explanation",
    )


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class SonarQubeProtocol(Protocol):
    """Protocol for SonarQube operations.
    
    Defines interface for code quality analysis.
    """

    async def analyze_file(
        self, file_path: str
    ) -> SonarQubeAnalysisResult:
        """Analyze a file and return metrics and issues."""
        ...

    async def get_metrics(
        self, file_path: str
    ) -> SonarQubeMetrics:
        """Get quality metrics for a file."""
        ...

    async def get_issues(
        self,
        file_path: str,
        severities: list[str] | None = None,
        types: list[str] | None = None,
    ) -> list[SonarQubeIssue]:
        """Get issues for a file."""
        ...

    async def validate_claim(
        self, file_path: str, claim: str
    ) -> ClaimValidationResult:
        """Validate a claim against actual metrics."""
        ...

    async def close(self) -> None:
        """Close the client and release resources."""
        ...


# =============================================================================
# SonarQubeClient
# =============================================================================


class SonarQubeClient(SonarQubeProtocol):
    """HTTP client for SonarQube API.
    
    AC-KB7.5: sonarqube_analyze tool integrated for quality metrics
    
    Provides async methods for:
    - File analysis with metrics and issues
    - Quality metrics retrieval
    - Issue searching and filtering
    - Claim validation against metrics
    
    Example:
        >>> client = SonarQubeClient(
        ...     base_url="http://localhost:9000",
        ...     token="your-token",
        ...     project_key="ai-agents",
        ... )
        >>> result = await client.analyze_file("src/main.py")
        >>> print(result.metrics.complexity)
        12
        >>> await client.close()
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        project_key: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the SonarQube client.
        
        Args:
            base_url: Base URL for SonarQube server
            token: Authentication token
            project_key: Default project key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.token = token
        self.project_key = project_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_config(cls, config: SonarQubeConfig) -> "SonarQubeClient":
        """Create client from configuration.
        
        Args:
            config: SonarQubeConfig instance
            
        Returns:
            Configured SonarQubeClient
        """
        return cls(
            base_url=config.base_url,
            token=config.token,
            project_key=config.project_key,
            timeout=config.timeout,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialization).
        
        Returns:
            Shared httpx.AsyncClient instance
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.token}"},
            )
            await asyncio.sleep(0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Parsed JSON response
        """
        client = await self._get_client()
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    # =========================================================================
    # API Methods (AC-KB7.5)
    # =========================================================================

    async def analyze_file(
        self, file_path: str
    ) -> SonarQubeAnalysisResult:
        """Analyze a file and return metrics and issues.
        
        AC-KB7.5: sonarqube_analyze tool integrated for quality metrics
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            SonarQubeAnalysisResult with metrics and issues
        """
        metrics = await self.get_metrics(file_path)
        issues = await self.get_issues(file_path)

        # Determine quality gate pass/fail
        quality_passed = self._evaluate_quality_gate(metrics, issues)

        return SonarQubeAnalysisResult(
            file_path=file_path,
            project_key=self.project_key,
            metrics=metrics,
            issues=issues,
            quality_passed=quality_passed,
        )

    async def get_metrics(
        self, file_path: str
    ) -> SonarQubeMetrics:
        """Get quality metrics for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SonarQubeMetrics with quality data
        """
        component_key = f"{self.project_key}:{file_path}"

        data = await self._request_get(
            endpoint=_ENDPOINT_MEASURES,
            params={
                "component": component_key,
                "metricKeys": _METRIC_KEYS,
            },
        )

        return self._parse_metrics(data)

    async def get_issues(
        self,
        file_path: str,
        severities: list[str] | None = None,
        types: list[str] | None = None,
    ) -> list[SonarQubeIssue]:
        """Get issues for a file.
        
        Args:
            file_path: Path to file
            severities: Filter by severities
            types: Filter by issue types
            
        Returns:
            List of SonarQubeIssue
        """
        component_key = f"{self.project_key}:{file_path}"

        params: dict[str, Any] = {"componentKeys": component_key}
        if severities:
            params["severities"] = ",".join(severities)
        if types:
            params["types"] = ",".join(types)

        data = await self._request_get(
            endpoint=_ENDPOINT_ISSUES,
            params=params,
        )

        return self._parse_issues(data)

    async def validate_claim(
        self, file_path: str, claim: str
    ) -> ClaimValidationResult:
        """Validate a claim against actual metrics.
        
        Args:
            file_path: Path to file
            claim: Claim to validate (e.g., "complexity < 10")
            
        Returns:
            ClaimValidationResult with validation outcome
        """
        metrics = await self.get_metrics(file_path)
        return self._evaluate_claim(claim, metrics)

    async def _get_metrics(
        self, file_path: str
    ) -> dict[str, Any]:
        """Internal method to get metrics as dict.
        
        Args:
            file_path: Path to file
            
        Returns:
            Metrics dictionary
        """
        metrics = await self.get_metrics(file_path)
        return metrics.model_dump()

    async def _get_issues(
        self, file_path: str
    ) -> list[dict[str, Any]]:
        """Internal method to get issues as list of dicts.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of issue dictionaries
        """
        issues = await self.get_issues(file_path)
        return [issue.model_dump() for issue in issues]

    # =========================================================================
    # Parsing Methods
    # =========================================================================

    def _parse_metrics(self, data: dict[str, Any]) -> SonarQubeMetrics:
        """Parse metrics from API response.
        
        Args:
            data: API response data
            
        Returns:
            SonarQubeMetrics
        """
        measures = {}
        component = data.get("component", {})

        for measure in component.get("measures", []):
            metric = measure.get("metric", "")
            value = measure.get("value", "0")
            measures[metric] = value

        return SonarQubeMetrics(
            complexity=int(measures.get("complexity", 0)),
            cognitive_complexity=int(measures.get("cognitive_complexity", 0)),
            lines_of_code=int(measures.get("ncloc", 0)),
            bugs=int(measures.get("bugs", 0)),
            vulnerabilities=int(measures.get("vulnerabilities", 0)),
            code_smells=int(measures.get("code_smells", 0)),
            coverage=float(measures.get("coverage", 0.0)),
        )

    def _parse_issues(self, data: dict[str, Any]) -> list[SonarQubeIssue]:
        """Parse issues from API response.
        
        Args:
            data: API response data
            
        Returns:
            List of SonarQubeIssue
        """
        issues = []
        for issue_data in data.get("issues", []):
            issues.append(
                SonarQubeIssue(
                    rule=issue_data.get("rule", ""),
                    message=issue_data.get("message", ""),
                    severity=issue_data.get("severity", "INFO"),
                    type=issue_data.get("type", "CODE_SMELL"),
                    line=issue_data.get("line", 0),
                )
            )
        return issues

    def _evaluate_quality_gate(
        self,
        metrics: SonarQubeMetrics,
        issues: list[SonarQubeIssue],
    ) -> bool:
        """Evaluate quality gate based on metrics and issues.
        
        Args:
            metrics: Quality metrics
            issues: List of issues
            
        Returns:
            True if quality gate passes
        """
        # Fail if there are critical issues
        critical_issues = sum(
            1 for issue in issues
            if issue.severity in ("CRITICAL", "BLOCKER")
        )
        if critical_issues > 0:
            return False

        # Fail if complexity too high
        if metrics.cognitive_complexity > 15:
            return False

        # Fail if vulnerabilities exist
        if metrics.vulnerabilities > 0:
            return False

        return True

    def _evaluate_claim(
        self,
        claim: str,
        metrics: SonarQubeMetrics,
    ) -> ClaimValidationResult:
        """Evaluate a claim against metrics.
        
        Args:
            claim: Claim string (e.g., "complexity < 10")
            metrics: Actual metrics
            
        Returns:
            ClaimValidationResult
        """
        # Parse claim
        match = re.match(
            r"(\w+)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)",
            claim.strip(),
        )

        if not match:
            return ClaimValidationResult(
                is_valid=False,
                actual_value=0,
                claim=claim,
                explanation=f"Invalid claim format: {claim}",
            )

        metric_name = match.group(1).lower()
        operator = match.group(2)
        threshold = float(match.group(3))

        # Get actual value
        actual_value = self._get_metric_value(metric_name, metrics)

        # Evaluate
        is_valid = self._compare_values(actual_value, operator, threshold)

        return ClaimValidationResult(
            is_valid=is_valid,
            actual_value=actual_value,
            claim=claim,
            explanation=f"{metric_name}={actual_value} {operator} {threshold} is {is_valid}",
        )

    def _get_metric_value(
        self,
        metric_name: str,
        metrics: SonarQubeMetrics,
    ) -> float | int:
        """Get metric value by name.
        
        Args:
            metric_name: Metric name
            metrics: Metrics object
            
        Returns:
            Metric value
        """
        metric_map = {
            "complexity": metrics.complexity,
            "cognitive_complexity": metrics.cognitive_complexity,
            "lines_of_code": metrics.lines_of_code,
            "bugs": metrics.bugs,
            "vulnerabilities": metrics.vulnerabilities,
            "code_smells": metrics.code_smells,
            "coverage": metrics.coverage,
        }
        return metric_map.get(metric_name, 0)

    def _compare_values(
        self,
        actual: float | int,
        operator: str,
        threshold: float,
    ) -> bool:
        """Compare actual value against threshold.
        
        Args:
            actual: Actual value
            operator: Comparison operator
            threshold: Threshold value
            
        Returns:
            Comparison result
        """
        operations = {
            "<": lambda a, t: a < t,
            "<=": lambda a, t: a <= t,
            ">": lambda a, t: a > t,
            ">=": lambda a, t: a >= t,
            "==": lambda a, t: a == t,
            "!=": lambda a, t: a != t,
        }
        return operations.get(operator, lambda a, t: False)(actual, threshold)


# =============================================================================
# FakeSonarQubeClient
# =============================================================================


class FakeSonarQubeClient(SonarQubeProtocol):
    """Fake SonarQube client for testing.
    
    Produces deterministic results based on input hashes.
    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md
    """

    async def analyze_file(
        self, file_path: str
    ) -> SonarQubeAnalysisResult:
        """Analyze a file with deterministic results.
        
        Args:
            file_path: Path to file
            
        Returns:
            Deterministic SonarQubeAnalysisResult
        """
        metrics = await self.get_metrics(file_path)
        issues = await self.get_issues(file_path)

        # Deterministic quality gate
        quality_passed = metrics.cognitive_complexity <= 15

        return SonarQubeAnalysisResult(
            file_path=file_path,
            project_key="test-project",
            metrics=metrics,
            issues=issues,
            quality_passed=quality_passed,
        )

    async def get_metrics(
        self, file_path: str
    ) -> SonarQubeMetrics:
        """Get deterministic metrics for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Deterministic SonarQubeMetrics
        """
        # Hash-based deterministic values
        hash_val = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)

        return SonarQubeMetrics(
            complexity=(hash_val % 20) + 1,
            cognitive_complexity=(hash_val % 15) + 1,
            lines_of_code=(hash_val % 500) + 50,
            bugs=hash_val % 3,
            vulnerabilities=hash_val % 2,
            code_smells=(hash_val % 10),
            coverage=50.0 + (hash_val % 50),
        )

    async def get_issues(
        self,
        file_path: str,
        severities: list[str] | None = None,
        types: list[str] | None = None,
    ) -> list[SonarQubeIssue]:
        """Get deterministic issues for a file.
        
        Args:
            file_path: Path to file
            severities: Filter by severities
            types: Filter by types
            
        Returns:
            Deterministic list of SonarQubeIssue
        """
        hash_val = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)
        issue_count = hash_val % 5

        issues = []
        for i in range(issue_count):
            severity = ["INFO", "MINOR", "MAJOR"][i % 3]
            issue_type = ["CODE_SMELL", "BUG", "VULNERABILITY"][i % 3]

            if severities and severity not in severities:
                continue
            if types and issue_type not in types:
                continue

            issues.append(
                SonarQubeIssue(
                    rule=f"python:S{1000 + i}",
                    message=f"Test issue {i + 1} for {file_path}",
                    severity=severity,
                    type=issue_type,
                    line=(i + 1) * 10,
                )
            )

        return issues

    async def validate_claim(
        self, file_path: str, claim: str
    ) -> ClaimValidationResult:
        """Validate a claim with deterministic results.
        
        Args:
            file_path: Path to file
            claim: Claim to validate
            
        Returns:
            Deterministic ClaimValidationResult
        """
        metrics = await self.get_metrics(file_path)

        # Simple parsing
        match = re.match(
            r"(\w+)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)",
            claim.strip(),
        )

        if not match:
            return ClaimValidationResult(
                is_valid=False,
                actual_value=0,
                claim=claim,
                explanation="Invalid claim format",
            )

        metric_name = match.group(1).lower()
        operator = match.group(2)
        threshold = float(match.group(3))

        # Get actual value
        actual = getattr(metrics, metric_name, 0)

        # Simple evaluation
        is_valid = False
        if operator == "<":
            is_valid = actual < threshold
        elif operator == "<=":
            is_valid = actual <= threshold
        elif operator == ">":
            is_valid = actual > threshold
        elif operator == ">=":
            is_valid = actual >= threshold
        elif operator == "==":
            is_valid = actual == threshold

        return ClaimValidationResult(
            is_valid=is_valid,
            actual_value=actual,
            claim=claim,
            explanation=f"{metric_name}={actual} {operator} {threshold}",
        )

    async def close(self) -> None:
        """No-op for fake client."""
        pass


__all__ = [
    "ClaimValidationResult",
    "FakeSonarQubeClient",
    "SonarQubeAnalysisResult",
    "SonarQubeClient",
    "SonarQubeConfig",
    "SonarQubeIssue",
    "SonarQubeMetrics",
    "SonarQubeProtocol",
]

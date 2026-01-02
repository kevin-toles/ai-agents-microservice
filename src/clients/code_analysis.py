"""Code Analysis Client for Code-Orchestrator Tool Integration.

WBS Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Tasks: KB7.1-KB7.4

Acceptance Criteria:
- AC-KB7.1: CodeOrchestratorClient wraps Code-Orchestrator:8083 API
- AC-KB7.2: keyword_extraction tool uses CodeT5+ via Code-Orchestrator
- AC-KB7.3: term_validation tool uses GraphCodeBERT via Code-Orchestrator
- AC-KB7.4: code_ranking tool uses CodeBERT via Code-Orchestrator

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
from dataclasses import dataclass, field
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

_ENDPOINT_KEYWORDS_EXTRACT = "/api/v1/codet5/keywords"
_ENDPOINT_TERM_VALIDATE = "/api/v1/graphcodebert/validate"
_ENDPOINT_CODE_RANK = "/api/v1/codebert/rank"
_ENDPOINT_CODE_SIMILARITY = "/api/v1/codebert/similarity"

_MODEL_CODET5P = "codet5p"
_MODEL_GRAPHCODEBERT = "graphcodebert"
_MODEL_CODEBERT = "codebert"

_ERROR_SERVICE_UNAVAILABLE = "Code-Orchestrator service unavailable after {retries} retries"


# =============================================================================
# Configuration
# =============================================================================


class CodeAnalysisConfig(BaseModel):
    """Configuration for CodeAnalysisClient.
    
    Attributes:
        base_url: Base URL for Code-Orchestrator service
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """

    model_config = ConfigDict(frozen=True)

    base_url: str = Field(
        default="http://localhost:8083",
        description="Base URL for Code-Orchestrator service",
    )
    timeout: float = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1.0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=_DEFAULT_MAX_RETRIES,
        ge=0,
        description="Maximum retry attempts",
    )


# =============================================================================
# Result Models
# =============================================================================


class KeywordResult(BaseModel):
    """Result from keyword extraction.
    
    AC-KB7.2: keyword_extraction tool uses CodeT5+ via Code-Orchestrator
    """

    model_config = ConfigDict(frozen=True)

    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords from code",
    )
    scores: list[float] = Field(
        default_factory=list,
        description="Confidence scores for each keyword",
    )
    model: str = Field(
        default=_MODEL_CODET5P,
        description="Model used for extraction",
    )


class TermValidationResult(BaseModel):
    """Result from term validation.
    
    AC-KB7.3: term_validation tool uses GraphCodeBERT via Code-Orchestrator
    """

    model_config = ConfigDict(frozen=True)

    terms: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Validated terms with scores and validity flags",
    )
    model: str = Field(
        default=_MODEL_GRAPHCODEBERT,
        description="Model used for validation",
    )
    query: str = Field(
        default="",
        description="Original query used for validation context",
    )


class CodeRankingResult(BaseModel):
    """Result from code ranking.
    
    AC-KB7.4: code_ranking tool uses CodeBERT via Code-Orchestrator
    """

    model_config = ConfigDict(frozen=True)

    rankings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked code snippets with scores",
    )
    model: str = Field(
        default=_MODEL_CODEBERT,
        description="Model used for ranking",
    )
    query: str = Field(
        default="",
        description="Query used for ranking context",
    )


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class CodeAnalysisProtocol(Protocol):
    """Protocol for code analysis operations.
    
    Defines interface for CodeT5+, GraphCodeBERT, and CodeBERT operations.
    """

    async def extract_keywords(
        self, code: str, top_k: int = 5
    ) -> KeywordResult:
        """Extract keywords from code using CodeT5+."""
        ...

    async def validate_terms(
        self,
        terms: list[str],
        query: str,
        threshold: float = 0.5,
    ) -> TermValidationResult:
        """Validate terms against query using GraphCodeBERT."""
        ...

    async def rank_code_results(
        self,
        code_snippets: list[str],
        query: str,
        top_k: int | None = None,
    ) -> CodeRankingResult:
        """Rank code snippets by relevance using CodeBERT."""
        ...

    async def calculate_similarity(
        self, code_a: str, code_b: str
    ) -> float:
        """Calculate similarity between two code snippets."""
        ...

    async def close(self) -> None:
        """Close the client and release resources."""
        ...


# =============================================================================
# CodeAnalysisClient
# =============================================================================


class CodeAnalysisClient(CodeAnalysisProtocol):
    """HTTP client for Code-Orchestrator code analysis tools.
    
    AC-KB7.1: CodeOrchestratorClient wraps Code-Orchestrator:8083 API
    
    Provides async methods for:
    - CodeT5+ keyword extraction (AC-KB7.2)
    - GraphCodeBERT term validation (AC-KB7.3)
    - CodeBERT code ranking (AC-KB7.4)
    
    Uses connection pooling (single httpx.AsyncClient) and implements
    retry logic for transient errors.
    
    Example:
        >>> client = CodeAnalysisClient(base_url="http://localhost:8083")
        >>> result = await client.extract_keywords(code="class Repository: pass")
        >>> print(result.keywords)
        ["repository", "class"]
        >>> await client.close()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the Code Analysis client.
        
        Args:
            base_url: Base URL for Code-Orchestrator service
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_config(cls, config: CodeAnalysisConfig) -> "CodeAnalysisClient":
        """Create client from configuration.
        
        Args:
            config: CodeAnalysisConfig instance
            
        Returns:
            Configured CodeAnalysisClient
        """
        return cls(
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialization).
        
        Returns:
            Shared httpx.AsyncClient instance (connection pooling)
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            await asyncio.sleep(0)  # Yield to event loop on first init
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            json: Request body as JSON
            
        Returns:
            Parsed JSON response
            
        Raises:
            httpx.HTTPStatusError: On non-retryable errors
            RuntimeError: After exhausting retries
        """
        client = await self._get_client()
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._execute_request(client, method, endpoint, json)
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                last_exception = e

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e

            if attempt < self.max_retries and last_exception:
                await self._backoff(attempt)
                logger.warning(
                    "Retrying %s %s (attempt %d/%d): %s",
                    method,
                    endpoint,
                    attempt + 1,
                    self.max_retries,
                    str(last_exception),
                )

        raise RuntimeError(
            _ERROR_SERVICE_UNAVAILABLE.format(retries=self.max_retries)
        ) from last_exception

    async def _execute_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None,
    ) -> httpx.Response:
        """Execute a single HTTP request.
        
        Args:
            client: HTTP client instance
            method: HTTP method
            endpoint: API endpoint
            json: Request body
            
        Returns:
            HTTP response
        """
        if method.upper() == "POST":
            return await client.post(endpoint, json=json)
        return await client.get(endpoint)

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff between retries.
        
        Args:
            attempt: Current attempt number (0-indexed)
        """
        delay = _RETRY_BACKOFF_FACTOR * (2 ** attempt)
        await asyncio.sleep(delay)

    # =========================================================================
    # API Methods (AC-KB7.2, AC-KB7.3, AC-KB7.4)
    # =========================================================================

    async def extract_keywords(
        self, code: str, top_k: int = 5
    ) -> KeywordResult:
        """Extract keywords from code using CodeT5+.
        
        AC-KB7.2: keyword_extraction tool uses CodeT5+ via Code-Orchestrator
        
        Args:
            code: Source code to analyze
            top_k: Number of keywords to extract
            
        Returns:
            KeywordResult with keywords and confidence scores
        """
        if not code:
            return KeywordResult(keywords=[], scores=[], model=_MODEL_CODET5P)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_KEYWORDS_EXTRACT,
            json={"code": code, "top_k": top_k},
        )

        return KeywordResult(
            keywords=data.get("keywords", []),
            scores=data.get("scores", []),
            model=data.get("model", _MODEL_CODET5P),
        )

    async def validate_terms(
        self,
        terms: list[str],
        query: str,
        threshold: float = 0.5,
    ) -> TermValidationResult:
        """Validate terms against query using GraphCodeBERT.
        
        AC-KB7.3: term_validation tool uses GraphCodeBERT via Code-Orchestrator
        
        Args:
            terms: List of terms to validate
            query: Query context for validation
            threshold: Minimum score for term to be valid
            
        Returns:
            TermValidationResult with validation results
        """
        if not terms:
            return TermValidationResult(
                terms=[],
                model=_MODEL_GRAPHCODEBERT,
                query=query,
            )

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_TERM_VALIDATE,
            json={"terms": terms, "query": query, "threshold": threshold},
        )

        return TermValidationResult(
            terms=data.get("terms", []),
            model=data.get("model", _MODEL_GRAPHCODEBERT),
            query=data.get("query", query),
        )

    async def rank_code_results(
        self,
        code_snippets: list[str],
        query: str,
        top_k: int | None = None,
    ) -> CodeRankingResult:
        """Rank code snippets by relevance using CodeBERT.
        
        AC-KB7.4: code_ranking tool uses CodeBERT via Code-Orchestrator
        
        Args:
            code_snippets: List of code snippets to rank
            query: Query to rank against
            top_k: Maximum number of results to return
            
        Returns:
            CodeRankingResult with ranked code snippets
        """
        if not code_snippets:
            return CodeRankingResult(
                rankings=[],
                model=_MODEL_CODEBERT,
                query=query,
            )

        payload: dict[str, Any] = {"codes": code_snippets, "query": query}
        if top_k is not None:
            payload["top_k"] = top_k

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODE_RANK,
            json=payload,
        )

        return CodeRankingResult(
            rankings=data.get("rankings", []),
            model=data.get("model", _MODEL_CODEBERT),
            query=data.get("query", query),
        )

    async def calculate_similarity(
        self, code_a: str, code_b: str
    ) -> float:
        """Calculate similarity between two code snippets.
        
        Args:
            code_a: First code snippet
            code_b: Second code snippet
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not code_a or not code_b:
            return 0.0

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODE_SIMILARITY,
            json={"code_a": code_a, "code_b": code_b},
        )

        return float(data.get("similarity", 0.0))


# =============================================================================
# FakeCodeAnalysisClient (KB7.11)
# =============================================================================


class FakeCodeAnalysisClient(CodeAnalysisProtocol):
    """Fake Code Analysis client for testing.
    
    Produces deterministic results based on input hashes.
    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md
    """

    async def extract_keywords(
        self, code: str, top_k: int = 5
    ) -> KeywordResult:
        """Extract deterministic keywords from code.
        
        Args:
            code: Source code to analyze
            top_k: Number of keywords to extract
            
        Returns:
            Deterministic KeywordResult
        """
        if not code:
            return KeywordResult(keywords=[], scores=[], model=_MODEL_CODET5P)

        # Generate deterministic keywords based on code content
        keywords = self._extract_simple_keywords(code)[:top_k]
        # Generate high scores (0.85-0.95) for fake client validation
        scores = [0.95 - (i * 0.02) for i in range(len(keywords))]

        return KeywordResult(
            keywords=keywords,
            scores=scores,
            model=_MODEL_CODET5P,
        )

    async def validate_terms(
        self,
        terms: list[str],
        query: str,
        threshold: float = 0.5,
    ) -> TermValidationResult:
        """Validate terms with deterministic scores.
        
        Args:
            terms: List of terms to validate
            query: Query context
            threshold: Minimum score for validity
            
        Returns:
            Deterministic TermValidationResult
        """
        validated_terms = []
        for term in terms:
            # Deterministic scoring based on term/query similarity
            score = self._calculate_term_score(term, query)
            validated_terms.append({
                "term": term,
                "score": score,
                "valid": score >= threshold,
            })

        return TermValidationResult(
            terms=validated_terms,
            model=_MODEL_GRAPHCODEBERT,
            query=query,
        )

    async def rank_code_results(
        self,
        code_snippets: list[str],
        query: str,
        top_k: int | None = None,
    ) -> CodeRankingResult:
        """Rank code snippets deterministically.
        
        Args:
            code_snippets: List of code snippets
            query: Query to rank against
            top_k: Maximum results
            
        Returns:
            Deterministic CodeRankingResult
        """
        rankings = []
        for i, code in enumerate(code_snippets):
            score = self._calculate_code_score(code, query)
            rankings.append({
                "code": code,
                "score": score,
                "rank": i + 1,
            })

        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)

        # Update ranks after sorting
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        if top_k is not None:
            rankings = rankings[:top_k]

        return CodeRankingResult(
            rankings=rankings,
            model=_MODEL_CODEBERT,
            query=query,
        )

    async def calculate_similarity(
        self, code_a: str, code_b: str
    ) -> float:
        """Calculate deterministic similarity.
        
        Args:
            code_a: First code snippet
            code_b: Second code snippet
            
        Returns:
            Deterministic similarity score
        """
        if code_a == code_b:
            return 1.0

        if not code_a or not code_b:
            return 0.0

        # Hash-based similarity
        hash_a = hashlib.md5(code_a.encode()).hexdigest()
        hash_b = hashlib.md5(code_b.encode()).hexdigest()

        # Count matching characters
        matches = sum(a == b for a, b in zip(hash_a, hash_b))
        return matches / len(hash_a)

    async def close(self) -> None:
        """No-op for fake client."""
        pass

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_simple_keywords(self, code: str) -> list[str]:
        """Extract simple keywords from code.
        
        Args:
            code: Source code
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction: split by non-alphanumeric
        import re

        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code)
        # Filter common keywords and short words
        stopwords = {"def", "class", "self", "return", "if", "else", "for", "in", "import"}
        keywords = [w.lower() for w in words if len(w) > 2 and w.lower() not in stopwords]

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_keywords: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def _calculate_term_score(self, term: str, query: str) -> float:
        """Calculate term relevance score.
        
        Args:
            term: Term to score
            query: Query context
            
        Returns:
            Score between 0.0 and 1.0
        """
        term_lower = term.lower()
        query_lower = query.lower()

        # Exact match in query
        if term_lower in query_lower:
            return 0.95

        # Check for common substrings
        query_words = set(query_lower.split())
        if term_lower in query_words:
            return 0.92

        # Check for partial match (typo detection)
        for word in query_words:
            # If term is close to a query word but not exact, lower score
            if self._levenshtein_distance(term_lower, word) <= 2:
                if term_lower != word:
                    return 0.35  # Likely a typo
                return 0.85

        # Default score based on hash
        hash_val = int(hashlib.md5(f"{term}{query}".encode()).hexdigest()[:8], 16)
        return 0.3 + (hash_val % 50) / 100

    def _calculate_code_score(self, code: str, query: str) -> float:
        """Calculate code relevance score.
        
        Args:
            code: Code snippet
            query: Query to rank against
            
        Returns:
            Score between 0.0 and 1.0
        """
        code_lower = code.lower()
        query_words = query.lower().split()

        # Count query words found in code
        matches = sum(1 for word in query_words if word in code_lower)
        base_score = matches / len(query_words) if query_words else 0.5

        # Add hash-based variance for determinism
        hash_val = int(hashlib.md5(f"{code}{query}".encode()).hexdigest()[:8], 16)
        variance = (hash_val % 20) / 100

        return min(1.0, base_score * 0.7 + variance + 0.2)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


__all__ = [
    "CodeAnalysisClient",
    "CodeAnalysisConfig",
    "CodeAnalysisProtocol",
    "CodeRankingResult",
    "FakeCodeAnalysisClient",
    "KeywordResult",
    "TermValidationResult",
]

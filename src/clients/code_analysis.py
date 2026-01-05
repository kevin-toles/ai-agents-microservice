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

# =============================================================================
# CodeT5+ Endpoints (summarize, generate, translate, complete, understand, detect)
# =============================================================================
_ENDPOINT_CODET5_SUMMARIZE = "/v1/codet5/summarize"
_ENDPOINT_CODET5_GENERATE = "/v1/codet5/generate"
_ENDPOINT_CODET5_TRANSLATE = "/v1/codet5/translate"
_ENDPOINT_CODET5_COMPLETE = "/v1/codet5/complete"
_ENDPOINT_CODET5_UNDERSTAND = "/v1/codet5/understand"
_ENDPOINT_CODET5_DEFECTS = "/v1/codet5/detect-defects"
_ENDPOINT_CODET5_CLONES = "/v1/codet5/detect-clones"

# =============================================================================
# GraphCodeBERT Endpoints (validate, classify, expand, embeddings, similarity)
# =============================================================================
_ENDPOINT_GRAPHCODEBERT_VALIDATE = "/v1/graphcodebert/validate"
_ENDPOINT_GRAPHCODEBERT_CLASSIFY = "/v1/graphcodebert/classify-domain"
_ENDPOINT_GRAPHCODEBERT_EXPAND = "/v1/graphcodebert/expand"
_ENDPOINT_GRAPHCODEBERT_EMBEDDINGS = "/v1/graphcodebert/embeddings"
_ENDPOINT_GRAPHCODEBERT_SIMILARITY = "/v1/graphcodebert/similarity"

# =============================================================================
# CodeBERT Endpoints (embed, similarity, rank)
# =============================================================================
_ENDPOINT_CODEBERT_EMBED = "/api/v1/codebert/embed"
_ENDPOINT_CODEBERT_SIMILARITY = "/api/v1/codebert/similarity"
_ENDPOINT_CODEBERT_RANK = "/api/v1/codebert/rank"

# =============================================================================
# Embedding Endpoints (fused, BGE, UnixCoder, Instructor)
# =============================================================================
_ENDPOINT_EMBED_FUSED = "/api/v1/embed"
_ENDPOINT_BGE_EMBED = "/api/v1/bge/embed"
_ENDPOINT_UNIXCODER_EMBED = "/api/v1/unixcoder/embed"
_ENDPOINT_INSTRUCTOR_CONCEPTS = "/api/v1/instructor/concepts"

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


class CodeT5Result(BaseModel):
    """Result from CodeT5+ operations (summarize, generate, translate, etc.)."""

    model_config = ConfigDict(frozen=True)

    result: str = Field(
        default="",
        description="Generated text from CodeT5+",
    )
    model: str = Field(
        default=_MODEL_CODET5P,
        description="Model used",
    )


class DefectDetectionResult(BaseModel):
    """Result from CodeT5+ defect detection."""

    model_config = ConfigDict(frozen=True)

    has_defect: bool = Field(default=False, description="Whether defect detected")
    confidence: float = Field(default=0.0, description="Confidence score")
    defect_type: str | None = Field(None, description="Type of defect if detected")


class CloneDetectionResult(BaseModel):
    """Result from CodeT5+ clone detection."""

    model_config = ConfigDict(frozen=True)

    is_clone: bool = Field(default=False, description="Whether code is a clone")
    similarity: float = Field(default=0.0, description="Similarity score")
    clone_type: str | None = Field(None, description="Type of clone (1/2/3)")


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


class DomainClassificationResult(BaseModel):
    """Result from GraphCodeBERT domain classification."""

    model_config = ConfigDict(frozen=True)

    domain: str = Field(default="general", description="Classified domain")
    confidence: float = Field(default=0.0, description="Classification confidence")


class TermExpansionResult(BaseModel):
    """Result from GraphCodeBERT term expansion."""

    model_config = ConfigDict(frozen=True)

    original_terms: list[str] = Field(default_factory=list, description="Original terms")
    expanded_terms: list[str] = Field(default_factory=list, description="All terms including expansions")
    expansion_count: int = Field(default=0, description="Number of new terms added")


class EmbeddingResult(BaseModel):
    """Result from embedding operations (BGE, UnixCoder, Instructor)."""

    model_config = ConfigDict(frozen=True)

    embedding: list[float] = Field(default_factory=list, description="Embedding vector")
    dimension: int = Field(default=0, description="Embedding dimension")
    model: str = Field(default="", description="Model used")


class SimilarityScoresResult(BaseModel):
    """Result from batch similarity scoring."""

    model_config = ConfigDict(frozen=True)

    scores: dict[str, float] = Field(default_factory=dict, description="Term -> similarity score")
    query: str = Field(default="", description="Query used")


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class CodeAnalysisProtocol(Protocol):
    """Protocol for code analysis operations.
    
    Defines interface for CodeT5+, GraphCodeBERT, and CodeBERT operations.
    """

    # =========================================================================
    # CodeT5+ Operations (7 capabilities)
    # =========================================================================

    async def summarize_code(self, code: str, max_length: int = 64) -> CodeT5Result:
        """Summarize code using CodeT5+."""
        ...

    async def generate_code(self, prompt: str, max_length: int = 128) -> CodeT5Result:
        """Generate code from natural language prompt using CodeT5+."""
        ...

    async def translate_code(self, code: str, source_lang: str, target_lang: str) -> CodeT5Result:
        """Translate code between programming languages using CodeT5+."""
        ...

    async def complete_code(self, code: str, max_length: int = 64) -> CodeT5Result:
        """Complete partial code using CodeT5+."""
        ...

    async def understand_code(self, code: str) -> CodeT5Result:
        """Generate natural language description of code using CodeT5+."""
        ...

    async def detect_defects(self, code: str) -> DefectDetectionResult:
        """Detect potential defects in code using CodeT5+."""
        ...

    async def detect_clones(self, code1: str, code2: str) -> CloneDetectionResult:
        """Detect if two code snippets are clones using CodeT5+."""
        ...

    # =========================================================================
    # GraphCodeBERT Operations (5 capabilities)
    # =========================================================================

    async def validate_terms(
        self,
        terms: list[str],
        query: str,
        domain: str = "general",
        min_similarity: float = 0.15,
    ) -> TermValidationResult:
        """Validate terms against query using GraphCodeBERT."""
        ...

    async def classify_domain(self, text: str) -> DomainClassificationResult:
        """Classify the domain of text using GraphCodeBERT."""
        ...

    async def expand_terms(
        self,
        terms: list[str],
        domain: str = "general",
        max_expansions: int = 3,
        candidates: list[str] | None = None,
    ) -> TermExpansionResult:
        """Expand terms with related terms using GraphCodeBERT."""
        ...

    async def get_term_embeddings(self, terms: list[str]) -> dict[str, list[float]]:
        """Get GraphCodeBERT embeddings for terms."""
        ...

    async def batch_similarity(self, terms: list[str], query: str) -> SimilarityScoresResult:
        """Calculate batch similarity scores using GraphCodeBERT."""
        ...

    # =========================================================================
    # CodeBERT Operations (3 capabilities)
    # =========================================================================

    async def embed_code(self, code: str) -> EmbeddingResult:
        """Generate CodeBERT embedding for code."""
        ...

    async def rank_terms(self, terms: list[str], query: str) -> CodeRankingResult:
        """Rank terms by relevance to query using CodeBERT."""
        ...

    async def calculate_similarity(
        self, code_a: str, code_b: str
    ) -> float:
        """Calculate similarity between two code snippets using CodeBERT."""
        ...

    # =========================================================================
    # Direct Embedding Access (BGE, UnixCoder, Instructor)
    # =========================================================================

    async def bge_embed(self, text: str) -> EmbeddingResult:
        """Get BGE embedding for text (384-dim)."""
        ...

    async def unixcoder_embed(self, code: str) -> EmbeddingResult:
        """Get UnixCoder embedding for code (768-dim)."""
        ...

    async def instructor_embed_concepts(
        self, concepts: list[str], domain: str = "general"
    ) -> EmbeddingResult:
        """Get Instructor embedding for domain concepts (768-dim)."""
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
    # CodeT5+ API Methods
    # =========================================================================

    async def summarize_code(self, code: str, max_length: int = 64) -> CodeT5Result:
        """Summarize code using CodeT5+.
        
        Args:
            code: Source code to summarize
            max_length: Maximum output length
            
        Returns:
            CodeT5Result with generated summary
        """
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_SUMMARIZE,
            json={"code": code, "max_length": max_length},
        )
        return CodeT5Result(result=data.get("summary", ""), model=_MODEL_CODET5P)

    async def generate_code(self, prompt: str, max_length: int = 128) -> CodeT5Result:
        """Generate code from natural language prompt using CodeT5+.
        
        Args:
            prompt: Natural language description
            max_length: Maximum output length
            
        Returns:
            CodeT5Result with generated code
        """
        if not prompt:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_GENERATE,
            json={"prompt": prompt, "max_length": max_length},
        )
        return CodeT5Result(result=data.get("generated_code", ""), model=_MODEL_CODET5P)

    async def translate_code(self, code: str, source_lang: str, target_lang: str) -> CodeT5Result:
        """Translate code between programming languages using CodeT5+.
        
        Args:
            code: Source code to translate
            source_lang: Source programming language
            target_lang: Target programming language
            
        Returns:
            CodeT5Result with translated code
        """
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_TRANSLATE,
            json={"code": code, "source_lang": source_lang, "target_lang": target_lang},
        )
        return CodeT5Result(result=data.get("translated_code", ""), model=_MODEL_CODET5P)

    async def complete_code(self, code: str, max_length: int = 64) -> CodeT5Result:
        """Complete partial code using CodeT5+.
        
        Args:
            code: Partial code to complete
            max_length: Maximum output length
            
        Returns:
            CodeT5Result with completed code
        """
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_COMPLETE,
            json={"code": code, "max_length": max_length},
        )
        return CodeT5Result(result=data.get("completed_code", ""), model=_MODEL_CODET5P)

    async def understand_code(self, code: str) -> CodeT5Result:
        """Generate natural language description of code using CodeT5+.
        
        Args:
            code: Source code to describe
            
        Returns:
            CodeT5Result with natural language description
        """
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_UNDERSTAND,
            json={"code": code},
        )
        return CodeT5Result(result=data.get("description", ""), model=_MODEL_CODET5P)

    async def detect_defects(self, code: str) -> DefectDetectionResult:
        """Detect potential defects in code using CodeT5+.
        
        Args:
            code: Source code to analyze
            
        Returns:
            DefectDetectionResult with detection results
        """
        if not code:
            return DefectDetectionResult(has_defect=False, confidence=0.0)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_DEFECTS,
            json={"code": code},
        )
        return DefectDetectionResult(
            has_defect=data.get("has_defect", False),
            confidence=data.get("confidence", 0.0),
            defect_type=data.get("defect_type"),
        )

    async def detect_clones(self, code1: str, code2: str) -> CloneDetectionResult:
        """Detect if two code snippets are clones using CodeT5+.
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            CloneDetectionResult with detection results
        """
        if not code1 or not code2:
            return CloneDetectionResult(is_clone=False, similarity=0.0)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODET5_CLONES,
            json={"code1": code1, "code2": code2},
        )
        return CloneDetectionResult(
            is_clone=data.get("is_clone", False),
            similarity=data.get("similarity", 0.0),
            clone_type=data.get("clone_type"),
        )

    # =========================================================================
    # GraphCodeBERT API Methods
    # =========================================================================

    async def validate_terms(
        self,
        terms: list[str],
        query: str,
        domain: str = "general",
        min_similarity: float = 0.15,
    ) -> TermValidationResult:
        """Validate terms against query using GraphCodeBERT.
        
        Args:
            terms: List of terms to validate
            query: Query context for validation
            domain: Target domain (ai-ml, systems, web, data, general)
            min_similarity: Minimum similarity threshold
            
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
            endpoint=_ENDPOINT_GRAPHCODEBERT_VALIDATE,
            json={
                "terms": terms,
                "query": query,
                "domain": domain,
                "min_similarity": min_similarity,
            },
        )

        # Convert to expected format
        validated_terms = []
        for term in data.get("valid_terms", []):
            validated_terms.append({
                "term": term,
                "score": data.get("similarity_scores", {}).get(term, 0.0),
                "valid": True,
            })
        for term in data.get("rejected_terms", []):
            validated_terms.append({
                "term": term,
                "score": 0.0,
                "valid": False,
                "reason": data.get("rejection_reasons", {}).get(term, ""),
            })

        return TermValidationResult(
            terms=validated_terms,
            model=_MODEL_GRAPHCODEBERT,
            query=query,
        )

    async def classify_domain(self, text: str) -> DomainClassificationResult:
        """Classify the domain of text using GraphCodeBERT.
        
        Args:
            text: Text to classify
            
        Returns:
            DomainClassificationResult with domain and confidence
        """
        if not text:
            return DomainClassificationResult(domain="general", confidence=0.0)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_GRAPHCODEBERT_CLASSIFY,
            json={"text": text},
        )
        return DomainClassificationResult(
            domain=data.get("domain", "general"),
            confidence=data.get("confidence", 0.0),
        )

    async def expand_terms(
        self,
        terms: list[str],
        domain: str = "general",
        max_expansions: int = 3,
        candidates: list[str] | None = None,
    ) -> TermExpansionResult:
        """Expand terms with related terms using GraphCodeBERT.
        
        Args:
            terms: Terms to expand
            domain: Target domain for context
            max_expansions: Max related terms per input
            candidates: Optional candidate terms to search
            
        Returns:
            TermExpansionResult with expanded terms
        """
        if not terms:
            return TermExpansionResult(original_terms=[], expanded_terms=[], expansion_count=0)

        payload: dict[str, Any] = {
            "terms": terms,
            "domain": domain,
            "max_expansions": max_expansions,
        }
        if candidates:
            payload["candidates"] = candidates

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_GRAPHCODEBERT_EXPAND,
            json=payload,
        )
        return TermExpansionResult(
            original_terms=data.get("original_terms", terms),
            expanded_terms=data.get("expanded_terms", terms),
            expansion_count=data.get("expansion_count", 0),
        )

    async def get_term_embeddings(self, terms: list[str]) -> dict[str, list[float]]:
        """Get GraphCodeBERT embeddings for terms.
        
        Args:
            terms: Terms to embed
            
        Returns:
            Dictionary mapping term -> embedding vector
        """
        if not terms:
            return {}

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_GRAPHCODEBERT_EMBEDDINGS,
            json={"terms": terms},
        )
        return data.get("embeddings", {})

    async def batch_similarity(self, terms: list[str], query: str) -> SimilarityScoresResult:
        """Calculate batch similarity scores using GraphCodeBERT.
        
        Args:
            terms: Terms to score
            query: Query for comparison
            
        Returns:
            SimilarityScoresResult with scores for each term
        """
        if not terms or not query:
            return SimilarityScoresResult(scores={}, query=query)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_GRAPHCODEBERT_SIMILARITY,
            json={"terms": terms, "query": query},
        )
        return SimilarityScoresResult(
            scores=data.get("scores", {}),
            query=data.get("query", query),
        )

    # =========================================================================
    # CodeBERT API Methods
    # =========================================================================

    async def embed_code(self, code: str) -> EmbeddingResult:
        """Generate CodeBERT embedding for code.
        
        Args:
            code: Source code to embed
            
        Returns:
            EmbeddingResult with 768-dimensional embedding
        """
        if not code:
            return EmbeddingResult(embedding=[], dimension=0, model=_MODEL_CODEBERT)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODEBERT_EMBED,
            json={"code": code},
        )
        return EmbeddingResult(
            embedding=data.get("embedding", []),
            dimension=data.get("dimension", 768),
            model=_MODEL_CODEBERT,
        )

    async def rank_terms(self, terms: list[str], query: str) -> CodeRankingResult:
        """Rank terms by relevance to query using CodeBERT.
        
        Args:
            terms: List of terms to rank
            query: Query to rank against
            
        Returns:
            CodeRankingResult with ranked terms
        """
        if not terms:
            return CodeRankingResult(rankings=[], model=_MODEL_CODEBERT, query=query)

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_CODEBERT_RANK,
            json={"terms": terms, "query": query},
        )

        # Convert API response format to rankings list
        rankings = []
        for item in data.get("ranked_terms", []):
            rankings.append({
                "term": item.get("term", ""),
                "score": item.get("score", 0.0),
            })

        return CodeRankingResult(
            rankings=rankings,
            model=_MODEL_CODEBERT,
            query=data.get("query", query),
        )

    async def calculate_similarity(
        self, code_a: str, code_b: str
    ) -> float:
        """Calculate similarity between two code snippets using CodeBERT.
        
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
            endpoint=_ENDPOINT_CODEBERT_SIMILARITY,
            json={"code_a": code_a, "code_b": code_b},
        )

        return float(data.get("similarity", 0.0))

    # =========================================================================
    # Direct Embedding API Methods (BGE, UnixCoder, Instructor)
    # =========================================================================

    async def bge_embed(self, text: str) -> EmbeddingResult:
        """Get BGE embedding for text (384-dim).
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with 384-dimensional embedding
        """
        if not text:
            return EmbeddingResult(embedding=[], dimension=0, model="bge")

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_BGE_EMBED,
            json={"text": text},
        )
        return EmbeddingResult(
            embedding=data.get("embedding", []),
            dimension=data.get("dimension", 384),
            model=data.get("model", "BAAI/bge-small-en-v1.5"),
        )

    async def unixcoder_embed(self, code: str) -> EmbeddingResult:
        """Get UnixCoder embedding for code (768-dim).
        
        Args:
            code: Code to embed
            
        Returns:
            EmbeddingResult with 768-dimensional embedding
        """
        if not code:
            return EmbeddingResult(embedding=[], dimension=0, model="unixcoder")

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_UNIXCODER_EMBED,
            json={"code": code},
        )
        return EmbeddingResult(
            embedding=data.get("embedding", []),
            dimension=data.get("dimension", 768),
            model=data.get("model", "microsoft/unixcoder-base"),
        )

    async def instructor_embed_concepts(
        self, concepts: list[str], domain: str = "general"
    ) -> EmbeddingResult:
        """Get Instructor embedding for domain concepts (768-dim).
        
        Args:
            concepts: Concepts to embed
            domain: Domain for context (ai-ml, systems, web, data)
            
        Returns:
            EmbeddingResult with 768-dimensional embedding
        """
        if not concepts:
            return EmbeddingResult(embedding=[], dimension=0, model="instructor")

        data = await self._request_with_retry(
            method="POST",
            endpoint=_ENDPOINT_INSTRUCTOR_CONCEPTS,
            json={"concepts": concepts, "domain": domain},
        )
        return EmbeddingResult(
            embedding=data.get("embedding", []),
            dimension=data.get("dimension", 768),
            model=data.get("model", "hkunlp/instructor-xl"),
        )

    # =========================================================================
    # Legacy API Methods (for backward compatibility)
    # =========================================================================

    async def extract_keywords(self, code: str, top_k: int = 5) -> KeywordResult:
        """Extract keywords from code using CodeT5+ summarization.
        
        DEPRECATED: Use summarize_code() instead. This is a wrapper that
        extracts keywords from the summary for backward compatibility.
        
        Args:
            code: Source code to analyze
            top_k: Number of keywords to extract
            
        Returns:
            KeywordResult with keywords and scores
        """
        # Use summarization and extract keywords from summary
        result = await self.summarize_code(code)
        if not result.result:
            return KeywordResult(keywords=[], scores=[], model=_MODEL_CODET5P)

        # Simple keyword extraction from summary
        import re
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", result.result)
        stopwords = {"the", "a", "an", "is", "are", "and", "or", "of", "for", "to", "in", "it"}
        keywords = [w.lower() for w in words if len(w) > 2 and w.lower() not in stopwords]

        # Deduplicate
        seen: set[str] = set()
        unique_keywords: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        keywords_final = unique_keywords[:top_k]
        scores = [0.95 - (i * 0.05) for i in range(len(keywords_final))]

        return KeywordResult(keywords=keywords_final, scores=scores, model=_MODEL_CODET5P)

    async def rank_code_results(
        self,
        code_snippets: list[str],
        query: str,
        top_k: int | None = None,
    ) -> CodeRankingResult:
        """Rank code snippets by relevance using CodeBERT.
        
        DEPRECATED: Use rank_terms() for term ranking.
        This method ranks code snippets using similarity calculations.
        
        Args:
            code_snippets: List of code snippets to rank
            query: Query to rank against
            top_k: Maximum number of results to return
            
        Returns:
            CodeRankingResult with ranked code snippets
        """
        if not code_snippets:
            return CodeRankingResult(rankings=[], model=_MODEL_CODEBERT, query=query)

        # Calculate similarity for each snippet
        rankings = []
        for i, code in enumerate(code_snippets):
            score = await self.calculate_similarity(code, query)
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

        return CodeRankingResult(rankings=rankings, model=_MODEL_CODEBERT, query=query)


# =============================================================================
# FakeCodeAnalysisClient (KB7.11)
# =============================================================================


class FakeCodeAnalysisClient(CodeAnalysisProtocol):
    """Fake Code Analysis client for testing.
    
    Produces deterministic results based on input hashes.
    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md
    """

    # =========================================================================
    # CodeT5+ Fake Methods
    # =========================================================================

    async def summarize_code(self, code: str, max_length: int = 64) -> CodeT5Result:
        """Generate deterministic summary."""
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        # Extract key elements for fake summary
        keywords = self._extract_simple_keywords(code)[:3]
        summary = f"Code implementing {', '.join(keywords)}" if keywords else "Code snippet"
        return CodeT5Result(result=summary[:max_length], model=_MODEL_CODET5P)

    async def generate_code(self, prompt: str, max_length: int = 128) -> CodeT5Result:
        """Generate deterministic code from prompt."""
        if not prompt:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        # Simple template-based generation
        code = f"def generated_from_prompt():\n    # {prompt[:50]}\n    pass"
        return CodeT5Result(result=code[:max_length], model=_MODEL_CODET5P)

    async def translate_code(self, code: str, source_lang: str, target_lang: str) -> CodeT5Result:
        """Return translated code (fake)."""
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)
        # Just return original code with comment
        return CodeT5Result(
            result=f"# Translated from {source_lang} to {target_lang}\n{code}",
            model=_MODEL_CODET5P,
        )

    async def complete_code(self, code: str, max_length: int = 64) -> CodeT5Result:
        """Complete code deterministically."""
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)
        return CodeT5Result(result=f"{code}\n    pass  # auto-completed", model=_MODEL_CODET5P)

    async def understand_code(self, code: str) -> CodeT5Result:
        """Generate description of code."""
        if not code:
            return CodeT5Result(result="", model=_MODEL_CODET5P)

        keywords = self._extract_simple_keywords(code)[:3]
        desc = f"This code handles {', '.join(keywords)}" if keywords else "This is a code snippet"
        return CodeT5Result(result=desc, model=_MODEL_CODET5P)

    async def detect_defects(self, code: str) -> DefectDetectionResult:
        """Detect defects deterministically."""
        if not code:
            return DefectDetectionResult(has_defect=False, confidence=0.0)

        # Check for common issues
        has_defect = "except:" in code or "pass" in code
        return DefectDetectionResult(
            has_defect=has_defect,
            confidence=0.75 if has_defect else 0.1,
            defect_type="bare-except" if "except:" in code else None,
        )

    async def detect_clones(self, code1: str, code2: str) -> CloneDetectionResult:
        """Detect code clones deterministically."""
        if not code1 or not code2:
            return CloneDetectionResult(is_clone=False, similarity=0.0)

        if code1 == code2:
            return CloneDetectionResult(is_clone=True, similarity=1.0, clone_type="1")

        # Hash-based similarity
        hash_a = hashlib.md5(code1.encode()).hexdigest()
        hash_b = hashlib.md5(code2.encode()).hexdigest()
        matches = sum(a == b for a, b in zip(hash_a, hash_b))
        similarity = matches / len(hash_a)

        return CloneDetectionResult(
            is_clone=similarity > 0.5,
            similarity=similarity,
            clone_type="3" if similarity > 0.5 else None,
        )

    # =========================================================================
    # GraphCodeBERT Fake Methods
    # =========================================================================

    async def validate_terms(
        self,
        terms: list[str],
        query: str,
        domain: str = "general",
        min_similarity: float = 0.15,
    ) -> TermValidationResult:
        """Validate terms with deterministic scores."""
        validated_terms = []
        for term in terms:
            score = self._calculate_term_score(term, query)
            validated_terms.append({
                "term": term,
                "score": score,
                "valid": score >= min_similarity,
            })

        return TermValidationResult(
            terms=validated_terms,
            model=_MODEL_GRAPHCODEBERT,
            query=query,
        )

    async def classify_domain(self, text: str) -> DomainClassificationResult:
        """Classify domain deterministically."""
        if not text:
            return DomainClassificationResult(domain="general", confidence=0.0)

        text_lower = text.lower()
        if any(w in text_lower for w in ["neural", "ml", "model", "train"]):
            return DomainClassificationResult(domain="ai-ml", confidence=0.85)
        if any(w in text_lower for w in ["http", "api", "web", "html"]):
            return DomainClassificationResult(domain="web", confidence=0.85)
        if any(w in text_lower for w in ["docker", "k8s", "deploy", "server"]):
            return DomainClassificationResult(domain="systems", confidence=0.85)
        if any(w in text_lower for w in ["sql", "database", "table", "query"]):
            return DomainClassificationResult(domain="data", confidence=0.85)

        return DomainClassificationResult(domain="general", confidence=0.5)

    async def expand_terms(
        self,
        terms: list[str],
        domain: str = "general",
        max_expansions: int = 3,
        candidates: list[str] | None = None,
    ) -> TermExpansionResult:
        """Expand terms deterministically."""
        if not terms:
            return TermExpansionResult(original_terms=[], expanded_terms=[], expansion_count=0)

        # Simple suffix-based expansion
        expanded = list(terms)
        for term in terms[:max_expansions]:
            expanded.append(f"{term}_related")

        return TermExpansionResult(
            original_terms=terms,
            expanded_terms=expanded,
            expansion_count=len(expanded) - len(terms),
        )

    async def get_term_embeddings(self, terms: list[str]) -> dict[str, list[float]]:
        """Get fake embeddings for terms."""
        result = {}
        for term in terms:
            # Hash-based deterministic embedding
            hash_bytes = hashlib.sha256(term.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:768]]
            result[term] = embedding
        return result

    async def batch_similarity(self, terms: list[str], query: str) -> SimilarityScoresResult:
        """Calculate batch similarity deterministically."""
        scores = {}
        for term in terms:
            scores[term] = self._calculate_term_score(term, query)
        return SimilarityScoresResult(scores=scores, query=query)

    # =========================================================================
    # CodeBERT Fake Methods
    # =========================================================================

    async def embed_code(self, code: str) -> EmbeddingResult:
        """Generate fake CodeBERT embedding."""
        if not code:
            return EmbeddingResult(embedding=[], dimension=0, model=_MODEL_CODEBERT)

        hash_bytes = hashlib.sha256(code.encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes[:768]]
        return EmbeddingResult(embedding=embedding, dimension=768, model=_MODEL_CODEBERT)

    async def rank_terms(self, terms: list[str], query: str) -> CodeRankingResult:
        """Rank terms deterministically."""
        rankings = []
        for term in terms:
            score = self._calculate_term_score(term, query)
            rankings.append({"term": term, "score": score})

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return CodeRankingResult(rankings=rankings, model=_MODEL_CODEBERT, query=query)

    async def calculate_similarity(self, code_a: str, code_b: str) -> float:
        """Calculate fake similarity."""
        if code_a == code_b:
            return 1.0
        if not code_a or not code_b:
            return 0.0

        hash_a = hashlib.md5(code_a.encode()).hexdigest()
        hash_b = hashlib.md5(code_b.encode()).hexdigest()
        matches = sum(a == b for a, b in zip(hash_a, hash_b))
        return matches / len(hash_a)

    # =========================================================================
    # Direct Embedding Fake Methods
    # =========================================================================

    async def bge_embed(self, text: str) -> EmbeddingResult:
        """Generate fake BGE embedding."""
        if not text:
            return EmbeddingResult(embedding=[], dimension=0, model="bge")

        hash_bytes = hashlib.sha256(f"bge:{text}".encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes[:384]]
        return EmbeddingResult(embedding=embedding, dimension=384, model="BAAI/bge-small-en-v1.5")

    async def unixcoder_embed(self, code: str) -> EmbeddingResult:
        """Generate fake UnixCoder embedding."""
        if not code:
            return EmbeddingResult(embedding=[], dimension=0, model="unixcoder")

        hash_bytes = hashlib.sha256(f"unix:{code}".encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes[:768]]
        return EmbeddingResult(embedding=embedding, dimension=768, model="microsoft/unixcoder-base")

    async def instructor_embed_concepts(
        self, concepts: list[str], domain: str = "general"
    ) -> EmbeddingResult:
        """Generate fake Instructor embedding."""
        if not concepts:
            return EmbeddingResult(embedding=[], dimension=0, model="instructor")

        combined = f"{domain}:{','.join(concepts)}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes[:768]]
        return EmbeddingResult(embedding=embedding, dimension=768, model="hkunlp/instructor-xl")

    async def close(self) -> None:
        """No-op for fake client."""
        pass

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================

    async def extract_keywords(self, code: str, top_k: int = 5) -> KeywordResult:
        """Extract deterministic keywords from code."""
        if not code:
            return KeywordResult(keywords=[], scores=[], model=_MODEL_CODET5P)

        keywords = self._extract_simple_keywords(code)[:top_k]
        scores = [0.95 - (i * 0.02) for i in range(len(keywords))]

        return KeywordResult(keywords=keywords, scores=scores, model=_MODEL_CODET5P)

    async def rank_code_results(
        self,
        code_snippets: list[str],
        query: str,
        top_k: int | None = None,
    ) -> CodeRankingResult:
        """Rank code snippets deterministically."""
        rankings = []
        for i, code in enumerate(code_snippets):
            score = self._calculate_code_score(code, query)
            rankings.append({"code": code, "score": score, "rank": i + 1})

        rankings.sort(key=lambda x: x["score"], reverse=True)
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        if top_k is not None:
            rankings = rankings[:top_k]

        return CodeRankingResult(rankings=rankings, model=_MODEL_CODEBERT, query=query)

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
    "CodeT5Result",
    "CloneDetectionResult",
    "DefectDetectionResult",
    "DomainClassificationResult",
    "EmbeddingResult",
    "FakeCodeAnalysisClient",
    "KeywordResult",
    "SimilarityScoresResult",
    "TermExpansionResult",
    "TermValidationResult",
]

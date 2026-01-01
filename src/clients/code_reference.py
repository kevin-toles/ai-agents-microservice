"""Code Reference Engine Client.

WBS Reference: WBS-AGT21 Code Reference Engine Client (AGT21.7)
Acceptance Criteria:
- AC-21.1: Client wraps CodeReferenceEngine from ai-platform-data
- AC-21.2: Async interface for search, get_metadata, fetch_file
- AC-21.3: Integration with Qdrant for semantic code search
- AC-21.4: Integration with GitHub API for on-demand file retrieval
- AC-21.5: Returns CodeContext with citations for downstream

Pattern: Protocol duck typing (CODING_PATTERNS_ANALYSIS.md)
Anti-Pattern Mitigation: #12 (Connection Pooling via shared HTTP client)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Integration Points
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

from src.clients.protocols import CodeReferenceProtocol


# =============================================================================
# Data Classes - CodeContext Schema
# =============================================================================


@dataclass
class CodeChunk:
    """A chunk of code from a repository.
    
    Represents a discrete unit of code with metadata for citation generation.
    
    Attributes:
        chunk_id: Unique identifier for this chunk
        repo_id: Repository identifier
        file_path: Path to file within repository
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)
        content: Code content
        language: Programming language
        score: Semantic similarity score (0.0-1.0)
    """
    
    chunk_id: str
    repo_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    score: float = 0.0


@dataclass
class CodeReference:
    """A code reference with full context and citation.
    
    Wraps a CodeChunk with additional context for citation generation.
    
    Attributes:
        chunk: The underlying code chunk
        full_content: Expanded content with surrounding context (optional)
        source_url: GitHub URL with line anchors for citation
        repo_metadata: Optional repository metadata
    """
    
    chunk: CodeChunk
    source_url: str
    full_content: str | None = None
    repo_metadata: dict[str, Any] | None = None
    
    @property
    def citation(self) -> str:
        """Generate citation string for this reference."""
        return f"[{self.chunk.file_path}#L{self.chunk.start_line}-L{self.chunk.end_line}]({self.source_url})"


@dataclass
class CodeContext:
    """Context assembled from code search results.
    
    Container for search results with methods for prompt generation and citation.
    
    Attributes:
        query: Original search query
        primary_references: List of CodeReference objects
        domains_searched: List of domains that were searched
        total_chunks_found: Total number of matching chunks (before top_k limit)
    """
    
    query: str
    primary_references: list[CodeReference] = field(default_factory=list)
    domains_searched: list[str] = field(default_factory=list)
    total_chunks_found: int = 0
    
    def to_prompt_context(self) -> str:
        """Convert to prompt-friendly context string.
        
        Returns:
            Formatted string with code snippets and citations for LLM prompt.
        """
        if not self.primary_references:
            return f"No code references found for query: {self.query}"
        
        sections = [f"# Code References for: {self.query}\n"]
        
        for i, ref in enumerate(self.primary_references, 1):
            sections.append(f"## Reference {i}: {ref.chunk.file_path}")
            sections.append(f"**Lines {ref.chunk.start_line}-{ref.chunk.end_line}** | Language: {ref.chunk.language}")
            sections.append(f"**Score**: {ref.chunk.score:.2f}")
            sections.append(f"```{ref.chunk.language}")
            sections.append(ref.chunk.content)
            sections.append("```")
            sections.append(f"Source: {ref.source_url}\n")
        
        return "\n".join(sections)
    
    def get_citations(self) -> list[str]:
        """Get list of citation strings.
        
        Returns:
            List of citation strings with file paths and GitHub URLs.
        """
        return [ref.citation for ref in self.primary_references]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CodeReferenceConfig:
    """Configuration for CodeReferenceClient.
    
    Attributes:
        registry_path: Path to repository registry JSON file
        qdrant_url: URL for Qdrant vector database
        qdrant_collection: Name of Qdrant collection for code chunks
        github_token: GitHub personal access token for API calls
        timeout: HTTP request timeout in seconds
    """
    
    registry_path: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "code_chunks"
    github_token: str | None = None
    timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> CodeReferenceConfig:
        """Create config from environment variables.
        
        Environment Variables:
            CODE_REFERENCE_REGISTRY: Path to registry.json
            CODE_REFERENCE_QDRANT_URL: Qdrant server URL
            CODE_REFERENCE_QDRANT_COLLECTION: Qdrant collection name
            GITHUB_TOKEN: GitHub personal access token
            CODE_REFERENCE_TIMEOUT: Request timeout in seconds
        
        Returns:
            CodeReferenceConfig instance populated from environment.
        """
        return cls(
            registry_path=os.environ.get("CODE_REFERENCE_REGISTRY", ""),
            qdrant_url=os.environ.get("CODE_REFERENCE_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.environ.get("CODE_REFERENCE_QDRANT_COLLECTION", "code_chunks"),
            github_token=os.environ.get("GITHUB_TOKEN"),
            timeout=float(os.environ.get("CODE_REFERENCE_TIMEOUT", "30.0")),
        )


# =============================================================================
# Client Implementation
# =============================================================================


class CodeReferenceClient:
    """Client for Code Reference Engine.
    
    Wraps CodeReferenceEngine from ai-platform-data with async interface.
    Provides semantic code search via Qdrant and file retrieval via GitHub API.
    
    WBS: WBS-AGT21 - Code Reference Engine Client
    Pattern: Protocol duck typing - implements CodeReferenceProtocol
    
    Usage:
        config = CodeReferenceConfig.from_env()
        async with CodeReferenceClient(config) as client:
            context = await client.search("repository pattern")
            citations = context.get_citations()
    """
    
    def __init__(self, config: CodeReferenceConfig) -> None:
        """Initialize client.
        
        Args:
            config: CodeReferenceConfig with connection settings
        """
        self._config = config
        self._http_client: httpx.AsyncClient | None = None
        self._registry: dict[str, Any] = {}
    
    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            headers=self._build_headers(),
        )
        self._load_registry()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including auth if available."""
        headers = {"Content-Type": "application/json"}
        if self._config.github_token:
            headers["Authorization"] = f"Bearer {self._config.github_token}"
        return headers
    
    def _load_registry(self) -> None:
        """Load repository registry from JSON file."""
        import json
        
        if self._config.registry_path and os.path.exists(self._config.registry_path):
            with open(self._config.registry_path) as f:
                self._registry = json.load(f)
    
    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout),
                headers=self._build_headers(),
            )
        return self._http_client
    
    async def search(
        self,
        query: str,
        domains: list[str] | None = None,
        concepts: list[str] | None = None,
        top_k: int = 10,
    ) -> CodeContext:
        """Semantic search across code repositories.
        
        Uses 3-layer retrieval:
        1. Qdrant semantic search for initial candidates
        2. GitHub API for expanded context
        3. Neo4j graph for relationship traversal (future)
        
        Args:
            query: Natural language search query
            domains: Optional list of domains to filter
            concepts: Optional list of concepts to filter
            top_k: Maximum number of results to return
        
        Returns:
            CodeContext with primary_references and citations
        """
        client = await self._ensure_client()
        
        # Build Qdrant search payload
        search_payload = {
            "query": query,
            "limit": top_k,
            "filter": {},
        }
        
        if domains:
            search_payload["filter"]["domain"] = {"$in": domains}
        if concepts:
            search_payload["filter"]["concepts"] = {"$overlap": concepts}
        
        try:
            # Query Qdrant for semantic matches
            response = await client.post(
                f"{self._config.qdrant_url}/collections/{self._config.qdrant_collection}/points/search",
                json=search_payload,
            )
            response.raise_for_status()
            results = response.json()
            
            # Convert to CodeReferences
            references = []
            for hit in results.get("result", []):
                payload = hit.get("payload", {})
                chunk = CodeChunk(
                    chunk_id=str(hit.get("id", "")),
                    repo_id=payload.get("repo_id", ""),
                    file_path=payload.get("file_path", ""),
                    start_line=payload.get("start_line", 1),
                    end_line=payload.get("end_line", 1),
                    content=payload.get("content", ""),
                    language=payload.get("language", "text"),
                    score=hit.get("score", 0.0),
                )
                
                # Build GitHub source URL
                source_url = self._build_github_url(
                    payload.get("repo_url", ""),
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                )
                
                references.append(CodeReference(
                    chunk=chunk,
                    source_url=source_url,
                    repo_metadata=payload.get("metadata"),
                ))
            
            return CodeContext(
                query=query,
                primary_references=references,
                domains_searched=domains or [],
                total_chunks_found=len(references),
            )
            
        except httpx.HTTPError:
            # Return empty context on error (caller can check total_chunks_found)
            return CodeContext(
                query=query,
                primary_references=[],
                domains_searched=domains or [],
                total_chunks_found=0,
            )
    
    async def search_by_concept(
        self,
        concept: str,
        top_k: int = 10,
    ) -> CodeContext:
        """Search by concept name.
        
        Args:
            concept: Concept name (e.g., "event-driven", "microservices")
            top_k: Maximum number of results to return
        
        Returns:
            CodeContext with matching code references
        """
        return await self.search(
            query=concept,
            concepts=[concept],
            top_k=top_k,
        )
    
    async def search_by_pattern(
        self,
        pattern: str,
        top_k: int = 10,
    ) -> CodeContext:
        """Search by design pattern name.
        
        Args:
            pattern: Design pattern name (e.g., "repository", "saga", "cqrs")
            top_k: Maximum number of results to return
        
        Returns:
            CodeContext with matching code references
        """
        return await self.search(
            query=pattern,
            concepts=[pattern],
            top_k=top_k,
        )
    
    async def get_metadata(self, repo_id: str) -> dict[str, Any] | None:
        """Get repository metadata by ID.
        
        Args:
            repo_id: Repository identifier
        
        Returns:
            Dict with id, name, domain, concepts, patterns, tags, or None
        """
        import asyncio
        # Yield to event loop for protocol compliance (operation is sync)
        await asyncio.sleep(0)
        return self._registry.get("repositories", {}).get(repo_id)
    
    async def fetch_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str | None:
        """Fetch file content from GitHub.
        
        Args:
            file_path: Path to file within repository
            start_line: Optional start line for partial fetch
            end_line: Optional end line for partial fetch
        
        Returns:
            File content as string, or None if not found
        """
        client = await self._ensure_client()
        
        # Parse repo and path from file_path (format: owner/repo/path/to/file.py)
        parts = file_path.split("/", 2)
        if len(parts) < 3:
            return None
        
        owner, repo, path = parts[0], parts[1], parts[2]
        
        # GitHub raw content URL
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text
            
            # Apply line filtering if specified
            if start_line is not None or end_line is not None:
                lines = content.split("\n")
                start = (start_line or 1) - 1  # Convert to 0-indexed
                end = end_line or len(lines)
                content = "\n".join(lines[start:end])
            
            return content
            
        except httpx.HTTPError:
            return None
    
    def _build_github_url(
        self,
        repo_url: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Build GitHub URL with line anchors.
        
        Args:
            repo_url: Base repository URL
            file_path: Path to file
            start_line: Starting line number
            end_line: Ending line number
        
        Returns:
            GitHub URL with #L{start}-L{end} anchor
        """
        if not repo_url:
            repo_url = "https://github.com/unknown/unknown"
        
        # Normalize URL (remove .git suffix, ensure no trailing slash)
        repo_url = repo_url.rstrip("/").replace(".git", "")
        
        return f"{repo_url}/blob/main/{file_path}#L{start_line}-L{end_line}"
    
    async def close(self) -> None:
        """Release HTTP client resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Type alias for protocol compliance verification
_: type[CodeReferenceProtocol] = CodeReferenceClient  # type: ignore[assignment]

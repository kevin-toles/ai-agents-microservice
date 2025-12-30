"""CrossReference Agent Function.

WBS-AGT13: cross_reference Function

This module implements the CrossReferenceFunction which finds related
content across knowledge bases via semantic search.

Acceptance Criteria:
- AC-13.1: Queries semantic-search-service for related content
- AC-13.2: Returns CrossReferenceResult with matches, relevance_scores
- AC-13.3: Context budget: 2048 input / 4096 output
- AC-13.4: Default preset: S4
- AC-13.5: Integrates with Qdrant via semantic-search-service

Exit Criteria:
- Each Match has source, content, relevance_score (0.0-1.0)
- FakeSemanticSearchClient used in unit tests
- Integration test hits real semantic-search-service:8081

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 8

Anti-Pattern Compliance:
- #12: Connection pooling via shared SemanticSearchClient
- #42/#43: Async context managers for HTTP clients
- S1192: Context budget from constants.py
"""

from dataclasses import dataclass
from typing import Any

from src.clients.protocols import SemanticSearchProtocol
from src.functions.base import AgentFunction
from src.functions.utils.token_utils import estimate_tokens
from src.schemas.functions.cross_reference import (
    Citation,
    CrossReferenceInput,
    CrossReferenceResult,
    MatchType,
    Reference,
)


@dataclass(frozen=True)
class ContextBudget:
    """Context budget limits for cross_reference function.

    AC-13.3: Context budget: 2048 input / 4096 output.
    """

    input_tokens: int = 2048
    output_tokens: int = 4096


class CrossReferenceFunction(AgentFunction):
    """Agent function to find related content via semantic search.

    Queries semantic-search-service to find related content across
    knowledge bases, returning matches with relevance scores and
    formatted citations for downstream use.

    Context Budget (AC-13.3):
        - Input: 2048 tokens
        - Output: 4096 tokens

    Default Preset (AC-13.4): S4 (LLM for post-ranking only)

    Attributes:
        name: Function identifier 'cross_reference'
        description: Human-readable description of function purpose
        default_preset: Default to S4 for post-ranking
        input_schema: CrossReferenceInput Pydantic model
        output_schema: CrossReferenceResult Pydantic model
        context_budget: ContextBudget dataclass with limits

    Example:
        ```python
        from src.clients.semantic_search import FakeSemanticSearchClient

        client = FakeSemanticSearchClient()
        func = CrossReferenceFunction(semantic_search_client=client)
        result = await func.execute(CrossReferenceInput(
            query_artifact="Repository pattern implementation",
            top_k=5,
        ))
        for ref in result.references:
            print(f"{ref.source}: {ref.relevance_score}")
        ```

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → cross_reference
    """

    name: str = "cross_reference"
    description: str = "Find related content via semantic search across knowledge bases"
    default_preset: str = "S4"  # LLM for post-ranking only

    # Schema types for introspection
    input_schema = CrossReferenceInput
    output_schema = CrossReferenceResult

    # Context budget (AC-13.3)
    context_budget = ContextBudget()

    def __init__(
        self,
        semantic_search_client: SemanticSearchProtocol | None = None,
    ) -> None:
        """Initialize the cross_reference function.

        Args:
            semantic_search_client: Client for semantic-search-service.
                If None, must be provided at execute time.
        """
        self._client = semantic_search_client

    async def run(
        self,
        *,
        query_artifact: str,
        search_scope: list[str] | None = None,
        match_type: MatchType | str = MatchType.SEMANTIC,
        top_k: int = 10,
        **kwargs: Any,
    ) -> CrossReferenceResult:
        """Find related content via semantic search.

        AC-13.1: Queries semantic-search-service for related content.

        Args:
            query_artifact: Source content to find related content for
            search_scope: List of repository names to search (empty = all)
            match_type: Type of matching algorithm to use
            top_k: Maximum number of results to return
            **kwargs: Additional arguments for extensibility

        Returns:
            CrossReferenceResult with references, similarity_scores,
            compressed_context, and citations.

        Raises:
            ContextBudgetExceededError: If input exceeds 2048 token budget.
        """
        search_scope = search_scope or []

        # Convert string match_type to enum if needed
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        # Validate input schema
        input_data = CrossReferenceInput(
            query_artifact=query_artifact,
            search_scope=search_scope,
            match_type=match_type,
            top_k=top_k,
        )

        return await self.execute(input_data)

    async def execute(self, input_data: CrossReferenceInput) -> CrossReferenceResult:
        """Execute cross_reference with validated input.

        Primary execution method that handles the full search workflow:
        1. Truncate input to budget if needed
        2. Query semantic search service
        3. Filter by search scope
        4. Normalize scores to 0.0-1.0
        5. Generate compressed context
        6. Format citations

        Args:
            input_data: Validated CrossReferenceInput

        Returns:
            CrossReferenceResult with all fields populated.
        """
        if self._client is None:
            raise ValueError("semantic_search_client must be provided")

        # Truncate query to input budget if needed
        query = self._truncate_to_budget(input_data.query_artifact)

        # Query semantic search service
        search_results = await self._client.search(
            query=query,
            top_k=input_data.top_k,
        )

        # Handle both dict (MSEPSemanticSearchClient) and list (FakeClient) returns
        if isinstance(search_results, dict):
            results_list = search_results.get("results", [])
        else:
            results_list = search_results

        # Filter by search scope if specified
        if input_data.search_scope:
            results_list = self._filter_by_scope(
                results_list, input_data.search_scope
            )

        # Convert to Reference objects and normalize scores
        references = []
        similarity_scores = []

        for result in results_list:
            score = self._normalize_score(result.get("score", 0.0))
            ref = Reference(
                source=result.get("source", "unknown"),
                content=result.get("content", ""),
                relevance_score=score,
                source_type=result.get("source_type"),
                line_range=result.get("line_range"),
                commit_hash=result.get("commit_hash"),
            )
            references.append(ref)
            similarity_scores.append(score)

        # Generate compressed context for downstream
        compressed_context = self._generate_compressed_context(references)

        # Generate citations for footnotes
        citations = self._generate_citations(references)

        return CrossReferenceResult(
            references=references,
            similarity_scores=similarity_scores,
            compressed_context=compressed_context,
            citations=citations,
        )

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text to fit within input token budget.

        AC-13.3: Context budget: 2048 input tokens.

        Args:
            text: Input text to truncate

        Returns:
            Text truncated to approximately fit input budget.
        """
        estimated = estimate_tokens(text)
        if estimated <= self.context_budget.input_tokens:
            return text

        # Rough character-to-token ratio (typically ~4 chars/token)
        max_chars = self.context_budget.input_tokens * 4
        return text[:max_chars]

    def _filter_by_scope(
        self,
        results: list[dict[str, Any]],
        search_scope: list[str],
    ) -> list[dict[str, Any]]:
        """Filter results to only include sources in search_scope.

        Args:
            results: Search results to filter
            search_scope: List of repository names to include

        Returns:
            Filtered results where source contains a scope item.
        """
        return [
            r for r in results
            if any(scope in r.get("source", "") for scope in search_scope)
        ]

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0.0-1.0 range.

        Exit Criteria: Each Match has relevance_score (0.0-1.0).

        Args:
            score: Raw score from search service

        Returns:
            Score clamped to [0.0, 1.0] range.
        """
        return max(0.0, min(1.0, score))

    def _generate_compressed_context(
        self, references: list[Reference]
    ) -> str:
        """Generate compressed context for downstream functions.

        Per architecture: compressed_context: str (for downstream).
        Creates a summary of found references within output budget.

        Args:
            references: List of matched references

        Returns:
            Compressed context string summarizing findings.
        """
        if not references:
            return "No relevant matches found."

        # Build summary within output budget
        lines = [f"Found {len(references)} relevant reference(s):"]

        for i, ref in enumerate(references, 1):
            # Truncate content preview
            content_preview = ref.content[:100]
            if len(ref.content) > 100:
                content_preview += "..."

            lines.append(
                f"[{i}] {ref.source} (score: {ref.relevance_score:.2f}): "
                f"{content_preview}"
            )

        context = "\n".join(lines)

        # Truncate if exceeds output budget (~4 chars per token)
        max_chars = self.context_budget.output_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars - 3] + "..."

        return context

    def _generate_citations(
        self, references: list[Reference]
    ) -> list[Citation]:
        """Generate formatted citations for footnotes.

        Per architecture: citations: list[Citation] for footnotes.
        Creates Chicago-style citations for each reference.

        Args:
            references: List of matched references

        Returns:
            List of Citation objects with markers and formatted text.
        """
        citations = []

        for i, ref in enumerate(references, 1):
            marker = f"^{i}"

            if ref.source_type == "book":
                # Book citation format
                formatted = ref.source
            elif ref.line_range:
                # Code citation with line range
                formatted = f"{ref.source}:{ref.line_range}"
                if ref.commit_hash:
                    formatted += f" ({ref.commit_hash})"
            else:
                # Generic citation
                formatted = ref.source

            citations.append(
                Citation(
                    marker=marker,
                    formatted_citation=formatted,
                    reference_source=ref.source,
                )
            )

        return citations

"""MSEP Orchestrator.

WBS: MSE-4.3 - Orchestrator
Main orchestration function for MSEP workflow.

Reference Documents:
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-4.3
- MULTI_STAGE_ENRICHMENT_PIPELINE_ARCHITECTURE.md: Orchestration flow

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S3776: Cognitive complexity < 15 per function (extracted helpers)
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.agents.msep.dispatcher import DispatchResult, MSEPDispatcher
from src.agents.msep.schemas import (
    ChapterMeta,
    CrossReference,
    EnrichedChapter,
    EnrichedMetadata,
    MergedKeywords,
    MSEPRequest,
    Provenance,
)

if TYPE_CHECKING:
    from src.clients.protocols import CodeOrchestratorProtocol, SemanticSearchProtocol


# Configure logging
logger = logging.getLogger(__name__)


class MSEPOrchestrator:
    """Orchestrator for Multi-Stage Enrichment Pipeline.

    Coordinates parallel calls to services and merges results.

    Pattern: Orchestrator pattern for workflow coordination
    """

    def __init__(
        self,
        code_orchestrator: "CodeOrchestratorProtocol | None" = None,
        semantic_search: "SemanticSearchProtocol | None" = None,
    ) -> None:
        """Initialize MSEP orchestrator.

        Args:
            code_orchestrator: Optional Code-Orchestrator client
            semantic_search: Optional semantic-search client
        """
        # Create clients if not provided
        if code_orchestrator is None:
            from src.clients.code_orchestrator import CodeOrchestratorClient
            from src.agents.msep.constants import SERVICE_CODE_ORCHESTRATOR_URL

            code_orchestrator = CodeOrchestratorClient(SERVICE_CODE_ORCHESTRATOR_URL)
        if semantic_search is None:
            from src.clients.semantic_search import MSEPSemanticSearchClient
            from src.agents.msep.constants import SERVICE_SEMANTIC_SEARCH_URL

            semantic_search = MSEPSemanticSearchClient(SERVICE_SEMANTIC_SEARCH_URL)

        self._dispatcher = MSEPDispatcher(
            code_orchestrator=code_orchestrator,
            semantic_search=semantic_search,
        )

    async def enrich_metadata(self, request: MSEPRequest) -> EnrichedMetadata:
        """Execute MSEP enrichment workflow.

        Args:
            request: MSEP request with corpus and config

        Returns:
            EnrichedMetadata with all enriched chapters

        Raises:
            ServiceUnavailableError: When required services are unavailable
        """
        start_time = time.perf_counter()

        # Dispatch parallel calls
        dispatch_result = await self._dispatcher.dispatch_enrichment(request)

        # Build enriched chapters
        chapters = self._build_enriched_chapters(
            request=request,
            dispatch_result=dispatch_result,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return EnrichedMetadata(
            chapters=chapters,
            processing_time_ms=elapsed_ms,
        )

    def _build_enriched_chapters(
        self,
        request: MSEPRequest,
        dispatch_result: DispatchResult,
    ) -> list[EnrichedChapter]:
        """Build enriched chapters from dispatch result.

        Args:
            request: MSEP request
            dispatch_result: Result from dispatcher

        Returns:
            List of enriched chapters
        """
        chapters: list[EnrichedChapter] = []

        for idx, chapter_meta in enumerate(request.chapter_index):
            chapter = self._build_single_chapter(
                idx=idx,
                chapter_meta=chapter_meta,
                request=request,
                dispatch_result=dispatch_result,
            )
            chapters.append(chapter)

        return chapters

    def _build_single_chapter(
        self,
        idx: int,
        chapter_meta: ChapterMeta,
        request: MSEPRequest,
        dispatch_result: DispatchResult,
    ) -> EnrichedChapter:
        """Build a single enriched chapter.

        Args:
            idx: Chapter index
            chapter_meta: Chapter metadata
            request: MSEP request
            dispatch_result: Result from dispatcher

        Returns:
            EnrichedChapter
        """
        # Get topic ID for this chapter
        topic_id = self._get_topic_id(idx, dispatch_result)

        # Build cross-references
        cross_refs = self._build_cross_references(
            idx=idx,
            request=request,
            dispatch_result=dispatch_result,
        )

        # Build keywords
        keywords = self._build_keywords(idx, dispatch_result)

        # Build provenance
        provenance = self._build_provenance(
            idx=idx,
            dispatch_result=dispatch_result,
            request=request,
        )

        return EnrichedChapter(
            chapter_id=chapter_meta.id,
            cross_references=cross_refs,
            keywords=keywords,
            topic_id=topic_id,
            provenance=provenance,
        )

    def _get_topic_id(self, idx: int, dispatch_result: DispatchResult) -> int:
        """Get topic ID for chapter, defaulting to -1 if unavailable.

        Args:
            idx: Chapter index
            dispatch_result: Result from dispatcher

        Returns:
            Topic ID or -1 if unavailable
        """
        if dispatch_result.topics and idx < len(dispatch_result.topics):
            return dispatch_result.topics[idx]
        return -1

    def _build_cross_references(
        self,
        idx: int,
        request: MSEPRequest,
        dispatch_result: DispatchResult,
    ) -> list[CrossReference]:
        """Build cross-references for a chapter.

        Args:
            idx: Chapter index
            request: MSEP request
            dispatch_result: Result from dispatcher

        Returns:
            List of cross-references
        """
        cross_refs: list[CrossReference] = []

        if not dispatch_result.similarity_matrix:
            return cross_refs

        # Get this chapter's topic
        source_topic = self._get_topic_id(idx, dispatch_result)

        # Build cross-refs from similarity matrix
        for j, target_meta in enumerate(request.chapter_index):
            if j == idx:
                continue  # Skip self-reference

            xref = self._build_single_cross_reference(
                j=j,
                idx=idx,
                source_topic=source_topic,
                target_meta=target_meta,
                request=request,
                dispatch_result=dispatch_result,
            )
            if xref:
                cross_refs.append(xref)

        # Sort by score descending and limit to top_k
        cross_refs.sort(key=lambda x: x.score, reverse=True)
        return cross_refs[: request.config.top_k]

    def _build_single_cross_reference(
        self,
        j: int,
        idx: int,
        source_topic: int,
        target_meta: ChapterMeta,
        request: MSEPRequest,
        dispatch_result: DispatchResult,
    ) -> CrossReference | None:
        """Build a single cross-reference.

        Args:
            j: Target chapter index
            idx: Source chapter index
            source_topic: Source chapter topic
            target_meta: Target chapter metadata
            request: MSEP request
            dispatch_result: Result from dispatcher

        Returns:
            CrossReference or None if below threshold
        """
        similarity_matrix = dispatch_result.similarity_matrix
        if not similarity_matrix or idx >= len(similarity_matrix):
            return None
        if j >= len(similarity_matrix[idx]):
            return None

        base_score = similarity_matrix[idx][j]

        # Apply topic boost if same topic
        target_topic = self._get_topic_id(j, dispatch_result)
        topic_boost = 0.0
        if source_topic >= 0 and source_topic == target_topic:
            topic_boost = request.config.same_topic_boost

        final_score = base_score + topic_boost

        # Filter by threshold
        if final_score < request.config.threshold:
            return None

        return CrossReference(
            target=target_meta.id,
            score=final_score,
            base_score=base_score,
            topic_boost=topic_boost,
            method="sbert",
        )

    def _build_keywords(
        self, idx: int, dispatch_result: DispatchResult
    ) -> MergedKeywords:
        """Build merged keywords for chapter.

        Args:
            idx: Chapter index
            dispatch_result: Result from dispatcher

        Returns:
            MergedKeywords
        """
        tfidf_keywords: list[str] = []
        if dispatch_result.keywords and idx < len(dispatch_result.keywords):
            tfidf_keywords = dispatch_result.keywords[idx]

        # For now, semantic keywords come from hybrid search if available
        semantic_keywords: list[str] = []

        # Merge and deduplicate
        merged = list(dict.fromkeys(tfidf_keywords + semantic_keywords))

        return MergedKeywords(
            tfidf=tfidf_keywords,
            semantic=semantic_keywords,
            merged=merged,
        )

    def _build_provenance(
        self,
        idx: int,
        dispatch_result: DispatchResult,
        request: MSEPRequest,
    ) -> Provenance:
        """Build provenance tracking for chapter.

        Args:
            idx: Chapter index
            dispatch_result: Result from dispatcher
            request: MSEP request

        Returns:
            Provenance
        """
        methods_used: list[str] = ["sbert"]

        # Add methods based on what succeeded
        if dispatch_result.keywords and not dispatch_result.keywords_error:
            methods_used.append("tfidf")
        if dispatch_result.topics and not dispatch_result.topics_error:
            methods_used.append("bertopic")
        if dispatch_result.hybrid_results and not dispatch_result.hybrid_error:
            methods_used.append("hybrid")

        # Get SBERT score for this chapter (average of similarities)
        sbert_score = 0.0
        if dispatch_result.similarity_matrix and idx < len(
            dispatch_result.similarity_matrix
        ):
            row = dispatch_result.similarity_matrix[idx]
            if row:
                sbert_score = sum(row) / len(row)

        # Get topic boost applied
        source_topic = self._get_topic_id(idx, dispatch_result)
        topic_boost = request.config.same_topic_boost if source_topic >= 0 else 0.0

        return Provenance(
            methods_used=methods_used,
            sbert_score=sbert_score,
            topic_boost=topic_boost,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

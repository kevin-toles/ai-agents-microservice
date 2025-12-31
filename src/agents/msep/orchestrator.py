"""MSEP Orchestrator.

WBS: MSE-4.3 - Orchestrator
WBS: MSE-8.6 - Audit Service Integration

Main orchestration function for MSEP workflow.

Reference Documents:
- MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md: MSE-4.3, MSE-8.6
- MULTI_STAGE_ENRICHMENT_PIPELINE_ARCHITECTURE.md: Orchestration flow

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S3776: Cognitive complexity < 15 per function (extracted helpers)
- #42/#43: Proper async/await patterns
- #2.2: Full type annotations
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.agents.msep.dispatcher import DispatchResult, MSEPDispatcher
from src.agents.msep.exceptions import AuditServiceUnavailableError
from src.agents.msep.schemas import (
    ChapterMeta,
    CrossReference,
    EnrichedChapter,
    EnrichedMetadata,
    MergedKeywords,
    MSEPRequest,
    Provenance,
)
from src.agents.msep.scorers import (
    ConceptOverlapScorer,
    KeywordJaccardScorer,
    SimilarityWeights,
    compute_fused_score,
)


if TYPE_CHECKING:
    from src.clients.protocols import (
        AuditServiceProtocol,
        CodeOrchestratorProtocol,
        SemanticSearchProtocol,
    )


# Configure logging
logger = logging.getLogger(__name__)


class MSEPOrchestrator:
    """Orchestrator for Multi-Stage Enrichment Pipeline.

    WBS: MSE-4.3 - Orchestrator
    WBS: MSE-8.6 - Audit Service Integration

    Coordinates parallel calls to services and merges results.
    Optionally validates cross-references via audit-service.

    Pattern: Orchestrator pattern for workflow coordination
    """

    def __init__(
        self,
        code_orchestrator: CodeOrchestratorProtocol | None = None,
        semantic_search: SemanticSearchProtocol | None = None,
        audit_service: AuditServiceProtocol | None = None,
    ) -> None:
        """Initialize MSEP orchestrator.

        WBS: MSE-8.6.1 - Accepts optional audit_service parameter

        Args:
            code_orchestrator: Optional Code-Orchestrator client
            semantic_search: Optional semantic-search client
            audit_service: Optional Audit-Service client for validation
        """
        # Create clients if not provided
        if code_orchestrator is None:
            from src.agents.msep.constants import SERVICE_CODE_ORCHESTRATOR_URL
            from src.clients.code_orchestrator import CodeOrchestratorClient

            code_orchestrator = CodeOrchestratorClient(SERVICE_CODE_ORCHESTRATOR_URL)
        if semantic_search is None:
            from src.agents.msep.constants import SERVICE_SEMANTIC_SEARCH_URL
            from src.clients.semantic_search import MSEPSemanticSearchClient

            semantic_search = MSEPSemanticSearchClient(SERVICE_SEMANTIC_SEARCH_URL)

        self._dispatcher = MSEPDispatcher(
            code_orchestrator=code_orchestrator,
            semantic_search=semantic_search,
        )
        self._audit_service = audit_service

    async def enrich_metadata(self, request: MSEPRequest) -> EnrichedMetadata:
        """Execute MSEP enrichment workflow.

        WBS: MSE-8.6.2/3 - Audit integration based on config flag

        Args:
            request: MSEP request with corpus and config

        Returns:
            EnrichedMetadata with all enriched chapters and optional audit results

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

        # Run audit validation if enabled (MSE-8.6.2/3)
        audit_result = await self._run_audit_validation(request, chapters)

        return EnrichedMetadata(
            chapters=chapters,
            processing_time_ms=elapsed_ms,
            audit_passed=audit_result.get("passed") if audit_result else None,
            audit_findings=audit_result.get("findings") if audit_result else None,
            audit_best_similarity=audit_result.get("best_similarity") if audit_result else None,
        )

    async def _run_audit_validation(
        self,
        request: MSEPRequest,
        chapters: list[EnrichedChapter],
    ) -> dict[str, Any] | None:
        """Run audit validation if enabled.

        WBS: MSE-8.6.2/3/4 - Audit validation with graceful error handling

        Args:
            request: MSEP request with config
            chapters: Enriched chapters to validate

        Returns:
            Audit result dict or None if disabled/unavailable
        """
        # MSE-8.6.3: Skip if audit validation disabled
        if not request.config.enable_audit_validation:
            return None

        # MSE-8.6.4: Skip if no audit service configured
        if self._audit_service is None:
            logger.warning("Audit validation enabled but no audit service configured")
            return None

        # Skip if no chapters to validate
        if not chapters:
            return None

        try:
            # Build audit request from first chapter (sample validation)
            # In production, this could iterate over all chapters
            code = request.corpus[0] if request.corpus else ""
            references = self._build_audit_references(request, chapters)

            result = await self._audit_service.audit_cross_references(
                code=code,
                references=references,
                threshold=request.config.threshold,
            )

            logger.info(
                "Audit validation complete: passed=%s, best_similarity=%.3f",
                result.get("passed"),
                result.get("best_similarity", 0.0),
            )

            return result

        except AuditServiceUnavailableError as e:
            # MSE-8.6.4: Handle unavailable gracefully
            logger.warning("Audit service unavailable: %s", e.message)
            return None

    def _build_audit_references(
        self,
        request: MSEPRequest,
        _chapters: list[EnrichedChapter],
    ) -> list[dict[str, Any]]:
        """Build audit references from request data.

        Args:
            request: MSEP request with corpus
            _chapters: Enriched chapters (reserved for future use)

        Returns:
            List of reference dicts for audit service
        """
        references: list[dict[str, Any]] = []

        for idx, chapter_meta in enumerate(request.chapter_index):
            content = request.corpus[idx] if idx < len(request.corpus) else ""
            references.append({
                "chapter_id": chapter_meta.id,
                "title": chapter_meta.title,
                "content": content,
            })

        return references

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

        # Build similar chapters (cross-references)
        similar_chapters = self._build_similar_chapters(
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

        # Get topic name if available
        topic_name = self._get_topic_name(topic_id, dispatch_result)

        # Get graph relationships if available
        graph_relationships = self._get_graph_relationships(
            chapter_meta.id, dispatch_result
        )

        # Get original content from corpus and summary from chapter_meta
        content = request.corpus[idx] if idx < len(request.corpus) else ""
        summary = chapter_meta.summary or content

        return EnrichedChapter(
            book=chapter_meta.book,
            chapter=chapter_meta.chapter,
            title=chapter_meta.title,
            chapter_id=chapter_meta.id,
            summary=summary,
            content=content,
            similar_chapters=similar_chapters,
            keywords=keywords,
            topic_id=topic_id if topic_id >= 0 else None,
            topic_name=topic_name,
            graph_relationships=graph_relationships,
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

    def _get_topic_name(
        self, topic_id: int, dispatch_result: DispatchResult
    ) -> str | None:
        """Get human-readable topic name for a topic ID.

        Args:
            topic_id: Topic ID
            dispatch_result: Result from dispatcher

        Returns:
            Topic name or None if unavailable
        """
        if topic_id < 0 or not dispatch_result.topic_info:
            return None

        # Find topic info with matching ID
        for topic_info in dispatch_result.topic_info:
            if topic_info.get("topic_id") == topic_id:
                return topic_info.get("name") or topic_info.get("label")

        return f"Topic {topic_id}"

    def _get_graph_relationships(
        self, chapter_id: str, dispatch_result: DispatchResult
    ) -> list[str]:
        """Get graph relationships for a chapter.

        Args:
            chapter_id: Chapter identifier
            dispatch_result: Result from dispatcher

        Returns:
            List of relationship strings
        """
        if not dispatch_result.hybrid_results:
            return []

        # Handle both dict formats: {chapter_id: [rels]} or {relationships: {chapter_id: [rels]}}
        relationships = dispatch_result.hybrid_results
        if isinstance(relationships, dict):
            if "relationships" in relationships:
                relationships = relationships["relationships"]

            if chapter_id in relationships:
                rels = relationships[chapter_id]
                if isinstance(rels, list):
                    return [str(r) for r in rels]

        return []

    def _build_similar_chapters(
        self,
        idx: int,
        request: MSEPRequest,
        dispatch_result: DispatchResult,
    ) -> list[CrossReference]:
        """Build similar chapters list for a chapter.

        Args:
            idx: Chapter index
            request: MSEP request
            dispatch_result: Result from dispatcher

        Returns:
            List of similar chapters (CrossReference objects)
        """
        similar: list[CrossReference] = []

        if dispatch_result.similarity_matrix is None or (
            hasattr(dispatch_result.similarity_matrix, 'size') and dispatch_result.similarity_matrix.size == 0
        ):
            return similar

        # Get this chapter's topic
        source_topic = self._get_topic_id(idx, dispatch_result)

        # Build similar chapters from similarity matrix
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
                similar.append(xref)

        # Sort by score descending and limit to top_k (0 = unlimited)
        similar.sort(key=lambda x: x.score, reverse=True)
        if request.config.top_k > 0:
            return similar[: request.config.top_k]
        return similar

    def _build_single_cross_reference(
        self,
        j: int,
        idx: int,
        source_topic: int,
        target_meta: ChapterMeta,
        request: MSEPRequest,
        dispatch_result: DispatchResult,
    ) -> CrossReference | None:
        """Build a single cross-reference using multi-signal fusion.

        EEP-3.4.1 Update: Uses compute_fused_score() for multi-signal fusion.

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
        # Handle numpy arrays - check for None or empty
        if similarity_matrix is None:
            return None
        if hasattr(similarity_matrix, "size") and similarity_matrix.size == 0:
            return None
        if idx >= len(similarity_matrix):
            return None
        if j >= len(similarity_matrix[idx]):
            return None

        # Get SBERT base score
        sbert_score = float(similarity_matrix[idx][j])

        # Compute topic match
        target_topic = self._get_topic_id(j, dispatch_result)
        topic_match = source_topic >= 0 and source_topic == target_topic

        # Compute keyword Jaccard (EEP-3.3)
        keyword_jaccard = 0.0
        matched_concepts: list[str] = []
        if dispatch_result.keywords:
            source_keywords = (
                dispatch_result.keywords[idx]
                if idx < len(dispatch_result.keywords)
                else []
            )
            target_keywords = (
                dispatch_result.keywords[j]
                if j < len(dispatch_result.keywords)
                else []
            )
            keyword_scorer = KeywordJaccardScorer()
            keyword_jaccard = keyword_scorer.compute(source_keywords, target_keywords)

            # Use keywords as proxy for concepts (EEP-3.2)
            concept_scorer = ConceptOverlapScorer()
            concept_result = concept_scorer.compute(source_keywords, target_keywords)
            concept_overlap = concept_result.score
            matched_concepts = concept_result.matched_concepts
        else:
            concept_overlap = 0.0

        # Compute fused score using EEP-3 formula
        weights = SimilarityWeights()
        final_score = compute_fused_score(
            sbert_sim=sbert_score,
            codebert_sim=None,  # CodeBERT not yet integrated
            concept_jaccard=concept_overlap,
            keyword_jaccard=keyword_jaccard,
            topic_match=topic_match,
            weights=weights,
        )

        # Calculate topic boost value for schema
        topic_boost = weights.topic_boost if topic_match else 0.0

        # Filter by threshold
        if final_score < request.config.threshold:
            return None

        return CrossReference(
            target=target_meta.id,
            score=final_score,
            base_score=sbert_score,
            topic_boost=topic_boost,
            method="multi-signal",
            concept_overlap=concept_overlap,
            keyword_jaccard=keyword_jaccard,
            matched_concepts=matched_concepts,
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
        similarity_matrix = dispatch_result.similarity_matrix
        if (
            similarity_matrix is not None
            and not (hasattr(similarity_matrix, 'size') and similarity_matrix.size == 0)
            and idx < len(similarity_matrix)
        ):
            row = similarity_matrix[idx]
            if len(row) > 0:
                sbert_score = float(sum(row) / len(row))

        # Get topic boost applied
        source_topic = self._get_topic_id(idx, dispatch_result)
        topic_boost = request.config.same_topic_boost if source_topic >= 0 else 0.0

        return Provenance(
            methods_used=methods_used,
            sbert_score=sbert_score,
            topic_boost=topic_boost,
            timestamp=datetime.now(UTC).isoformat(),
        )

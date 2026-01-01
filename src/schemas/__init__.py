"""Pydantic Schemas Package.

This package contains Pydantic models for:
- Agent function inputs and outputs
- Citation and provenance tracking
- Analysis findings and violations
- Pipeline state and handoff
- Graph reference models (WBS-AGT22)

All schemas support JSON Schema export via model_json_schema().

Pattern: Typed Data Transfer Objects (DTOs)
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Pydantic Schemas
"""

# Citation and provenance schemas (AC-4.1, AC-4.2, AC-4.3)
# Analysis schemas (AC-4.4)
from src.schemas.analysis import (
    SEVERITY_ORDER,
    AnalysisResult,
    Finding,
    Severity,
    ValidationResult,
    Violation,
)
from src.schemas.citations import (
    Citation,
    CitedContent,
    SourceMetadata,
    SourceType,
    create_citation_from_retrieval,
    merge_citations,
)
# Graph reference models (WBS-AGT22)
from src.schemas.graph_models import (
    ChapterReference,
    CodeFileReference,
    CodeQueryResult,
    Concept,
    ConceptQueryResult,
    GraphReference,
    PatternQueryResult,
    PatternRelationship,
)
# Passage models (WBS-AGT23)
from src.schemas.passage_models import (
    BookPassage,
    ChapterPassageRef,
    PassageFilter,
    PassageMetadata,
    PassageSearchResult,
)
# Retrieval models (WBS-AGT24)
from src.schemas.retrieval_models import (
    RetrievalItem,
    RetrievalOptions,
    RetrievalResult,
    RetrievalScope,
    SourceType as RetrievalSourceType,
)


__all__: list[str] = [
    "SEVERITY_ORDER",
    "AnalysisResult",
    "BookPassage",
    "ChapterPassageRef",
    "ChapterReference",
    "Citation",
    "CitedContent",
    "CodeFileReference",
    "CodeQueryResult",
    "Concept",
    "ConceptQueryResult",
    "Finding",
    "GraphReference",
    "PassageFilter",
    "PassageMetadata",
    "PassageSearchResult",
    "PatternQueryResult",
    "PatternRelationship",
    # Retrieval models (WBS-AGT24)
    "RetrievalItem",
    "RetrievalOptions",
    "RetrievalResult",
    "RetrievalScope",
    "RetrievalSourceType",
    # Analysis schemas
    "Severity",
    "SourceMetadata",
    # Citation schemas
    "SourceType",
    "ValidationResult",
    "Violation",
    "create_citation_from_retrieval",
    "merge_citations",
]

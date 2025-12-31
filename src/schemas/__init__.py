"""Pydantic Schemas Package.

This package contains Pydantic models for:
- Agent function inputs and outputs
- Citation and provenance tracking
- Analysis findings and violations
- Pipeline state and handoff

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


__all__: list[str] = [
    "SEVERITY_ORDER",
    "AnalysisResult",
    "Citation",
    "CitedContent",
    "Finding",
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

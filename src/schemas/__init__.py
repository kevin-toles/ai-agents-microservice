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
from src.schemas.citations import (
    SourceType,
    SourceMetadata,
    Citation,
    CitedContent,
    create_citation_from_retrieval,
    merge_citations,
)

# Analysis schemas (AC-4.4)
from src.schemas.analysis import (
    Severity,
    Finding,
    Violation,
    AnalysisResult,
    ValidationResult,
    SEVERITY_ORDER,
)

__all__: list[str] = [
    # Citation schemas
    "SourceType",
    "SourceMetadata",
    "Citation",
    "CitedContent",
    "create_citation_from_retrieval",
    "merge_citations",
    # Analysis schemas
    "Severity",
    "Finding",
    "Violation",
    "AnalysisResult",
    "ValidationResult",
    "SEVERITY_ORDER",
]

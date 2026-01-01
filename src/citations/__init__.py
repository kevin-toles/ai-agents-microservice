"""Citation Management Package.

This package handles citation tracking, formatting, and audit:
- CitationManager for tracking sources through pipelines
- ChicagoFormatter for Chicago-style footnote generation
- Citation audit records for audit-service integration
- CodeCitation for code reference citations (WBS-AGT21)
- GraphCitation for Neo4j graph reference citations (WBS-AGT22)
- BookCitation for book passage citations (WBS-AGT23)

Every piece of generated content must be traceable to its sources.

Pattern: Provenance Tracking
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
"""

from src.citations.book_citation import (
    BookCitation,
    citation_from_dict,
    passage_to_citation,
    passages_to_citations,
)
from src.citations.code_citation import (
    CodeCitation,
    code_context_to_citations,
    format_citations_for_prompt,
)
from src.citations.formatter import ChicagoFormatter
from src.citations.graph_citation import (
    GraphCitation,
    code_file_to_citation,
    code_files_to_citations,
    concept_to_citation,
    concepts_to_citations,
    format_graph_citations_for_prompt,
    pattern_to_citation,
    patterns_to_citations,
)
from src.citations.manager import CitationManager
# Mixed citations (WBS-AGT24)
from src.citations.mixed_citation import (
    MixedCitation,
    SourceType as MixedSourceType,
    citations_from_retrieval_items,
    from_book_citation,
    from_code_citation,
    from_graph_citation,
    from_retrieval_item,
)


__all__ = [
    "BookCitation",
    "ChicagoFormatter",
    "CitationManager",
    "CodeCitation",
    "GraphCitation",
    "MixedCitation",
    "MixedSourceType",
    "citation_from_dict",
    "citations_from_retrieval_items",
    "code_context_to_citations",
    "code_file_to_citation",
    "code_files_to_citations",
    "concept_to_citation",
    "concepts_to_citations",
    "format_citations_for_prompt",
    "format_graph_citations_for_prompt",
    "from_book_citation",
    "from_code_citation",
    "from_graph_citation",
    "from_retrieval_item",
    "passage_to_citation",
    "passages_to_citations",
    "pattern_to_citation",
    "patterns_to_citations",
]


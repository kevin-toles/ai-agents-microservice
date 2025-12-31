"""Audit schemas for citation tracking.

Implements AC-17.3, AC-17.4 from WBS-AGT17.

Models:
- CitationAuditRecord: Individual citation audit entry
- CitationAuditBatch: Batch of citation audit records

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Citation Flow
Audit record structure:
- conversation_id: Conversation identifier
- message_id: Message identifier
- citations_used: List of citation markers
- retrieval_scores: Similarity scores from retrieval
- timestamp: Audit record timestamp
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# =============================================================================
# AC-17.3, AC-17.4: CitationAuditRecord
# =============================================================================

class CitationAuditRecord(BaseModel):
    """Individual citation audit record.

    Captures citation usage for audit purposes. Records where and how
    a citation was used within a conversation.

    Attributes:
        conversation_id: Unique identifier for the conversation
        message_id: Unique identifier for the message containing citation
        source_id: Unique identifier for the source (e.g., "book:fowler-peaa-2002")
        source_type: Type of source (book, code, schema, internal_doc)
        retrieval_score: Semantic search relevance score (0.0-1.0)
        usage_context: Description of how the citation was used
        marker: Citation footnote marker number
        timestamp: When the audit record was created

    Example:
        >>> record = CitationAuditRecord(
        ...     conversation_id="conv-123",
        ...     message_id="msg-456",
        ...     source_id="book:fowler-peaa-2002",
        ...     source_type="book",
        ...     retrieval_score=0.89,
        ...     usage_context="Referenced in Repository pattern explanation",
        ...     marker=1,
        ... )
    """

    conversation_id: str = Field(
        ...,
        description="Unique identifier for the conversation",
    )
    message_id: str = Field(
        ...,
        description="Unique identifier for the message containing citation",
    )
    source_id: str = Field(
        ...,
        description="Unique identifier for the source",
    )
    source_type: str = Field(
        ...,
        description="Type of source: book, code, schema, internal_doc",
    )
    retrieval_score: float = Field(
        ...,
        description="Semantic search relevance score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    usage_context: str = Field(
        ...,
        description="Description of how the citation was used",
    )
    marker: int | None = Field(
        default=None,
        description="Citation footnote marker number",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the audit record was created",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "conv-123",
                    "message_id": "msg-456",
                    "source_id": "book:fowler-peaa-2002",
                    "source_type": "book",
                    "retrieval_score": 0.89,
                    "usage_context": "Referenced in Repository pattern explanation",
                    "marker": 1,
                    "timestamp": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }


# =============================================================================
# CitationAuditBatch
# =============================================================================

class CitationAuditBatch(BaseModel):
    """Batch of citation audit records for bulk submission.

    Groups multiple citation audit records for efficient submission
    to the audit service.

    Attributes:
        conversation_id: Shared conversation identifier
        message_id: Shared message identifier
        records: List of individual citation audit records

    Properties:
        total_count: Number of records in the batch

    Example:
        >>> from src.schemas.audit import CitationAuditRecord, CitationAuditBatch
        >>> records = [
        ...     CitationAuditRecord(
        ...         conversation_id="conv-123",
        ...         message_id="msg-456",
        ...         source_id=f"book:source-{i}",
        ...         source_type="book",
        ...         retrieval_score=0.8,
        ...         usage_context=f"Context {i}",
        ...     )
        ...     for i in range(3)
        ... ]
        >>> batch = CitationAuditBatch(
        ...     conversation_id="conv-123",
        ...     message_id="msg-456",
        ...     records=records,
        ... )
        >>> batch.total_count
        3
    """

    conversation_id: str = Field(
        ...,
        description="Shared conversation identifier for all records",
    )
    message_id: str = Field(
        ...,
        description="Shared message identifier for all records",
    )
    records: list[CitationAuditRecord] = Field(
        default_factory=list,
        description="List of citation audit records",
    )

    @property
    def total_count(self) -> int:
        """Return the total number of records in the batch.

        Returns:
            Number of citation audit records
        """
        return len(self.records)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "conv-123",
                    "message_id": "msg-456",
                    "records": [
                        {
                            "conversation_id": "conv-123",
                            "message_id": "msg-456",
                            "source_id": "book:fowler-peaa-2002",
                            "source_type": "book",
                            "retrieval_score": 0.89,
                            "usage_context": "Pattern reference",
                            "marker": 1,
                        }
                    ],
                }
            ]
        }
    }


__all__ = [
    "CitationAuditBatch",
    "CitationAuditRecord",
]

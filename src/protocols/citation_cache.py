#!/usr/bin/env python3
"""
Citation Cache - Session-Scoped Circular Buffer for Citations

Tracks and validates citations across protocol discussion rounds without
consuming excessive LLM context. Uses FIFO circular buffer pattern
(like CCTV - oldest gets overwritten when full).

Architecture:
- CONFIRMED: Citations validated against cross-reference evidence
- PENDING: Citations awaiting validation (LLM claimed but not yet verified)

Lifecycle: Created at protocol start → Destroyed at protocol end

Reference: ARCHITECTURE_DECISION_RECORD.md - Citation Traceability
"""

from __future__ import annotations

import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Citation Data Models
# =============================================================================

@dataclass
class Citation:
    """A single citation extracted from LLM output.
    
    Chicago-style format: [^N] Author, Title, Chapter, Page
    """
    marker: str                     # "[^1]", "[^2]", etc.
    source_type: str                # "book", "code", "textbook", "web"
    source_id: str                  # "Building Microservices", "kubernetes/kubernetes"
    chapter: Optional[str] = None   # "Ch. 4", "Section 3.2"
    page: Optional[int] = None      # 87
    content_claim: str = ""         # What LLM claims this cites
    relevance_score: float = 0.0    # 0.0-1.0 similarity score
    confirmed: bool = False         # Cross-ref validated?
    confirmed_by: str = ""          # "cross_reference" | "audit_service"
    round_num: int = 0              # Which round added this
    role: str = ""                  # Which LLM role created this
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_footnote(self) -> str:
        """Generate Chicago-style footnote text."""
        parts = [self.source_id]
        if self.chapter:
            parts.append(self.chapter)
        if self.page:
            parts.append(f"p. {self.page}")
        return f"{self.marker} {', '.join(parts)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "marker": self.marker,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "chapter": self.chapter,
            "page": self.page,
            "content_claim": self.content_claim[:200] if self.content_claim else "",
            "relevance_score": self.relevance_score,
            "confirmed": self.confirmed,
            "confirmed_by": self.confirmed_by,
            "round_num": self.round_num,
            "role": self.role,
        }


# =============================================================================
# Citation Cache
# =============================================================================

class CitationCache:
    """Session-scoped circular buffer for citation tracking.
    
    Features:
    - Circular buffer: Oldest citations auto-evicted when max_size reached
    - Pending queue: Unvalidated citations awaiting cross-reference match
    - Context-efficient: Only recent citations injected into prompts
    - Audit-ready: Full history for post-protocol validation
    
    Usage:
        cache = CitationCache(max_size=100)
        
        # Add citation from LLM output
        cache.add_pending(Citation(marker="[^1]", source_id="Building Microservices", ...))
        
        # Validate against cross-reference evidence
        cache.validate_pending("[^1]", evidence, threshold=0.7)
        
        # Get summary for next prompt (context-efficient)
        summary = cache.get_prompt_summary(max_citations=10)
        
        # Export for Chicago footnotes
        footnotes = cache.to_footnotes()
    """
    
    # Regex patterns for extracting citations from LLM output
    CITATION_MARKER_PATTERN = re.compile(r'\[\^(\d+)\]')
    CITATION_FULL_PATTERN = re.compile(
        r'\[\^(\d+)\]:\s*([^,\n]+)(?:,\s*([^,\n]+))?(?:,\s*(?:p\.?\s*)?(\d+))?'
    )
    
    def __init__(self, max_size: int = 100, session_id: Optional[str] = None):
        """Initialize citation cache.
        
        Args:
            max_size: Maximum confirmed citations (circular buffer)
            session_id: Optional session ID (auto-generated if not provided)
        """
        self._confirmed: deque[Citation] = deque(maxlen=max_size)
        self._pending: Dict[str, Citation] = {}  # marker -> Citation
        self._all_markers: Set[str] = set()  # Track all seen markers
        self._session_id = session_id or str(uuid.uuid4())
        self._created_at = datetime.now()
        self._max_size = max_size
        
    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id
    
    @property
    def confirmed_count(self) -> int:
        """Number of confirmed citations."""
        return len(self._confirmed)
    
    @property
    def pending_count(self) -> int:
        """Number of pending citations."""
        return len(self._pending)
    
    # =========================================================================
    # Citation Extraction
    # =========================================================================
    
    def extract_citations_from_text(
        self, 
        text: str, 
        round_num: int = 0, 
        role: str = ""
    ) -> List[Citation]:
        """Extract citation markers from LLM output text.
        
        Supports formats:
        - Simple: [^1], [^2]
        - Full: [^1]: Building Microservices, Ch. 4, p. 87
        - Inline: "According to [^1], the pattern..."
        
        Args:
            text: LLM output text
            round_num: Current round number
            role: LLM role that produced this text
            
        Returns:
            List of extracted Citation objects
        """
        citations = []
        
        # First, try to extract full citation definitions
        for match in self.CITATION_FULL_PATTERN.finditer(text):
            marker = f"[^{match.group(1)}]"
            source_id = match.group(2).strip() if match.group(2) else ""
            chapter = match.group(3).strip() if match.group(3) else None
            page = int(match.group(4)) if match.group(4) else None
            
            if marker not in self._all_markers:
                citation = Citation(
                    marker=marker,
                    source_type=self._infer_source_type(source_id),
                    source_id=source_id,
                    chapter=chapter,
                    page=page,
                    round_num=round_num,
                    role=role,
                )
                citations.append(citation)
                self._all_markers.add(marker)
        
        # Then, find any simple markers without definitions
        for match in self.CITATION_MARKER_PATTERN.finditer(text):
            marker = f"[^{match.group(1)}]"
            if marker not in self._all_markers:
                # Extract surrounding context as content_claim
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                citation = Citation(
                    marker=marker,
                    source_type="unknown",
                    source_id="",
                    content_claim=context,
                    round_num=round_num,
                    role=role,
                )
                citations.append(citation)
                self._all_markers.add(marker)
        
        return citations
    
    def _infer_source_type(self, source_id: str) -> str:
        """Infer source type from source ID."""
        source_lower = source_id.lower()
        
        if "/" in source_id:  # GitHub-style: owner/repo
            return "code"
        elif any(kw in source_lower for kw in ["http", "www", ".com", ".org"]):
            return "web"
        elif any(kw in source_lower for kw in ["textbook", "manual", "guide"]):
            return "textbook"
        else:
            return "book"
    
    # =========================================================================
    # Add/Validate Citations
    # =========================================================================
    
    def add_pending(self, citation: Citation) -> None:
        """Add citation to pending queue (awaiting validation).
        
        Args:
            citation: Citation to add
        """
        self._pending[citation.marker] = citation
        self._all_markers.add(citation.marker)
    
    def add_confirmed(self, citation: Citation) -> None:
        """Add pre-validated citation directly to confirmed buffer.
        
        Args:
            citation: Citation to add (must have confirmed=True)
        """
        citation.confirmed = True
        self._confirmed.append(citation)
        self._all_markers.add(citation.marker)
        
        # Remove from pending if it was there
        self._pending.pop(citation.marker, None)
    
    def validate_pending(
        self, 
        marker: str, 
        evidence: Dict[str, Any],
        threshold: float = 0.7,
        confirmed_by: str = "cross_reference"
    ) -> bool:
        """Validate a pending citation against cross-reference evidence.
        
        Args:
            marker: Citation marker (e.g., "[^1]")
            evidence: Cross-reference evidence dict with source info
            threshold: Minimum relevance score to confirm
            confirmed_by: Validation source identifier
            
        Returns:
            True if citation was confirmed, False otherwise
        """
        if marker not in self._pending:
            return False
        
        citation = self._pending[marker]
        
        # Try to match against evidence sources
        best_match = self._find_best_match(citation, evidence)
        
        if best_match and best_match["score"] >= threshold:
            citation.confirmed = True
            citation.confirmed_by = confirmed_by
            citation.relevance_score = best_match["score"]
            citation.source_id = best_match.get("source_id", citation.source_id)
            citation.source_type = best_match.get("source_type", citation.source_type)
            citation.chapter = best_match.get("chapter", citation.chapter)
            
            # Move from pending to confirmed
            del self._pending[marker]
            self._confirmed.append(citation)
            return True
        
        return False
    
    def validate_all_pending(
        self, 
        cross_reference_evidence: Dict[str, Any],
        threshold: float = 0.7
    ) -> Dict[str, bool]:
        """Validate all pending citations against cross-reference evidence.
        
        Args:
            cross_reference_evidence: Full cross-reference evidence dict
            threshold: Minimum relevance score to confirm
            
        Returns:
            Dict mapping marker -> validation result
        """
        results = {}
        
        # Flatten all evidence into searchable format
        all_evidence = self._flatten_evidence(cross_reference_evidence)
        
        for marker in list(self._pending.keys()):
            results[marker] = self.validate_pending(
                marker, 
                {"all": all_evidence}, 
                threshold
            )
        
        return results
    
    def _find_best_match(
        self, 
        citation: Citation, 
        evidence: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find best matching source in evidence for a citation.
        
        Uses simple text matching (full implementation would use embeddings).
        """
        best_match = None
        best_score = 0.0
        
        # Search through all evidence sources
        for source_type, items in evidence.items():
            if not isinstance(items, list):
                continue
                
            for item in items:
                score = self._calculate_match_score(citation, item)
                if score > best_score:
                    best_score = score
                    best_match = {
                        "score": score,
                        "source_id": item.get("book", item.get("title", "")),
                        "source_type": source_type,
                        "chapter": item.get("chapter", ""),
                    }
        
        return best_match
    
    def _calculate_match_score(
        self, 
        citation: Citation, 
        evidence_item: Dict[str, Any]
    ) -> float:
        """Calculate match score between citation and evidence item.
        
        Simple text matching - production would use embeddings.
        """
        score = 0.0
        
        # Match source ID
        source = evidence_item.get("book", evidence_item.get("title", "")).lower()
        if citation.source_id and citation.source_id.lower() in source:
            score += 0.5
        elif source and citation.content_claim and source in citation.content_claim.lower():
            score += 0.3
        
        # Match chapter
        chapter = str(evidence_item.get("chapter", "")).lower()
        if citation.chapter and citation.chapter.lower() in chapter:
            score += 0.2
        
        # Content relevance (if available)
        content = evidence_item.get("content", "").lower()
        if citation.content_claim:
            # Count overlapping words
            claim_words = set(citation.content_claim.lower().split())
            content_words = set(content.split())
            overlap = len(claim_words & content_words)
            if overlap > 5:
                score += min(0.3, overlap * 0.02)
        
        return min(1.0, score)
    
    def _flatten_evidence(
        self, 
        cross_reference_evidence: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Flatten nested cross-reference evidence into searchable list."""
        flattened = []
        
        for query, evidence in cross_reference_evidence.items():
            if not isinstance(evidence, dict):
                continue
            
            for source_type in ["textbooks", "qdrant", "neo4j", "code_orchestrator", "code_reference"]:
                items = evidence.get(source_type, [])
                if isinstance(items, list):
                    for item in items:
                        item["_source_type"] = source_type
                        flattened.append(item)
        
        return flattened
    
    # =========================================================================
    # Context Generation
    # =========================================================================
    
    def get_prompt_summary(self, max_citations: int = 10) -> str:
        """Get compact citation summary for LLM prompt injection.
        
        Only includes recent confirmed citations to save context.
        
        Args:
            max_citations: Maximum citations to include
            
        Returns:
            Formatted citation summary string
        """
        if not self._confirmed:
            return ""
        
        recent = list(self._confirmed)[-max_citations:]
        
        lines = ["## Confirmed Citations (use these for references)"]
        for c in recent:
            status = "✓" if c.confirmed else "?"
            lines.append(f"- {c.marker}: {c.source_id}"
                        f"{f', {c.chapter}' if c.chapter else ''}"
                        f"{f', p.{c.page}' if c.page else ''} {status}")
        
        return "\n".join(lines)
    
    def get_validation_summary(self) -> str:
        """Get summary of citation validation status."""
        confirmed = len(self._confirmed)
        pending = len(self._pending)
        total = confirmed + pending
        
        return (f"Citations: {confirmed}/{total} confirmed, "
                f"{pending} pending validation")
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def to_footnotes(self) -> List[Dict[str, Any]]:
        """Export confirmed citations for Chicago-style footnote generation.
        
        Returns:
            List of citation dicts for audit_generate_footnotes
        """
        return [
            {
                "marker": c.marker,
                "source_id": c.source_id,
                "source_type": c.source_type,
            }
            for c in self._confirmed
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export full cache state for API response."""
        return {
            "session_id": self._session_id,
            "confirmed": [c.to_dict() for c in self._confirmed],
            "pending": [c.to_dict() for c in self._pending.values()],
            "summary": {
                "confirmed_count": len(self._confirmed),
                "pending_count": len(self._pending),
                "total_markers_seen": len(self._all_markers),
            }
        }
    
    def get_confirmed_citations(self) -> List[Citation]:
        """Get all confirmed citations."""
        return list(self._confirmed)
    
    def get_pending_citations(self) -> List[Citation]:
        """Get all pending citations."""
        return list(self._pending.values())

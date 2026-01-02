"""Information Request Extraction from LLM Analyses.

WBS Reference: WBS-KB2 - Information Request Detection
Tasks: KB2.3, KB2.4 - Implement extract_information_requests() parser and priority scoring
Acceptance Criteria:
- AC-KB2.2: extract_information_requests() parses LLM analysis for requests
- AC-KB2.4: Requests specify source_types (code, books, textbooks, graph)
- AC-KB2.5: Requests have priority based on disagreement severity
- AC-KB2.6: Zero requests returned when agreement_score > threshold

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity < 15 via helper methods
- Proper error handling for malformed JSON
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.discussion.models import (
    DEFAULT_PRIORITY,
    VALID_PRIORITIES,
    VALID_SOURCE_TYPES,
    InformationRequest,
)


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_AGREEMENT_THRESHOLD = 0.85
_CONST_HIGH_DISAGREEMENT_GAP = 0.3
_CONST_MEDIUM_DISAGREEMENT_GAP = 0.15
_CONST_LOW_AGREEMENT_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


# =============================================================================
# Priority Calculation (AC-KB2.5)
# =============================================================================


def calculate_priority_from_disagreement(
    confidence_gap: float,
    agreement_score: float,
) -> str:
    """Calculate request priority based on disagreement severity.
    
    AC-KB2.5: Priority correlates with confidence gap.
    
    Args:
        confidence_gap: Difference between highest and lowest confidence (0.0-1.0)
        agreement_score: Overall agreement score (0.0-1.0)
        
    Returns:
        Priority string: "high", "medium", or "low"
    """
    # Low agreement always boosts to at least medium
    if agreement_score < _CONST_LOW_AGREEMENT_THRESHOLD:
        if confidence_gap >= _CONST_MEDIUM_DISAGREEMENT_GAP:
            return "high"
        return "medium"
    
    # Standard priority based on confidence gap
    if confidence_gap >= _CONST_HIGH_DISAGREEMENT_GAP:
        return "high"
    if confidence_gap >= _CONST_MEDIUM_DISAGREEMENT_GAP:
        return "medium"
    return "low"


# =============================================================================
# JSON Extraction Helpers
# =============================================================================


def _extract_json_block(text: str) -> str | None:
    """Extract JSON code block from text."""
    # Match ```json ... ``` blocks
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _parse_json_requests(json_str: str) -> list[dict[str, Any]]:
    """Parse JSON string containing information_requests."""
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "information_requests" in data:
            requests = data["information_requests"]
            if isinstance(requests, list):
                return requests
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse JSON: %s", e)
    return []


def _validate_request_dict(req: dict[str, Any]) -> bool:
    """Validate a request dictionary has required fields."""
    if not isinstance(req, dict):
        return False
    if "query" not in req or not req["query"]:
        return False
    return True


def _filter_source_types(source_types: list[str]) -> list[str]:
    """Filter source_types to only valid values."""
    if not source_types:
        return list(VALID_SOURCE_TYPES)
    
    valid = [st for st in source_types if st in VALID_SOURCE_TYPES]
    return valid if valid else list(VALID_SOURCE_TYPES)


def _normalize_priority(priority: str | None) -> str:
    """Normalize priority to valid value."""
    if priority and priority.lower() in VALID_PRIORITIES:
        return priority.lower()
    return DEFAULT_PRIORITY


def _dict_to_information_request(req: dict[str, Any]) -> InformationRequest | None:
    """Convert validated dict to InformationRequest."""
    if not _validate_request_dict(req):
        return None
    
    return InformationRequest(
        query=req["query"],
        source_types=_filter_source_types(req.get("source_types", [])),
        priority=_normalize_priority(req.get("priority")),
        reasoning=req.get("reasoning", ""),
    )


# =============================================================================
# Markdown Extraction Helpers
# =============================================================================


def _extract_markdown_requests(text: str) -> list[dict[str, Any]]:
    """Extract information requests from markdown format.
    
    Looks for patterns like:
    1. **Query:** "..."
       - **Source Types:** code, books
       - **Priority:** high
    """
    requests: list[dict[str, Any]] = []
    
    # Pattern for numbered items with Query field
    pattern = r'\d+\.\s*\*\*Query:\*\*\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern, text)
    
    for query in matches:
        # Try to find associated source types and priority
        # This is a simplified extraction - JSON is preferred
        requests.append({
            "query": query,
            "source_types": list(VALID_SOURCE_TYPES),
            "priority": DEFAULT_PRIORITY,
        })
    
    return requests


# =============================================================================
# Main Extraction Functions (AC-KB2.2)
# =============================================================================


def extract_information_requests(analysis: str | None) -> list[InformationRequest]:
    """Extract information requests from LLM analysis.
    
    AC-KB2.2: Parses LLM analysis for structured information requests.
    Supports both JSON and markdown formats, preferring JSON.
    
    Args:
        analysis: The LLM analysis text containing information requests
        
    Returns:
        List of InformationRequest objects (empty if none found or invalid input)
    """
    if not analysis or not isinstance(analysis, str):
        return []
    
    # Prefer JSON format
    json_block = _extract_json_block(analysis)
    if json_block:
        request_dicts = _parse_json_requests(json_block)
        if request_dicts:
            requests = []
            for req_dict in request_dicts:
                req = _dict_to_information_request(req_dict)
                if req:
                    requests.append(req)
            return requests
    
    # Fallback to markdown parsing
    markdown_dicts = _extract_markdown_requests(analysis)
    if markdown_dicts:
        requests = []
        for req_dict in markdown_dicts:
            req = _dict_to_information_request(req_dict)
            if req:
                requests.append(req)
        return requests
    
    return []


def extract_information_requests_with_agreement(
    analysis: str,
    agreement_score: float,
    threshold: float = _CONST_DEFAULT_AGREEMENT_THRESHOLD,
) -> list[InformationRequest]:
    """Extract information requests, returning empty if agreement exceeds threshold.
    
    AC-KB2.6: Zero requests returned when agreement_score > threshold.
    
    Args:
        analysis: The LLM analysis text
        agreement_score: Current agreement score (0.0-1.0)
        threshold: Agreement threshold above which no requests are returned
        
    Returns:
        List of InformationRequest objects (empty if agreement >= threshold)
    """
    if agreement_score >= threshold:
        return []
    
    return extract_information_requests(analysis)

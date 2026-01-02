"""Unit tests for Information Request Extraction.

WBS Reference: WBS-KB2 - Information Request Detection
Tasks: KB2.3, KB2.4, KB2.7 - Implement and test extract_information_requests
Acceptance Criteria:
- AC-KB2.2: extract_information_requests() parses LLM analysis for requests
- AC-KB2.4: Requests specify source_types (code, books, textbooks, graph)
- AC-KB2.5: Requests have priority based on disagreement severity
- AC-KB2.6: Zero requests returned when agreement_score > threshold

TDD Phase: RED - Tests written before implementation

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- Proper docstrings and type annotations
"""

from __future__ import annotations

import pytest

# =============================================================================
# Test Constants (S1192 compliance)
# =============================================================================

_TEST_PARTICIPANT_ID_A = "participant-a"
_TEST_PARTICIPANT_ID_B = "participant-b"
_TEST_MODEL_ID_A = "qwen2.5-7b"
_TEST_MODEL_ID_B = "deepseek-r1-7b"
_TEST_QUERY = "What is the sub-agent pattern?"

# Sample LLM analysis with information requests embedded
_ANALYSIS_WITH_INFO_REQUESTS = """
Based on the provided evidence, the sub-agent pattern appears to use ParallelAgent.

However, I need more information to be certain:

## Information Requests

1. **Query:** "Show me the ParallelAgent implementation in agents.py"
   - **Source Types:** code
   - **Priority:** high
   - **Reasoning:** Need to verify asyncio.gather usage

2. **Query:** "Find textbook references to sub-agent patterns"
   - **Source Types:** books, textbooks
   - **Priority:** medium
   - **Reasoning:** Want to compare implementation with theory
"""

_ANALYSIS_WITHOUT_INFO_REQUESTS = """
Based on the provided evidence, the sub-agent pattern uses ParallelAgent 
with asyncio.gather for concurrent execution. The implementation is clear
and matches the architectural documentation. I am confident in this assessment.

Confidence: 0.92
"""

_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS = """
Based on the evidence, I see partial implementation.

```json
{
  "information_requests": [
    {
      "query": "Show AST chunking implementation",
      "source_types": ["code"],
      "priority": "high",
      "reasoning": "Participants disagree on chunking approach"
    },
    {
      "query": "Find Graph RAG documentation",
      "source_types": ["books", "graph"],
      "priority": "low"
    }
  ]
}
```
"""

_ANALYSIS_EMPTY_REQUESTS = """
The implementation is complete and well-documented.

```json
{
  "information_requests": []
}
```
"""


# =============================================================================
# KB2.3: Module and Function Existence Tests
# =============================================================================


class TestRequestExtractorModuleExists:
    """Request extractor module exists and is importable."""

    def test_request_extractor_module_exists(self) -> None:
        """request_extractor module is importable."""
        from src.discussion import request_extractor
        assert request_extractor is not None

    def test_extract_information_requests_function_exists(self) -> None:
        """extract_information_requests function exists."""
        from src.discussion.request_extractor import extract_information_requests
        assert callable(extract_information_requests)


# =============================================================================
# KB2.3: Basic Extraction Tests (AC-KB2.2)
# =============================================================================


class TestExtractInformationRequestsBasic:
    """AC-KB2.2: extract_information_requests() parses LLM analysis for requests."""

    def test_extract_returns_list(self) -> None:
        """extract_information_requests returns a list."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_INFO_REQUESTS)
        
        assert isinstance(result, list)

    def test_extract_returns_information_request_objects(self) -> None:
        """Extract returns list of InformationRequest objects."""
        from src.discussion.request_extractor import extract_information_requests
        from src.discussion.models import InformationRequest
        
        result = extract_information_requests(_ANALYSIS_WITH_INFO_REQUESTS)
        
        assert len(result) > 0
        assert all(isinstance(req, InformationRequest) for req in result)

    def test_extract_parses_query_field(self) -> None:
        """Extract parses query from structured output."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_INFO_REQUESTS)
        
        queries = [req.query for req in result]
        assert any("ParallelAgent" in q for q in queries)

    def test_extract_from_empty_analysis(self) -> None:
        """Extract from empty string returns empty list."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests("")
        
        assert result == []

    def test_extract_from_analysis_without_requests(self) -> None:
        """Extract from analysis without requests returns empty list."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITHOUT_INFO_REQUESTS)
        
        assert result == []


# =============================================================================
# KB2.3: JSON Format Extraction Tests (AC-KB2.2)
# =============================================================================


class TestExtractInformationRequestsJsonFormat:
    """Extract from JSON-formatted information_requests section."""

    def test_extract_from_json_block(self) -> None:
        """Extract parses JSON code block with information_requests."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        assert len(result) == 2

    def test_extract_preserves_query_from_json(self) -> None:
        """Extract preserves query field from JSON."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        queries = [req.query for req in result]
        assert "Show AST chunking implementation" in queries

    def test_extract_empty_json_array(self) -> None:
        """Extract handles empty information_requests array."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_EMPTY_REQUESTS)
        
        assert result == []


# =============================================================================
# KB2.3: Source Types Parsing Tests (AC-KB2.4)
# =============================================================================


class TestExtractInformationRequestsSourceTypes:
    """AC-KB2.4: Requests specify source_types (code, books, textbooks, graph)."""

    def test_extract_parses_single_source_type(self) -> None:
        """Extract parses single source_type."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        # First request has source_types: ["code"]
        code_request = next(r for r in result if "AST" in r.query)
        assert code_request.source_types == ["code"]

    def test_extract_parses_multiple_source_types(self) -> None:
        """Extract parses multiple source_types."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        # Second request has source_types: ["books", "graph"]
        graph_request = next(r for r in result if "Graph RAG" in r.query)
        assert "books" in graph_request.source_types
        assert "graph" in graph_request.source_types

    def test_extract_validates_source_type_values(self) -> None:
        """Extract validates source_types are in allowed set."""
        from src.discussion.request_extractor import extract_information_requests
        
        # Analysis with invalid source type should filter or default
        analysis_with_invalid = '''
        ```json
        {
          "information_requests": [
            {
              "query": "test query",
              "source_types": ["invalid_type"],
              "priority": "high"
            }
          ]
        }
        ```
        '''
        
        result = extract_information_requests(analysis_with_invalid)
        
        # Should either filter invalid types or default to valid types
        if result:
            valid_types = {"code", "books", "textbooks", "graph"}
            for req in result:
                assert all(st in valid_types for st in req.source_types)


# =============================================================================
# KB2.4: Priority Scoring Tests (AC-KB2.5)
# =============================================================================


class TestExtractInformationRequestsPriority:
    """AC-KB2.5: Requests have priority based on disagreement severity."""

    def test_extract_parses_high_priority(self) -> None:
        """Extract parses high priority."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        high_priority = [r for r in result if r.priority == "high"]
        assert len(high_priority) >= 1

    def test_extract_parses_low_priority(self) -> None:
        """Extract parses low priority."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        low_priority = [r for r in result if r.priority == "low"]
        assert len(low_priority) >= 1

    def test_extract_defaults_to_medium_priority(self) -> None:
        """Extract defaults to medium priority when not specified."""
        from src.discussion.request_extractor import extract_information_requests
        
        analysis_no_priority = '''
        ```json
        {
          "information_requests": [
            {
              "query": "test query",
              "source_types": ["code"]
            }
          ]
        }
        ```
        '''
        
        result = extract_information_requests(analysis_no_priority)
        
        assert len(result) == 1
        assert result[0].priority == "medium"


# =============================================================================
# KB2.4: Priority from Disagreement Tests (AC-KB2.5)
# =============================================================================


class TestCalculatePriorityFromDisagreement:
    """AC-KB2.5: Priority correlates with confidence gap."""

    def test_calculate_priority_function_exists(self) -> None:
        """calculate_priority_from_disagreement function exists."""
        from src.discussion.request_extractor import calculate_priority_from_disagreement
        assert callable(calculate_priority_from_disagreement)

    def test_high_disagreement_produces_high_priority(self) -> None:
        """High confidence gap (>0.3) produces high priority."""
        from src.discussion.request_extractor import calculate_priority_from_disagreement
        
        # Confidence gap of 0.4 (0.9 vs 0.5)
        priority = calculate_priority_from_disagreement(
            confidence_gap=0.4,
            agreement_score=0.5,
        )
        
        assert priority == "high"

    def test_medium_disagreement_produces_medium_priority(self) -> None:
        """Medium confidence gap (0.15-0.3) produces medium priority."""
        from src.discussion.request_extractor import calculate_priority_from_disagreement
        
        priority = calculate_priority_from_disagreement(
            confidence_gap=0.2,
            agreement_score=0.7,
        )
        
        assert priority == "medium"

    def test_low_disagreement_produces_low_priority(self) -> None:
        """Low confidence gap (<0.15) produces low priority."""
        from src.discussion.request_extractor import calculate_priority_from_disagreement
        
        priority = calculate_priority_from_disagreement(
            confidence_gap=0.1,
            agreement_score=0.8,
        )
        
        assert priority == "low"

    def test_low_agreement_boosts_priority(self) -> None:
        """Low agreement score boosts priority even with small gap."""
        from src.discussion.request_extractor import calculate_priority_from_disagreement
        
        # Small gap but very low agreement
        priority = calculate_priority_from_disagreement(
            confidence_gap=0.1,
            agreement_score=0.3,
        )
        
        # Should be at least medium due to low agreement
        assert priority in ["high", "medium"]


# =============================================================================
# KB2.5: Agreement Threshold Tests (AC-KB2.6)
# =============================================================================


class TestExtractWithAgreementThreshold:
    """AC-KB2.6: Zero requests returned when agreement_score > threshold."""

    def test_extract_with_agreement_score_function_exists(self) -> None:
        """extract_information_requests_with_agreement function exists."""
        from src.discussion.request_extractor import extract_information_requests_with_agreement
        assert callable(extract_information_requests_with_agreement)

    def test_returns_empty_when_agreement_exceeds_threshold(self) -> None:
        """Returns empty list when agreement_score > threshold."""
        from src.discussion.request_extractor import extract_information_requests_with_agreement
        
        result = extract_information_requests_with_agreement(
            analysis=_ANALYSIS_WITH_INFO_REQUESTS,
            agreement_score=0.95,
            threshold=0.85,
        )
        
        assert result == []

    def test_returns_requests_when_agreement_below_threshold(self) -> None:
        """Returns requests when agreement_score < threshold."""
        from src.discussion.request_extractor import extract_information_requests_with_agreement
        
        result = extract_information_requests_with_agreement(
            analysis=_ANALYSIS_WITH_INFO_REQUESTS,
            agreement_score=0.6,
            threshold=0.85,
        )
        
        assert len(result) > 0

    def test_returns_empty_when_agreement_equals_threshold(self) -> None:
        """Returns empty when agreement_score equals threshold (edge case)."""
        from src.discussion.request_extractor import extract_information_requests_with_agreement
        
        result = extract_information_requests_with_agreement(
            analysis=_ANALYSIS_WITH_INFO_REQUESTS,
            agreement_score=0.85,
            threshold=0.85,
        )
        
        assert result == []

    def test_default_threshold_is_0_85(self) -> None:
        """Default threshold is 0.85."""
        from src.discussion.request_extractor import extract_information_requests_with_agreement
        
        # High agreement - should return empty with default threshold
        result_high = extract_information_requests_with_agreement(
            analysis=_ANALYSIS_WITH_INFO_REQUESTS,
            agreement_score=0.90,
        )
        
        # Low agreement - should return requests with default threshold
        result_low = extract_information_requests_with_agreement(
            analysis=_ANALYSIS_WITH_INFO_REQUESTS,
            agreement_score=0.50,
        )
        
        assert result_high == []
        assert len(result_low) > 0


# =============================================================================
# KB2.3: Reasoning Field Extraction Tests
# =============================================================================


class TestExtractInformationRequestsReasoning:
    """Extract preserves reasoning field from analysis."""

    def test_extract_parses_reasoning_field(self) -> None:
        """Extract parses reasoning field when present."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        # First request has reasoning
        req_with_reasoning = next(r for r in result if r.reasoning)
        assert "disagree" in req_with_reasoning.reasoning.lower()

    def test_extract_defaults_reasoning_to_empty(self) -> None:
        """Extract defaults reasoning to empty string when not present."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        # Second request has no reasoning
        req_without_reasoning = next(r for r in result if "Graph RAG" in r.query)
        assert req_without_reasoning.reasoning == ""


# =============================================================================
# KB2.7: Edge Cases and Error Handling Tests
# =============================================================================


class TestExtractInformationRequestsEdgeCases:
    """Edge cases and error handling for extraction."""

    def test_extract_handles_malformed_json(self) -> None:
        """Extract handles malformed JSON gracefully."""
        from src.discussion.request_extractor import extract_information_requests
        
        malformed = '''
        ```json
        {
          "information_requests": [
            { "query": "missing closing brace"
          ]
        }
        ```
        '''
        
        result = extract_information_requests(malformed)
        
        # Should return empty list, not raise exception
        assert result == []

    def test_extract_handles_missing_fields(self) -> None:
        """Extract handles missing required fields gracefully."""
        from src.discussion.request_extractor import extract_information_requests
        
        missing_query = '''
        ```json
        {
          "information_requests": [
            {
              "source_types": ["code"],
              "priority": "high"
            }
          ]
        }
        ```
        '''
        
        result = extract_information_requests(missing_query)
        
        # Should skip invalid entries
        assert result == []

    def test_extract_handles_none_input(self) -> None:
        """Extract handles None input gracefully."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(None)  # type: ignore
        
        assert result == []

    def test_extract_handles_non_string_input(self) -> None:
        """Extract handles non-string input gracefully."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(123)  # type: ignore
        
        assert result == []


# =============================================================================
# KB2.7: Multiple Format Support Tests
# =============================================================================


class TestExtractInformationRequestsFormats:
    """Extract supports multiple output formats from LLMs."""

    def test_extract_from_markdown_format(self) -> None:
        """Extract parses markdown-formatted requests."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_INFO_REQUESTS)
        
        # Should parse from markdown format
        assert len(result) >= 1

    def test_extract_from_json_format(self) -> None:
        """Extract parses JSON-formatted requests."""
        from src.discussion.request_extractor import extract_information_requests
        
        result = extract_information_requests(_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS)
        
        # Should parse from JSON format
        assert len(result) == 2

    def test_extract_prefers_json_over_markdown(self) -> None:
        """Extract prefers JSON format when both present."""
        from src.discussion.request_extractor import extract_information_requests
        
        mixed_format = '''
        ## Information Requests
        1. **Query:** "Markdown request"
           - **Source Types:** code
        
        ```json
        {
          "information_requests": [
            {
              "query": "JSON request",
              "source_types": ["code"],
              "priority": "high"
            }
          ]
        }
        ```
        '''
        
        result = extract_information_requests(mixed_format)
        
        # JSON should take precedence
        json_requests = [r for r in result if r.query == "JSON request"]
        assert len(json_requests) >= 1


# =============================================================================
# KB2.7: Integration with ParticipantAnalysis Tests
# =============================================================================


class TestExtractFromParticipantAnalysis:
    """Extract works with ParticipantAnalysis.content field."""

    def test_extract_from_participant_analysis(self) -> None:
        """Extract works with content from ParticipantAnalysis."""
        from src.discussion.request_extractor import extract_information_requests
        from src.discussion.models import ParticipantAnalysis
        
        analysis = ParticipantAnalysis(
            participant_id="participant-a",
            model_id="qwen2.5-7b",
            content=_ANALYSIS_WITH_STRUCTURED_JSON_REQUESTS,
            confidence=0.6,
        )
        
        result = extract_information_requests(analysis.content)
        
        assert len(result) == 2

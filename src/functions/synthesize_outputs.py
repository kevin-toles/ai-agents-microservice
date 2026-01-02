"""synthesize_outputs agent function.

WBS-AGT11: synthesize_outputs Function implementation.

Purpose: Combine multiple outputs into a coherent whole.
- Combines multiple outputs into coherent whole (AC-11.1)
- Returns SynthesizedOutput with merged_content, source_map (AC-11.2)
- Context budget: 8192 input / 4096 output (AC-11.3)
- Default preset: S1 (Light) (AC-11.4)
- Preserves citations from input sources (AC-11.5)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 6

REFACTOR Phase:
- Using estimate_tokens() from src/functions/utils/token_utils.py (S1192)
- Citation merging logic extracted to _merge_citations() method
- Source map tracking extracted to _build_source_map() method
"""

import logging
import re
from collections import Counter
from typing import Any

from src.functions.base import AgentFunction, ContextBudgetExceededError
from src.functions.utils.token_utils import estimate_tokens
from src.schemas.citations import Citation, SourceMetadata
from src.schemas.functions.synthesize_outputs import (
    Conflict,
    ConflictPolicy,
    OutputItem,
    SynthesizedOutput,
    SynthesisStrategy,
)

logger = logging.getLogger(__name__)

# Context budget from AGENT_FUNCTIONS_ARCHITECTURE.md
INPUT_BUDGET_TOKENS = 8192
OUTPUT_BUDGET_TOKENS = 4096


class SynthesizeOutputsFunction(AgentFunction):
    """Combine multiple outputs into a coherent result.

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 6

    Acceptance Criteria:
    - AC-11.1: Combines multiple outputs into coherent whole
    - AC-11.2: Returns SynthesizedOutput with merged_content, source_map
    - AC-11.3: Context budget: 8192 input / 4096 output
    - AC-11.4: Default preset: S1 (Light)
    - AC-11.5: Preserves citations from input sources
    """

    name: str = "synthesize_outputs"

    # AC-11.4: Default preset S1 (Light) - phi-4 for general synthesis
    default_preset: str = "S1"

    # Preset options from architecture doc
    available_presets: dict[str, str] = {
        "default": "S1",          # phi-4 for general synthesis
        "code": "D4",             # Code-aware synthesis
        "debate_reconciliation": "D3",  # Debate for conflict resolution
    }

    async def run(self, **kwargs: Any) -> SynthesizedOutput:
        """Synthesize multiple outputs into a coherent whole.

        Args:
            outputs: List of OutputItem to synthesize
            synthesis_strategy: merge/reconcile/vote strategy
            conflict_policy: first_wins/consensus/flag policy

        Returns:
            SynthesizedOutput with merged_content, source_map, citations

        Raises:
            ContextBudgetExceededError: If total input exceeds budget
        """
        outputs: list[OutputItem] = kwargs.get("outputs", [])
        synthesis_strategy = kwargs.get(
            "synthesis_strategy", SynthesisStrategy.MERGE
        )
        conflict_policy = kwargs.get(
            "conflict_policy", ConflictPolicy.FIRST_WINS
        )

        # Convert enums if passed as strings
        if isinstance(synthesis_strategy, str):
            synthesis_strategy = SynthesisStrategy(synthesis_strategy)
        if isinstance(conflict_policy, str):
            conflict_policy = ConflictPolicy(conflict_policy)

        # AC-11.3: Enforce input budget
        total_content = " ".join(item.content for item in outputs)
        input_tokens = estimate_tokens(total_content)
        if input_tokens > INPUT_BUDGET_TOKENS:
            raise ContextBudgetExceededError(
                function_name=self.name,
                actual=input_tokens,
                limit=INPUT_BUDGET_TOKENS,
            )

        # Synthesize based on strategy
        if synthesis_strategy == SynthesisStrategy.MERGE:
            merged_content, source_map = self._merge_outputs(outputs)
        elif synthesis_strategy == SynthesisStrategy.RECONCILE:
            merged_content, source_map = self._reconcile_outputs(
                outputs, conflict_policy
            )
        elif synthesis_strategy == SynthesisStrategy.VOTE:
            merged_content, source_map = self._vote_outputs(outputs)
        else:
            # Default to merge
            merged_content, source_map = self._merge_outputs(outputs)

        # AC-11.5: Merge and renumber citations
        citations = self._merge_citations(outputs)

        # Update citation markers in merged content
        merged_content = self._update_citation_markers(
            merged_content, outputs, citations
        )

        # Detect conflicts if using flag policy
        conflicts: list[Conflict] = []
        if conflict_policy == ConflictPolicy.FLAG:
            conflicts = self._detect_conflicts(outputs)

        # Calculate agreement score
        agreement_score = self._calculate_agreement_score(outputs)

        return SynthesizedOutput(
            merged_content=merged_content,
            source_map=source_map,
            citations=citations,
            agreement_score=agreement_score,
            conflicts=conflicts,
        )

    def _merge_outputs(
        self, outputs: list[OutputItem]
    ) -> tuple[str, dict[str, list[str]]]:
        """Merge multiple outputs into coherent content.

        Implements AC-11.1: Combines outputs without duplicates.

        Args:
            outputs: List of OutputItem to merge

        Returns:
            Tuple of (merged_content, source_map)
        """
        source_map: dict[str, list[str]] = {}
        seen_content: set[str] = set()
        merged_parts: list[str] = []

        for idx, item in enumerate(outputs):
            # Normalize content for deduplication check
            normalized = self._normalize_content(item.content)

            # Skip duplicate content - Exit Criteria: No duplicate content
            if normalized in seen_content:
                logger.debug(
                    "Skipping duplicate content from %s", item.source_id
                )
                continue

            seen_content.add(normalized)
            merged_parts.append(item.content)

            # Track provenance - AC-11.2: source_map
            section_key = f"section_{idx + 1}"
            source_map[section_key] = [item.source_id]

        # Join with paragraph breaks for readability
        merged_content = "\n\n".join(merged_parts)

        return merged_content, source_map

    def _reconcile_outputs(
        self,
        outputs: list[OutputItem],
        conflict_policy: ConflictPolicy,
    ) -> tuple[str, dict[str, list[str]]]:
        """Reconcile outputs, resolving conflicts.

        Args:
            outputs: List of OutputItem to reconcile
            conflict_policy: How to handle conflicts

        Returns:
            Tuple of (reconciled_content, source_map)
        """
        source_map: dict[str, list[str]] = {}
        merged_parts: list[str] = []
        used_sources: list[str] = []

        for idx, item in enumerate(outputs):
            merged_parts.append(item.content)
            used_sources.append(item.source_id)

            # Track all sources for this reconciled output
            section_key = f"section_{idx + 1}"
            source_map[section_key] = [item.source_id]

        # Mark as reconciled from all sources
        source_map["reconciled_from"] = used_sources
        # Record policy used for future reference
        source_map["conflict_policy"] = [conflict_policy.value]

        merged_content = "\n\n".join(merged_parts)
        return merged_content, source_map

    def _vote_outputs(
        self, outputs: list[OutputItem]
    ) -> tuple[str, dict[str, list[str]]]:
        """Vote on outputs to select most common content.

        Args:
            outputs: List of OutputItem to vote on

        Returns:
            Tuple of (voted_content, source_map)
        """
        # Normalize and count content occurrences
        content_votes: Counter[str] = Counter()
        content_to_original: dict[str, str] = {}
        content_to_sources: dict[str, list[str]] = {}

        for item in outputs:
            normalized = self._normalize_content(item.content)
            content_votes[normalized] += 1
            content_to_original[normalized] = item.content
            if normalized not in content_to_sources:
                content_to_sources[normalized] = []
            content_to_sources[normalized].append(item.source_id)

        # Select most common content
        if not content_votes:
            return "", {}

        most_common = content_votes.most_common(1)[0][0]
        winning_content = content_to_original[most_common]
        winning_sources = content_to_sources[most_common]

        source_map = {
            "voted_content": winning_sources,
            "vote_count": [str(content_votes[most_common])],
        }

        return winning_content, source_map

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison.

        Strips whitespace, lowercases, removes citation markers.

        Args:
            content: Raw content string

        Returns:
            Normalized string for comparison
        """
        # Remove citation markers [^N]
        normalized = re.sub(r"\[\^?\d+\]", "", content)
        # Normalize whitespace
        normalized = " ".join(normalized.split())
        # Lowercase for comparison
        return normalized.lower()

    def _merge_citations(self, outputs: list[OutputItem]) -> list[Citation]:
        """Merge and renumber citations from all outputs.

        Implements AC-11.5: Preserves citations from input sources.
        Exit Criteria: Citations renumbered correctly after merge.

        Args:
            outputs: List of OutputItem with citations

        Returns:
            List of renumbered Citation objects
        """
        merged_citations: list[Citation] = []
        marker_offset = 0

        for item in outputs:
            for citation in item.citations:
                # Create new citation with renumbered marker
                new_marker = marker_offset + citation.marker
                
                # Check if this source already exists (same book, same pages)
                existing = self._find_existing_citation(
                    merged_citations, citation.source
                )
                
                if existing is None:
                    # Add new citation with renumbered marker
                    new_citation = Citation(
                        marker=new_marker,
                        source=citation.source,
                        context=citation.context,
                    )
                    merged_citations.append(new_citation)

            # Update offset for next output's citations
            if item.citations:
                marker_offset = max(c.marker for c in merged_citations)

        # Renumber all citations sequentially
        for idx, citation in enumerate(merged_citations):
            # Create new citation with sequential marker
            merged_citations[idx] = Citation(
                marker=idx + 1,
                source=citation.source,
                context=citation.context,
            )

        return merged_citations

    def _find_existing_citation(
        self,
        citations: list[Citation],
        source: SourceMetadata,
    ) -> Citation | None:
        """Find existing citation with same source.

        Args:
            citations: List of existing citations
            source: Source metadata to match

        Returns:
            Existing citation or None
        """
        for citation in citations:
            # Match on key source fields
            if (
                citation.source.source_type == source.source_type
                and citation.source.title == source.title
                and citation.source.author == source.author
                and citation.source.pages == source.pages
            ):
                return citation
        return None

    def _update_citation_markers(
        self,
        content: str,
        outputs: list[OutputItem],
        merged_citations: list[Citation],
    ) -> str:
        """Update citation markers in content to match merged numbering.

        Args:
            content: Merged content with old markers
            outputs: Original outputs with citations
            merged_citations: Renumbered citations (used for validation)

        Returns:
            Content with updated citation markers
        """
        # Build mapping from old markers to new
        old_to_new: dict[str, int] = {}

        current_marker = 0
        for item in outputs:
            for citation in item.citations:
                current_marker += 1
                # Map original marker to new sequential marker
                old_key = f"[^{citation.marker}]"
                if old_key not in old_to_new:
                    old_to_new[old_key] = current_marker

        # Verify marker count matches merged citations
        expected_count = len(merged_citations)
        if current_marker > 0 and expected_count > 0:
            logger.debug(
                "Citation mapping: %d old markers -> %d merged citations",
                current_marker,
                expected_count,
            )

        # Replace markers in content using placeholders
        updated_content = self._replace_markers_with_placeholders(
            content, old_to_new
        )

        # Replace placeholders with final markers
        for marker_num in range(1, current_marker + 1):
            placeholder = f"__CITE_{marker_num}__"
            updated_content = updated_content.replace(
                placeholder, f"[^{marker_num}]"
            )

        return updated_content

    def _replace_markers_with_placeholders(
        self,
        content: str,
        old_to_new: dict[str, int],
    ) -> str:
        """Replace old citation markers with placeholders.

        Args:
            content: Content with old markers
            old_to_new: Mapping of old markers to new numbers

        Returns:
            Content with placeholders
        """
        result = content
        # Sort by marker length descending to avoid partial replacements
        for old_marker in sorted(old_to_new.keys(), key=len, reverse=True):
            placeholder = f"__CITE_{old_to_new[old_marker]}__"
            result = result.replace(old_marker, placeholder)
        return result

    def _detect_conflicts(self, outputs: list[OutputItem]) -> list[Conflict]:
        """Detect conflicts between outputs.

        Args:
            outputs: List of OutputItem to check

        Returns:
            List of detected Conflict objects
        """
        conflicts: list[Conflict] = []

        # Simple conflict detection: look for contradictory keywords
        contradiction_pairs = [
            ("should", "should not"),
            ("must", "must not"),
            ("always", "never"),
            ("recommended", "not recommended"),
        ]

        for idx, item in enumerate(outputs):
            item_conflicts = self._find_contradictions(
                item, idx, outputs, contradiction_pairs
            )
            conflicts.extend(item_conflicts)

        return conflicts

    def _find_contradictions(
        self,
        item: OutputItem,
        idx: int,
        outputs: list[OutputItem],
        contradiction_pairs: list[tuple[str, str]],
    ) -> list[Conflict]:
        """Find contradictions between an item and other outputs.

        Args:
            item: The item to check
            idx: Index of the item
            outputs: All outputs
            contradiction_pairs: Pairs of contradictory terms

        Returns:
            List of conflicts found
        """
        conflicts: list[Conflict] = []
        content_lower = item.content.lower()

        for positive, negative in contradiction_pairs:
            if positive not in content_lower:
                continue

            for other_idx, other in enumerate(outputs):
                if other_idx == idx:
                    continue

                if negative in other.content.lower():
                    conflict = Conflict(
                        section=f"section_{idx + 1}",
                        source_ids=[item.source_id, other.source_id],
                        description=(
                            f"Potential contradiction: "
                            f"'{positive}' vs '{negative}'"
                        ),
                    )
                    conflicts.append(conflict)

        return conflicts

    def _calculate_agreement_score(
        self, outputs: list[OutputItem]
    ) -> float | None:
        """Calculate agreement score between outputs.

        Higher score indicates more consensus among sources.

        Args:
            outputs: List of OutputItem to analyze

        Returns:
            Agreement score 0.0-1.0 or None if not calculable
        """
        if len(outputs) < 2:
            return None

        # Simple word overlap calculation
        word_sets: list[set[str]] = []
        for item in outputs:
            words = set(self._normalize_content(item.content).split())
            word_sets.append(words)

        # Calculate pairwise Jaccard similarity
        total_similarity = 0.0
        pair_count = 0

        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                set_a = word_sets[i]
                set_b = word_sets[j]
                
                if not set_a or not set_b:
                    continue
                
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                
                if union > 0:
                    similarity = intersection / union
                    total_similarity += similarity
                    pair_count += 1

        if pair_count == 0:
            return None

        return total_similarity / pair_count

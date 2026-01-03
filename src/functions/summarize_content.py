"""summarize_content agent function.

WBS-AGT7: summarize_content Function implementation.

Purpose: Compress content while preserving invariants.
- Generates summaries with citation markers [^N]
- Returns CitedContent with footnotes list
- Context budget: 8192 input / 4096 output
- Default preset: D4 (Standard)
- Supports detail_level parameter (brief/standard/comprehensive)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 2

REFACTOR Phase:
- Extracted CHARS_PER_TOKEN to shared utilities (S1192)
- Using estimate_tokens() from src/functions/utils/token_utils.py
- AGT7.7: Now calls inference-service for LLM-powered summarization
"""

import logging
import os
import re
from typing import Any

from src.functions.base import AgentFunction, ContextBudgetExceededError
from src.functions.utils.token_utils import estimate_tokens, tokens_to_chars
from src.schemas.citations import Citation, SourceMetadata
from src.schemas.functions.summarize_content import (
    DetailLevel,
    SummarizeContentOutput,
    SummaryStyle,
)

logger = logging.getLogger(__name__)

# Environment variable to enable LLM-powered summarization
# Kitchen Brigade: LLM summarization enabled by default for full pipeline
USE_LLM_SUMMARIZATION = os.getenv("USE_LLM_SUMMARIZATION", "true").lower() == "true"


# Context budget for summarize_content
INPUT_BUDGET_TOKENS = 8192
OUTPUT_BUDGET_TOKENS = 4096

# Target token counts by detail level
DETAIL_LEVEL_TOKENS = {
    DetailLevel.BRIEF: 200,       # <500 tokens per AC-7.5
    DetailLevel.STANDARD: 500,    # Balanced
    DetailLevel.COMPREHENSIVE: 1000,  # Detailed
}


class SummarizeContentFunction(AgentFunction):
    """Compress content while preserving invariants.

    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 2

    Acceptance Criteria:
    - AC-7.1: Generates summaries with citation markers [^N]
    - AC-7.2: Returns CitedContent with footnotes list
    - AC-7.3: Context budget: 8192 input / 4096 output
    - AC-7.4: Default preset: D4 (Standard)
    - AC-7.5: Supports detail_level parameter
    """

    name: str = "summarize_content"

    # AC-7.4: Default preset D4 (Standard)
    default_preset: str = "D4"

    # Preset options from architecture doc
    available_presets: dict[str, str] = {
        "default": "D4",      # deepseek + qwen (critique)
        "short_input": "S4",  # llama-3.2-3b (fast)
        "long_input": "S5",   # phi-3-medium-128k
        "technical": "D4",    # deepseek + qwen (critique)
    }

    async def run(self, **kwargs: Any) -> SummarizeContentOutput:
        """Summarize content with citation markers.

        Args:
            content: Text to summarize
            detail_level: brief/standard/comprehensive
            target_tokens: Override target token count
            preserve: Concepts to preserve in summary
            style: technical/executive/bullets
            sources: SourceMetadata list for citations

        Returns:
            SummarizeContentOutput with summary, footnotes, invariants

        Raises:
            ContextBudgetExceededError: If input exceeds budget
        """
        content: str = kwargs.get("content", "")
        detail_level = kwargs.get("detail_level", DetailLevel.STANDARD)
        target_tokens: int | None = kwargs.get("target_tokens")
        preserve: list[str] = kwargs.get("preserve", [])
        style = kwargs.get("style", SummaryStyle.TECHNICAL)
        sources: list[SourceMetadata] = kwargs.get("sources", [])
        preset: str = kwargs.get("preset", self.default_preset)

        # Convert enums if passed as strings
        if isinstance(detail_level, str):
            detail_level = DetailLevel(detail_level)
        if isinstance(style, str):
            style = SummaryStyle(style)

        # AC-7.3: Enforce input budget - using shared utility
        input_tokens = estimate_tokens(content)
        if input_tokens > INPUT_BUDGET_TOKENS:
            raise ContextBudgetExceededError(
                function_name=self.name,
                actual=input_tokens,
                limit=INPUT_BUDGET_TOKENS,
            )

        # Determine target token count
        if target_tokens is None:
            target_tokens = DETAIL_LEVEL_TOKENS.get(detail_level, 500)

        # Generate summary based on style (uses LLM if enabled)
        summary, invariants = self._generate_summary(
            content=content,
            target_tokens=target_tokens,
            preserve=preserve,
            style=style,
            preset=preset,
        )

        # AC-7.1: Inject citation markers
        footnotes: list[Citation] = []
        if sources:
            summary, footnotes = self._inject_citations(
                summary=summary,
                sources=sources,
            )

        # Calculate metrics - using shared utility
        summary_tokens = estimate_tokens(summary)
        compression_ratio = summary_tokens / input_tokens if input_tokens > 0 else 0.0

        return SummarizeContentOutput(
            summary=summary,
            footnotes=footnotes,
            invariants=invariants,
            compression_ratio=compression_ratio,
            token_count=summary_tokens,
        )

    def _generate_summary(
        self,
        content: str,
        target_tokens: int,
        preserve: list[str],
        style: SummaryStyle,
        preset: str = "D4",
    ) -> tuple[str, list[str]]:
        """Generate summary with specified style.

        Uses inference-service LLM when USE_LLM_SUMMARIZATION=true,
        otherwise falls back to local extraction.

        Args:
            content: Text to summarize
            target_tokens: Target output token count
            preserve: Concepts to preserve in summary
            style: Summary style (technical/executive/bullets)
            preset: Model preset (D4/S1/D10)

        Returns:
            Tuple of (summary, invariants)
        """
        if USE_LLM_SUMMARIZATION:
            return self._generate_summary_llm(
                content=content,
                target_tokens=target_tokens,
                preserve=preserve,
                style=style,
                preset=preset,
            )
        else:
            return self._generate_summary_local(
                content=content,
                target_tokens=target_tokens,
                preserve=preserve,
                style=style,
            )

    def _generate_summary_llm(
        self,
        content: str,
        target_tokens: int,
        preserve: list[str],
        style: SummaryStyle,
        preset: str = "D4",  # Kept for backward compat, but ignored
    ) -> tuple[str, list[str]]:
        """Generate summary using inference-service LLM.

        AGT7.7: Implement run() with inference-service call.
        
        Model Selection:
            Uses MODEL_PREFERENCES["summarize_content"] to express preference.
            inference-service owns model lifecycle - we just express preference
            and fallback to whatever is loaded.

        Args:
            content: Text to summarize
            target_tokens: Target output token count
            preserve: Concepts to preserve
            style: Summary style
            preset: DEPRECATED - kept for backward compat, ignored

        Returns:
            Tuple of (summary, invariants)
        """
        import asyncio
        from src.clients.inference_service import create_inference_client

        # Map detail level from target tokens
        if target_tokens <= 300:
            detail_level = "brief"
        elif target_tokens <= 600:
            detail_level = "standard"
        else:
            detail_level = "comprehensive"

        # Map style enum to string
        style_str = style.value if hasattr(style, "value") else str(style).lower()

        logger.info(
            "Generating LLM summary: detail=%s, style=%s, model_preference=summarize_content",
            detail_level,
            style_str,
        )

        async def _call_llm() -> str:
            client = create_inference_client()
            try:
                # Use model_preference instead of preset
                # inference-service owns model selection - we express preference
                return await client.summarize(
                    content=content,
                    model_preference="summarize_content",  # Uses MODEL_PREFERENCES
                    detail_level=detail_level,
                    style=style_str,
                    preserve=preserve if preserve else None,
                )
            finally:
                await client.close()

        # Run async call - check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use await
            # But this is a sync method, so we need to handle this
            # Create a new task and wait for it
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _call_llm())
                summary = future.result(timeout=120)
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            summary = asyncio.run(_call_llm())

        # Extract invariants from preserved concepts
        invariants = [f"Preserved: {concept}" for concept in (preserve or [])]

        return summary, invariants

    def _generate_summary_local(
        self,
        content: str,
        target_tokens: int,
        preserve: list[str],
        style: SummaryStyle,
    ) -> tuple[str, list[str]]:
        """Generate summary using local extraction (fallback).

        This is the original local implementation that extracts key sentences.
        Used when USE_LLM_SUMMARIZATION=false or inference-service unavailable.

        Returns:
            Tuple of (summary, invariants)
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())

        # Calculate target character count - using shared utility
        target_chars = tokens_to_chars(target_tokens)

        # Collect key sentences
        selected: list[str] = []
        invariants: list[str] = []
        current_chars = 0

        # First pass: find sentences containing preserved concepts
        for sentence in sentences:
            for concept in preserve:
                if concept.lower() in sentence.lower():
                    if sentence not in selected:
                        selected.append(sentence)
                        invariants.append(f"Preserved: {concept}")
                        current_chars += len(sentence)
                    break

        # Second pass: add other important sentences until target reached
        for sentence in sentences:
            if sentence in selected:
                continue
            if current_chars >= target_chars:
                break

            # Skip very short sentences
            if len(sentence) < 20:
                continue

            selected.append(sentence)
            current_chars += len(sentence)

        # Format based on style
        if style == SummaryStyle.BULLETS:
            summary = self._format_bullets(selected)
        elif style == SummaryStyle.EXECUTIVE:
            summary = self._format_executive(selected)
        else:  # TECHNICAL
            summary = self._format_technical(selected)

        # Ensure we have at least some output
        if not summary:
            summary = content[:target_chars].strip()
            if len(content) > target_chars:
                summary += "..."

        return summary, invariants

    def _format_bullets(self, sentences: list[str]) -> str:
        """Format sentences as bullet points."""
        bullets = []
        for sentence in sentences:
            # Clean up sentence
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                bullets.append(f"- {stripped_sentence}")
        return "\n".join(bullets)

    def _format_executive(self, sentences: list[str]) -> str:
        """Format as executive summary."""
        if not sentences:
            return ""

        # Join sentences into paragraphs
        return " ".join(s.strip() for s in sentences if s.strip())

    def _format_technical(self, sentences: list[str]) -> str:
        """Format as technical prose."""
        if not sentences:
            return ""

        # Join sentences into coherent prose
        return " ".join(s.strip() for s in sentences if s.strip())

    def _inject_citations(
        self,
        summary: str,
        sources: list[SourceMetadata],
    ) -> tuple[str, list[Citation]]:
        """Inject citation markers [^N] into summary.

        AC-7.1: Generates summaries with citation markers
        AC-7.2: Returns footnotes list with Chicago-style citations

        Returns:
            Tuple of (summary with markers, footnotes list)
        """
        footnotes: list[Citation] = []

        # Create citations from sources
        for i, source in enumerate(sources, start=1):
            # Citation requires marker and source (SourceMetadata)
            citation = Citation(
                marker=i,
                source=source,
            )
            footnotes.append(citation)

        # Inject citation markers at the end of sentences
        if footnotes:
            # Find good places to insert citations
            # Simple approach: add citation after first sentence that mentions source topic
            sentences = summary.split(". ")
            if sentences:
                # Add citation to first substantive sentence
                for i, sentence in enumerate(sentences):
                    if len(sentence) > 30:  # Substantive sentence
                        # Add all source citations to first sentence
                        markers = "".join(f"[^{j+1}]" for j in range(len(footnotes)))
                        sentences[i] = sentence.rstrip(".") + markers
                        break

                summary = ". ".join(sentences)
                # Ensure proper ending
                if not summary.endswith("."):
                    summary += "."

        return summary, footnotes

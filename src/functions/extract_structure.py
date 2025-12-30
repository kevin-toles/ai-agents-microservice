"""extract_structure agent function.

WBS-AGT6: extract_structure Function implementation.

Purpose: Extract structured data from unstructured content.
- Parses JSON/Markdown/Code into hierarchical structure
- Returns StructuredOutput with headings, sections, code_blocks
- Context budget: 16384 input / 2048 output
- Default preset: S1 (Light)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 1

REFACTOR Phase:
- Extracted CHARS_PER_TOKEN to shared utilities (S1192)
- Using estimate_tokens() from src/functions/utils/token_utils.py
"""

import json
import re
from typing import Any

from src.functions.base import AgentFunction, ContextBudgetExceededError
from src.functions.utils.token_utils import estimate_tokens
from src.schemas.functions.extract_structure import (
    ArtifactType,
    CodeBlock,
    ExtractedItem,
    ExtractionType,
    Heading,
    Section,
    StructuredOutput,
)


# Context budget for extract_structure
INPUT_BUDGET_TOKENS = 16384
OUTPUT_BUDGET_TOKENS = 2048


class ExtractStructureFunction(AgentFunction):
    """Extract structured data from unstructured content.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 1
    
    Acceptance Criteria:
    - AC-6.1: Parses JSON/Markdown/Code into hierarchical structure
    - AC-6.2: Returns StructuredOutput with headings, sections, code_blocks
    - AC-6.3: Context budget: 16384 input / 2048 output
    - AC-6.4: Default preset: S1 (Light)
    - AC-6.5: Supports artifact_type parameter
    """
    
    name: str = "extract_structure"
    
    # AC-6.4: Default preset S1 (Light)
    default_preset: str = "S1"
    
    # Preset options from architecture doc
    available_presets: dict[str, str] = {
        "default": "S1",      # phi-4 for general extraction
        "code_heavy": "S6",   # granite-8b-code-128k
        "long_input": "S5",   # phi-3-medium-128k
    }
    
    async def run(self, **kwargs: Any) -> StructuredOutput:
        """Extract structure from content.
        
        Args:
            content: Raw text/code to extract structure from
            artifact_type: Type of artifact (auto/json/markdown/code)
            extraction_type: Type of extraction (keywords/concepts/entities/outline)
            domain: Domain context (ai-ml/systems/web/general)
            
        Returns:
            StructuredOutput with headings, sections, code_blocks, etc.
            
        Raises:
            ContextBudgetExceededError: If input exceeds budget
        """
        content: str = kwargs.get("content", "")
        artifact_type = kwargs.get("artifact_type", ArtifactType.AUTO)
        extraction_type = kwargs.get("extraction_type", ExtractionType.OUTLINE)
        domain: str = kwargs.get("domain", "general")
        
        # Convert enum if passed as string
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)
        if isinstance(extraction_type, str):
            extraction_type = ExtractionType(extraction_type)
        
        # AC-6.3: Enforce input budget - using shared utility
        input_tokens = estimate_tokens(content)
        if input_tokens > INPUT_BUDGET_TOKENS:
            raise ContextBudgetExceededError(
                function_name=self.name,
                actual=input_tokens,
                limit=INPUT_BUDGET_TOKENS,
            )
        
        # AC-6.5: Auto-detect artifact type if needed
        if artifact_type == ArtifactType.AUTO:
            artifact_type = self._detect_artifact_type(content)
        
        # Dispatch to appropriate parser
        if artifact_type == ArtifactType.JSON:
            return self._parse_json(content, extraction_type)
        elif artifact_type == ArtifactType.MARKDOWN:
            return self._parse_markdown(content, extraction_type)
        elif artifact_type == ArtifactType.CODE:
            return self._parse_code(content, extraction_type, domain)
        else:
            # Fallback to markdown parsing
            return self._parse_markdown(content, extraction_type)
    
    def _detect_artifact_type(self, content: str) -> ArtifactType:
        """Auto-detect artifact type from content.
        
        Detection heuristics:
        1. Starts with { or [ → JSON
        2. Contains # headings → Markdown
        3. Contains def/class/function → Code
        4. Default → Markdown
        """
        stripped = content.strip()
        
        # JSON detection
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return ArtifactType.JSON
            except json.JSONDecodeError:
                pass
        
        # Markdown detection (# headings)
        if re.search(r"^#{1,6}\s+\S", content, re.MULTILINE):
            return ArtifactType.MARKDOWN
        
        # Code detection (Python patterns)
        code_patterns = [
            r"^\s*def\s+\w+\s*\(",
            r"^\s*class\s+\w+",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import",
        ]
        for pattern in code_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return ArtifactType.CODE
        
        # Default to markdown
        return ArtifactType.MARKDOWN
    
    def _parse_json(
        self,
        content: str,
        extraction_type: ExtractionType,
    ) -> StructuredOutput:
        """Parse JSON content into structured output.
        
        AC-6.1: Nested structure with depth levels
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Return empty structure for invalid JSON
            return StructuredOutput(
                compressed_summary="Invalid JSON content",
            )
        
        sections: list[Section] = []
        extracted_items: list[ExtractedItem] = []
        
        # Recursively extract structure
        self._extract_json_structure(data, sections, extracted_items, depth=0)
        
        # Generate compressed summary
        summary = self._generate_json_summary(data)
        
        return StructuredOutput(
            sections=sections,
            extracted_items=extracted_items,
            compressed_summary=summary,
        )
    
    def _extract_json_structure(
        self,
        data: Any,
        sections: list[Section],
        items: list[ExtractedItem],
        depth: int,
        path: str = "",
    ) -> None:
        """Recursively extract structure from JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, dict):
                    # Create section for nested object at current depth
                    sections.append(Section(
                        title=key,
                        content=json.dumps(value, indent=2)[:200],  # Truncate
                        depth=depth,
                    ))
                    # Recurse into nested dict with incremented depth
                    self._extract_json_structure(
                        value, sections, items, depth + 1, current_path
                    )
                elif isinstance(value, list):
                    # Extract list items
                    sections.append(Section(
                        title=key,
                        content=f"Array with {len(value)} items",
                        depth=depth,
                    ))
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            self._extract_json_structure(
                                item, sections, items, depth + 1, f"{current_path}[{i}]"
                            )
                else:
                    # Leaf value - also add as section to track depth
                    sections.append(Section(
                        title=key,
                        content=str(value),
                        depth=depth,
                    ))
                    # Extract as item for keywords/concepts
                    items.append(ExtractedItem(
                        value=f"{key}: {value}",
                        category="property",
                        confidence_score=1.0,
                    ))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._extract_json_structure(
                    item, sections, items, depth, f"{path}[{i}]"
                )
    
    def _generate_json_summary(self, data: Any) -> str:
        """Generate compressed summary for JSON data."""
        if isinstance(data, dict):
            keys = list(data.keys())
            return f"JSON object with keys: {', '.join(keys[:5])}" + (
                f" and {len(keys) - 5} more" if len(keys) > 5 else ""
            )
        elif isinstance(data, list):
            return f"JSON array with {len(data)} items"
        else:
            return f"JSON value: {str(data)[:50]}"
    
    def _parse_markdown(
        self,
        content: str,
        extraction_type: ExtractionType,
    ) -> StructuredOutput:
        """Parse Markdown content into structured output.
        
        AC-6.1: Identifies headings H1-H6
        AC-6.2: Extracts code blocks
        """
        lines = content.split("\n")
        headings: list[Heading] = []
        code_blocks: list[CodeBlock] = []
        sections: list[Section] = []
        
        in_code_block = False
        code_block_start = 0
        code_block_language = None
        code_block_lines: list[str] = []
        
        for i, line in enumerate(lines, start=1):
            # Handle fenced code blocks
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    code_block_start = i
                    # Extract language
                    lang_match = re.match(r"```(\w+)", line.strip())
                    code_block_language = lang_match.group(1) if lang_match else None
                    code_block_lines = []
                else:
                    # End of code block
                    in_code_block = False
                    code_blocks.append(CodeBlock(
                        code="\n".join(code_block_lines),
                        language=code_block_language,
                        start_line=code_block_start,
                        end_line=i,
                    ))
                continue
            
            if in_code_block:
                code_block_lines.append(line)
                continue
            
            # Extract headings
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                headings.append(Heading(
                    level=level,
                    text=text,
                    line_number=i,
                ))
        
        # Generate sections from headings
        sections = self._headings_to_sections(headings, lines)
        
        # Generate compressed summary
        summary = self._generate_markdown_summary(headings, code_blocks)
        
        return StructuredOutput(
            headings=headings,
            sections=sections,
            code_blocks=code_blocks,
            compressed_summary=summary,
        )
    
    def _headings_to_sections(
        self,
        headings: list[Heading],
        lines: list[str],
    ) -> list[Section]:
        """Convert headings to sections with content."""
        sections: list[Section] = []
        
        for i, heading in enumerate(headings):
            start_line = heading.line_number or 1
            
            # Find end line (next heading or end of document)
            if i + 1 < len(headings):
                end_line = (headings[i + 1].line_number or len(lines)) - 1
            else:
                end_line = len(lines)
            
            # Extract content between headings
            content_lines = lines[start_line:end_line]
            content = "\n".join(content_lines).strip()
            
            sections.append(Section(
                title=heading.text,
                content=content[:500],  # Truncate for budget
                depth=heading.level - 1,
                start_line=start_line,
                end_line=end_line,
            ))
        
        return sections
    
    def _generate_markdown_summary(
        self,
        headings: list[Heading],
        code_blocks: list[CodeBlock],
    ) -> str:
        """Generate compressed summary for Markdown content."""
        parts: list[str] = []
        
        if headings:
            h1_count = sum(1 for h in headings if h.level == 1)
            parts.append(f"{len(headings)} headings ({h1_count} H1)")
        
        if code_blocks:
            languages = list({b.language for b in code_blocks if b.language})
            parts.append(f"{len(code_blocks)} code blocks ({', '.join(languages) or 'no language'})")
        
        return f"Markdown document with {', '.join(parts) or 'no structure'}"
    
    def _parse_code(
        self,
        content: str,
        extraction_type: ExtractionType,
        domain: str,
    ) -> StructuredOutput:
        """Parse source code into structured output.
        
        AC-6.1: Parses code into hierarchical structure
        """
        lines = content.split("\n")
        code_blocks: list[CodeBlock] = []
        sections: list[Section] = []
        extracted_items: list[ExtractedItem] = []
        
        # Extract Python-style definitions
        current_block_start: int | None = None
        current_block_lines: list[str] = []
        current_block_name: str | None = None
        current_block_type: str | None = None
        indent_level = 0
        
        for i, line in enumerate(lines, start=1):
            # Detect class definitions
            class_match = re.match(r"^class\s+(\w+)", line)
            if class_match:
                # Save previous block
                if current_block_start is not None:
                    code_blocks.append(CodeBlock(
                        code="\n".join(current_block_lines),
                        language="python",
                        start_line=current_block_start,
                        end_line=i - 1,
                    ))
                    sections.append(Section(
                        title=current_block_name or "Unknown",
                        content=f"{current_block_type}: {current_block_name}",
                        depth=0,
                        start_line=current_block_start,
                        end_line=i - 1,
                    ))
                
                current_block_start = i
                current_block_lines = [line]
                current_block_name = class_match.group(1)
                current_block_type = "class"
                indent_level = 0
                
                extracted_items.append(ExtractedItem(
                    value=class_match.group(1),
                    category="class",
                    confidence_score=1.0,
                    source_positions=[i],
                ))
                continue
            
            # Detect function definitions
            func_match = re.match(r"^(\s*)def\s+(\w+)", line)
            if func_match:
                func_indent = len(func_match.group(1))
                func_name = func_match.group(2)
                
                # Top-level function
                if func_indent == 0:
                    if current_block_start is not None:
                        code_blocks.append(CodeBlock(
                            code="\n".join(current_block_lines),
                            language="python",
                            start_line=current_block_start,
                            end_line=i - 1,
                        ))
                        sections.append(Section(
                            title=current_block_name or "Unknown",
                            content=f"{current_block_type}: {current_block_name}",
                            depth=0,
                            start_line=current_block_start,
                            end_line=i - 1,
                        ))
                    
                    current_block_start = i
                    current_block_lines = [line]
                    current_block_name = func_name
                    current_block_type = "function"
                else:
                    # Method inside class
                    current_block_lines.append(line)
                
                extracted_items.append(ExtractedItem(
                    value=func_name,
                    category="function" if func_indent == 0 else "method",
                    confidence_score=1.0,
                    source_positions=[i],
                ))
                continue
            
            # Accumulate lines in current block
            if current_block_start is not None:
                current_block_lines.append(line)
        
        # Save last block
        if current_block_start is not None:
            code_blocks.append(CodeBlock(
                code="\n".join(current_block_lines),
                language="python",
                start_line=current_block_start,
                end_line=len(lines),
            ))
            sections.append(Section(
                title=current_block_name or "Unknown",
                content=f"{current_block_type}: {current_block_name}",
                depth=0,
                start_line=current_block_start,
                end_line=len(lines),
            ))
        
        # If no blocks found, treat entire content as one block
        if not code_blocks:
            code_blocks.append(CodeBlock(
                code=content,
                language="python",
                start_line=1,
                end_line=len(lines),
            ))
        
        # Generate summary
        class_count = sum(1 for i in extracted_items if i.category == "class")
        func_count = sum(1 for i in extracted_items if i.category in ("function", "method"))
        summary = f"Code with {class_count} classes and {func_count} functions/methods"
        
        return StructuredOutput(
            code_blocks=code_blocks,
            sections=sections,
            extracted_items=extracted_items,
            compressed_summary=summary,
        )

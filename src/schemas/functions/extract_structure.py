"""Schemas for extract_structure function.

WBS-AGT6: extract_structure Function schemas.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 1
"""

from enum import Enum

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Type of artifact being processed.

    Controls the parsing strategy:
    - AUTO: Detect type from content
    - JSON: Parse as JSON structure
    - MARKDOWN: Parse as Markdown document
    - CODE: Parse as source code
    """
    AUTO = "auto"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"


class ExtractionType(str, Enum):
    """Type of extraction to perform.

    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - keywords: Extract key terms
    - concepts: Extract domain concepts
    - entities: Extract named entities
    - outline: Extract document structure
    """
    KEYWORDS = "keywords"
    CONCEPTS = "concepts"
    ENTITIES = "entities"
    OUTLINE = "outline"


class ExtractStructureInput(BaseModel):
    """Input schema for extract_structure function.

    Reference: AC-6.1, AC-6.5
    """
    content: str = Field(
        ...,
        description="Raw text/code content to extract structure from",
    )
    artifact_type: ArtifactType = Field(
        default=ArtifactType.AUTO,
        description="Type of artifact (auto-detected if not specified)",
    )
    extraction_type: ExtractionType = Field(
        default=ExtractionType.OUTLINE,
        description="Type of extraction to perform",
    )
    domain: str = Field(
        default="general",
        description="Domain context: ai-ml | systems | web | general",
    )


class Heading(BaseModel):
    """Heading extracted from document structure.

    Reference: AC-6.2 - Markdown H1-H6 support
    """
    level: int = Field(
        ...,
        ge=1,
        le=6,
        description="Heading level (1-6, like H1-H6)",
    )
    text: str = Field(
        ...,
        description="Heading text content",
    )
    line_number: int | None = Field(
        default=None,
        description="Line number in source for audit",
    )
    children: list["Heading"] = Field(
        default_factory=list,
        description="Nested child headings",
    )


class Section(BaseModel):
    """Section extracted from document structure.

    Reference: AC-6.2 - Sections with depth tracking
    """
    title: str = Field(
        ...,
        description="Section title",
    )
    content: str = Field(
        ...,
        description="Section content",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Nesting depth (0 = top level)",
    )
    start_line: int | None = Field(
        default=None,
        description="Starting line number for audit",
    )
    end_line: int | None = Field(
        default=None,
        description="Ending line number for audit",
    )


class CodeBlock(BaseModel):
    """Code block extracted from content.

    Reference: AC-6.2 - code_blocks in output
    """
    code: str = Field(
        ...,
        description="Code content",
    )
    language: str | None = Field(
        default=None,
        description="Programming language (if detected)",
    )
    start_line: int | None = Field(
        default=None,
        description="Starting line number",
    )
    end_line: int | None = Field(
        default=None,
        description="Ending line number",
    )


class ExtractedItem(BaseModel):
    """Extracted item for keywords/concepts/entities.

    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - Structured items with confidence scores
    - Source locations for audit
    """
    value: str = Field(
        ...,
        description="Extracted value (keyword, concept, or entity)",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)",
    )
    category: str | None = Field(
        default=None,
        description="Category or type of the extracted item",
    )
    source_positions: list[int] = Field(
        default_factory=list,
        description="Character positions in source for audit",
    )


class StructuredOutput(BaseModel):
    """Output schema for extract_structure function.

    Reference: AC-6.2 - headings, sections, code_blocks

    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - extracted: list[dict] → Structured items with confidence scores
    - raw_positions: list → Source locations for audit
    - compressed_summary: str → Max 500 tokens for downstream
    """
    headings: list[Heading] = Field(
        default_factory=list,
        description="Extracted headings (H1-H6)",
    )
    sections: list[Section] = Field(
        default_factory=list,
        description="Extracted sections with depth",
    )
    code_blocks: list[CodeBlock] = Field(
        default_factory=list,
        description="Extracted code blocks",
    )
    extracted_items: list[ExtractedItem] = Field(
        default_factory=list,
        description="Keywords, concepts, or entities",
    )
    raw_positions: list[int] = Field(
        default_factory=list,
        description="Source positions for audit",
    )
    compressed_summary: str | None = Field(
        default=None,
        description="Compressed summary for downstream (max 500 tokens)",
    )


# Update forward references for Heading.children
Heading.model_rebuild()

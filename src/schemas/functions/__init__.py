"""Function-specific schemas for agent functions.

Each agent function has its own input/output schemas defined here.
"""

from src.schemas.functions.extract_structure import (
    ArtifactType,
    ExtractionType,
    ExtractStructureInput,
    StructuredOutput,
    Heading,
    Section,
    CodeBlock,
    ExtractedItem,
)

from src.schemas.functions.summarize_content import (
    DetailLevel,
    SummaryStyle,
    SummarizeContentInput,
    SummarizeContentOutput,
)

from src.schemas.functions.generate_code import (
    TargetLanguage,
    GenerateCodeInput,
    CodeOutput,
)

from src.schemas.functions.analyze_artifact import (
    ArtifactKind,
    AnalysisType,
    Severity,
    AnalyzeArtifactInput,
    Finding,
    AnalysisResult,
)

from src.schemas.functions.validate_against_spec import (
    ViolationSeverity,
    Violation,
    ValidationResult,
    ValidateAgainstSpecInput,
)

from src.schemas.functions.decompose_task import (
    DecomposeTaskInput,
    Subtask,
    TaskDecomposition,
)

__all__ = [
    # extract_structure
    "ArtifactType",
    "ExtractionType",
    "ExtractStructureInput",
    "StructuredOutput",
    "Heading",
    "Section",
    "CodeBlock",
    "ExtractedItem",
    # summarize_content
    "DetailLevel",
    "SummaryStyle",
    "SummarizeContentInput",
    "SummarizeContentOutput",
    # generate_code
    "TargetLanguage",
    "GenerateCodeInput",
    "CodeOutput",
    # analyze_artifact
    "ArtifactKind",
    "AnalysisType",
    "Severity",
    "AnalyzeArtifactInput",
    "Finding",
    "AnalysisResult",
    # validate_against_spec
    "ViolationSeverity",
    "Violation",
    "ValidationResult",
    "ValidateAgainstSpecInput",
    # decompose_task
    "DecomposeTaskInput",
    "Subtask",
    "TaskDecomposition",
]

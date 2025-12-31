"""Function-specific schemas for agent functions.

Each agent function has its own input/output schemas defined here.
"""

from src.schemas.functions.analyze_artifact import (
    AnalysisResult,
    AnalysisType,
    AnalyzeArtifactInput,
    ArtifactKind,
    Finding,
    Severity,
)
from src.schemas.functions.decompose_task import (
    DecomposeTaskInput,
    Subtask,
    TaskDecomposition,
)
from src.schemas.functions.extract_structure import (
    ArtifactType,
    CodeBlock,
    ExtractedItem,
    ExtractionType,
    ExtractStructureInput,
    Heading,
    Section,
    StructuredOutput,
)
from src.schemas.functions.generate_code import (
    CodeOutput,
    GenerateCodeInput,
    TargetLanguage,
)
from src.schemas.functions.summarize_content import (
    DetailLevel,
    SummarizeContentInput,
    SummarizeContentOutput,
    SummaryStyle,
)
from src.schemas.functions.validate_against_spec import (
    ValidateAgainstSpecInput,
    ValidationResult,
    Violation,
    ViolationSeverity,
)


__all__ = [
    "AnalysisResult",
    "AnalysisType",
    "AnalyzeArtifactInput",
    # analyze_artifact
    "ArtifactKind",
    # extract_structure
    "ArtifactType",
    "CodeBlock",
    "CodeOutput",
    # decompose_task
    "DecomposeTaskInput",
    # summarize_content
    "DetailLevel",
    "ExtractStructureInput",
    "ExtractedItem",
    "ExtractionType",
    "Finding",
    "GenerateCodeInput",
    "Heading",
    "Section",
    "Severity",
    "StructuredOutput",
    "Subtask",
    "SummarizeContentInput",
    "SummarizeContentOutput",
    "SummaryStyle",
    # generate_code
    "TargetLanguage",
    "TaskDecomposition",
    "ValidateAgainstSpecInput",
    "ValidationResult",
    "Violation",
    # validate_against_spec
    "ViolationSeverity",
]

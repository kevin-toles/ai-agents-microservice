"""Schemas for generate_code function.

WBS-AGT8: generate_code Function schemas.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 3

Acceptance Criteria:
- AC-8.1: Generates code from natural language spec
- AC-8.2: Returns CodeOutput with language, code, explanation
- AC-8.5: Supports target_language parameter
- AC-8.6: Includes test stubs when include_tests=True
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.citations import Citation


class TargetLanguage(str, Enum):
    """Target programming language for code generation.
    
    Supported languages for AC-8.5:
    - python: Default, generates type-hinted Python 3.10+
    - javascript: ES6+ JavaScript
    - typescript: TypeScript with type annotations
    - java: Java 11+
    - sql: Standard SQL
    - go: Go 1.18+
    - rust: Rust edition 2021
    - cpp: C++17
    """
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    SQL = "sql"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"


class GenerateCodeInput(BaseModel):
    """Input schema for generate_code function.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 3
    
    Attributes:
        specification: Natural language description of what to build
        target_language: Programming language for output (AC-8.5)
        include_tests: Generate test stubs when True (AC-8.6)
        context_artifacts: Related code from handoff cache
        patterns_to_follow: Patterns from CODING_PATTERNS_ANALYSIS
        constraints: Must-have requirements
    """
    specification: str = Field(
        ...,
        description="Natural language specification of what to build",
        min_length=1,
    )
    target_language: TargetLanguage | str = Field(
        default=TargetLanguage.PYTHON,
        description="Target programming language (default: python)",
    )
    include_tests: bool = Field(
        default=False,
        description="Generate test stubs when True (AC-8.6)",
    )
    context_artifacts: list[str] = Field(
        default_factory=list,
        description="Related code from handoff cache for context",
    )
    patterns_to_follow: list[str] = Field(
        default_factory=list,
        description="Design patterns to follow from CODING_PATTERNS_ANALYSIS",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Must-have requirements (e.g., 'Must use async/await')",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "specification": "Create a function that adds two numbers",
                    "target_language": "python",
                    "include_tests": True,
                    "constraints": ["Must have type hints"],
                },
                {
                    "specification": "Create a UserRepository class",
                    "target_language": "python",
                    "include_tests": True,
                    "patterns_to_follow": ["repository-pattern"],
                    "context_artifacts": ["class BaseRepository: ..."],
                },
            ]
        }
    }


class CodeOutput(BaseModel):
    """Output schema for generate_code function.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 3
    
    Attributes:
        code: Generated code
        language: Programming language of the code
        explanation: Why these implementation choices were made
        test_hints: Suggested test cases
        test_code: Generated test code when include_tests=True (AC-8.6)
        compressed_intent: Summary for downstream validation
        citations: Sources used in generation
    """
    code: str = Field(
        ...,
        description="Generated code",
    )
    language: str = Field(
        default="python",
        description="Programming language of the generated code",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of implementation choices",
    )
    test_hints: list[str] = Field(
        default_factory=list,
        description="Suggested test cases for the generated code",
    )
    test_code: Optional[str] = Field(
        default=None,
        description="Generated test code (when include_tests=True)",
    )
    compressed_intent: Optional[str] = Field(
        default=None,
        description="Compressed intent for downstream validation",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations to sources used in generation",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "code": "def add(a: int, b: int) -> int:\n    return a + b",
                    "language": "python",
                    "explanation": "Simple addition function with type hints",
                    "test_hints": ["Test with positive numbers", "Test with zero"],
                    "test_code": "def test_add():\n    assert add(1, 2) == 3",
                }
            ]
        }
    }

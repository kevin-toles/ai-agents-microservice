"""Function utilities package.

Shared utilities for agent functions to reduce code duplication (S1192).
"""

from src.functions.utils.artifact_parser import (
    # Specification requirement checking
    spec_requires_function,
    spec_requires_class,
    spec_requires_docstring,
    # Artifact checking
    ArtifactParser,
    has_function,
    has_function_named,
    has_async_function,
    has_class,
    has_class_named,
    has_docstring,
    has_type_hints,
    has_method,
    extract_function_name,
    extract_class_name,
    extract_entity_name,
    extract_method_name,
    extract_func_name_from_text,
    extract_class_name_from_text,
    extract_method_name_from_text,
)
from src.functions.utils.token_utils import (
    estimate_tokens,
    check_budget,
    tokens_to_chars,
)

__all__ = [
    # Specification requirements
    "spec_requires_function",
    "spec_requires_class",
    "spec_requires_docstring",
    # Artifact parsing
    "ArtifactParser",
    "has_function",
    "has_function_named",
    "has_async_function",
    "has_class",
    "has_class_named",
    "has_docstring",
    "has_type_hints",
    "has_method",
    "extract_function_name",
    "extract_class_name",
    "extract_entity_name",
    "extract_method_name",
    # Violation extraction
    "extract_func_name_from_text",
    "extract_class_name_from_text",
    "extract_method_name_from_text",
    # Token utilities
    "estimate_tokens",
    "check_budget",
    "tokens_to_chars",
]

"""Function utilities package.

Shared utilities for agent functions to reduce code duplication (S1192).
"""

from src.functions.utils.artifact_parser import (
    # Artifact checking
    ArtifactParser,
    extract_class_name,
    extract_class_name_from_text,
    extract_entity_name,
    extract_func_name_from_text,
    extract_function_name,
    extract_method_name,
    extract_method_name_from_text,
    has_async_function,
    has_class,
    has_class_named,
    has_docstring,
    has_function,
    has_function_named,
    has_method,
    has_type_hints,
    spec_requires_class,
    spec_requires_docstring,
    # Specification requirement checking
    spec_requires_function,
)
from src.functions.utils.token_utils import (
    check_budget,
    estimate_tokens,
    tokens_to_chars,
)


__all__ = [
    # Artifact parsing
    "ArtifactParser",
    "check_budget",
    # Token utilities
    "estimate_tokens",
    "extract_class_name",
    "extract_class_name_from_text",
    "extract_entity_name",
    # Violation extraction
    "extract_func_name_from_text",
    "extract_function_name",
    "extract_method_name",
    "extract_method_name_from_text",
    "has_async_function",
    "has_class",
    "has_class_named",
    "has_docstring",
    "has_function",
    "has_function_named",
    "has_method",
    "has_type_hints",
    "spec_requires_class",
    "spec_requires_docstring",
    # Specification requirements
    "spec_requires_function",
    "tokens_to_chars",
]

"""Artifact parsing utilities.

Provides reusable functions for analyzing Python code artifacts,
eliminating duplication across validate_against_spec and other functions.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Functions
Anti-Pattern: S1192 (duplicated string literals)
"""

import re
from typing import Optional


# =============================================================================
# Compiled Regex Patterns (Performance Optimization)
# =============================================================================

# Function patterns
FUNCTION_DEF_PATTERN = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
ASYNC_FUNCTION_DEF_PATTERN = re.compile(r"^\s*async\s+def\s+\w+\s*\(", re.MULTILINE)

# Class patterns
CLASS_DEF_PATTERN = re.compile(r"^\s*class\s+\w+", re.MULTILINE)

# Docstring patterns (after def/class)
DOCSTRING_PATTERN = re.compile(
    r'(def|class)\s+\w+[^:]*:\s*\n\s*["\'][\'"]{2}',
    re.MULTILINE,
)

# Type hint patterns
PARAM_TYPE_HINT_PATTERN = re.compile(r"def\s+\w+\s*\([^)]*:\s*\w+")
RETURN_TYPE_HINT_PATTERN = re.compile(r"->\s*\w+")

# Common non-name words to filter
NON_NAME_WORDS = frozenset({
    "a", "an", "the", "with", "that", "this", "new", "and", "or", "for",
    "in", "on", "to", "from", "by", "as", "is", "are", "was", "were",
})

# Specification requirement patterns
SPEC_FUNCTION_PATTERNS = [
    re.compile(r"\bfunction\b"),
    re.compile(r"\bcreate\s+.*\bfunction\b"),
    re.compile(r"\bdef\b"),
    re.compile(r"\bmethod\b"),
]
SPEC_CLASS_PATTERNS = [
    re.compile(r"\bclass\b"),
    re.compile(r"\bcreate\s+.*\bclass\b"),
]
SPEC_DOCSTRING_PATTERNS = [
    re.compile(r"\bdocstring\b"),
    re.compile(r"\bdocumented\b"),
    re.compile(r"\bwith\s+.*\bdocstring\b"),
]


# =============================================================================
# Specification Requirement Detection
# =============================================================================

def spec_requires_function(spec_text: str) -> bool:
    """Check if specification text requires a function.
    
    Args:
        spec_text: Specification text (should be lowercase).
        
    Returns:
        True if specification requires a function.
    """
    return any(p.search(spec_text) for p in SPEC_FUNCTION_PATTERNS)


def spec_requires_class(spec_text: str) -> bool:
    """Check if specification text requires a class.
    
    Args:
        spec_text: Specification text (should be lowercase).
        
    Returns:
        True if specification requires a class.
    """
    return any(p.search(spec_text) for p in SPEC_CLASS_PATTERNS)


def spec_requires_docstring(spec_text: str) -> bool:
    """Check if specification text requires a docstring.
    
    Args:
        spec_text: Specification text (should be lowercase).
        
    Returns:
        True if specification requires a docstring.
    """
    return any(p.search(spec_text) for p in SPEC_DOCSTRING_PATTERNS)


# =============================================================================
# Function Detection
# =============================================================================

def has_function(artifact: str) -> bool:
    """Check if artifact contains any function definition.
    
    Args:
        artifact: Python code to check.
        
    Returns:
        True if artifact contains a function definition.
        
    Example:
        >>> has_function("def foo(): pass")
        True
        >>> has_function("x = 1")
        False
    """
    return bool(FUNCTION_DEF_PATTERN.search(artifact))


def has_function_named(artifact: str, name: str) -> bool:
    """Check if artifact has a function with specific name.
    
    Args:
        artifact: Python code to check.
        name: Expected function name.
        
    Returns:
        True if function with given name exists.
        
    Example:
        >>> has_function_named("def calculate(x): pass", "calculate")
        True
    """
    pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\("
    return bool(re.search(pattern, artifact, re.MULTILINE))


def has_async_function(artifact: str) -> bool:
    """Check if artifact contains any async function definition.
    
    Args:
        artifact: Python code to check.
        
    Returns:
        True if artifact contains an async function.
    """
    return bool(ASYNC_FUNCTION_DEF_PATTERN.search(artifact))


# =============================================================================
# Class Detection
# =============================================================================

def has_class(artifact: str) -> bool:
    """Check if artifact contains any class definition.
    
    Args:
        artifact: Python code to check.
        
    Returns:
        True if artifact contains a class definition.
        
    Example:
        >>> has_class("class Foo: pass")
        True
        >>> has_class("def foo(): pass")
        False
    """
    return bool(CLASS_DEF_PATTERN.search(artifact))


def has_class_named(artifact: str, name: str) -> bool:
    """Check if artifact has a class with specific name.
    
    Args:
        artifact: Python code to check.
        name: Expected class name.
        
    Returns:
        True if class with given name exists.
        
    Example:
        >>> has_class_named("class Calculator: pass", "Calculator")
        True
    """
    pattern = rf"^\s*class\s+{re.escape(name)}\b"
    return bool(re.search(pattern, artifact, re.MULTILINE))


# =============================================================================
# Documentation Detection
# =============================================================================

def has_docstring(artifact: str) -> bool:
    """Check if artifact contains a docstring after def/class.
    
    Args:
        artifact: Python code to check.
        
    Returns:
        True if artifact has a docstring.
        
    Example:
        >>> has_docstring('def foo():\\n    \"\"\"Doc.\"\"\"\\n    pass')
        True
    """
    return bool(DOCSTRING_PATTERN.search(artifact))


# =============================================================================
# Type Hint Detection
# =============================================================================

def has_type_hints(artifact: str) -> bool:
    """Check if artifact has type hints.
    
    Checks for parameter type hints or return type annotations.
    
    Args:
        artifact: Python code to check.
        
    Returns:
        True if type hints are present.
        
    Example:
        >>> has_type_hints("def foo(x: int) -> str: pass")
        True
        >>> has_type_hints("def foo(x): pass")
        False
    """
    return bool(
        PARAM_TYPE_HINT_PATTERN.search(artifact) or
        RETURN_TYPE_HINT_PATTERN.search(artifact)
    )


# =============================================================================
# Method Detection
# =============================================================================

def has_method(artifact: str, method_name: str) -> bool:
    """Check if artifact has a method with specific name.
    
    Methods are functions with 'self' as first parameter.
    
    Args:
        artifact: Python code to check.
        method_name: Expected method name.
        
    Returns:
        True if method exists.
        
    Example:
        >>> has_method("def calculate(self, x): pass", "calculate")
        True
    """
    pattern = rf"^\s*def\s+{re.escape(method_name)}\s*\(\s*self"
    return bool(re.search(pattern, artifact, re.MULTILINE))


# =============================================================================
# Name Extraction
# =============================================================================

def extract_function_name(text: str) -> Optional[str]:
    """Extract expected function name from specification text.
    
    Parses natural language specification to find function name mentions.
    
    Args:
        text: Specification or criterion text.
        
    Returns:
        Function name if found, None otherwise.
        
    Example:
        >>> extract_function_name("Create a function named 'calculate'")
        'calculate'
    """
    patterns = [
        r"function\s+named\s+['\"]?(\w+)['\"]?",
        r"function\s+called\s+['\"]?(\w+)['\"]?",
        r"['\"](\w+)['\"]?\s+function",
        r"(?:create|implement|write)\s+(?:a\s+)?(\w+)\s+function",
        r"(?:create|implement|write)\s+an?\s+(\w+)\s+function",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).lower()
            if name not in NON_NAME_WORDS:
                return match.group(1)
    return None


def extract_class_name(text: str) -> Optional[str]:
    """Extract expected class name from specification text.
    
    Args:
        text: Specification or criterion text.
        
    Returns:
        Class name if found, None otherwise.
        
    Example:
        >>> extract_class_name("Create a Calculator class")
        'Calculator'
    """
    patterns = [
        r"class\s+named\s+['\"]?(\w+)['\"]?",
        r"class\s+called\s+['\"]?(\w+)['\"]?",
        r"['\"](\w+)['\"]?\s+class",
        r"(?:create|implement)\s+(?:a\s+)?(\w+)\s+class",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).lower()
            if name not in NON_NAME_WORDS:
                return match.group(1)
    return None


def extract_entity_name(text: str, entity_type: str) -> Optional[str]:
    """Extract entity name from criterion text.
    
    Generic extraction for function, class, method names.
    
    Args:
        text: Criterion or specification text.
        entity_type: Type of entity ("function", "class", "method").
        
    Returns:
        Entity name if found, None otherwise.
    """
    patterns = [
        rf"{entity_type}\s+['\"](\w+)['\"]",
        rf"{entity_type}\s+named\s+['\"]?(\w+)['\"]?",
        rf"['\"](\w+)['\"]?\s+{entity_type}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_method_name(text: str) -> Optional[str]:
    """Extract method name from criterion text.
    
    Args:
        text: Criterion text.
        
    Returns:
        Method name if found, None otherwise.
    """
    patterns = [
        r"['\"](\w+)['\"]?\s+method",
        r"method\s+['\"](\w+)['\"]?",
        r"have\s+['\"](\w+)['\"]?\s+method",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# =============================================================================
# ArtifactParser Class (OOP Interface)
# =============================================================================

class ArtifactParser:
    """Object-oriented interface for artifact parsing.
    
    Provides stateful parsing with caching for repeated checks.
    
    Example:
        >>> parser = ArtifactParser("def foo(): pass")
        >>> parser.has_function()
        True
    """
    
    def __init__(self, artifact: str) -> None:
        """Initialize parser with artifact.
        
        Args:
            artifact: Python code to parse.
        """
        self._artifact = artifact
        self._cache: dict[str, bool] = {}
    
    @property
    def artifact(self) -> str:
        """Get the artifact being parsed."""
        return self._artifact
    
    def has_function(self) -> bool:
        """Check if artifact has any function."""
        if "function" not in self._cache:
            self._cache["function"] = has_function(self._artifact)
        return self._cache["function"]
    
    def has_function_named(self, name: str) -> bool:
        """Check if artifact has function with name."""
        key = f"function:{name}"
        if key not in self._cache:
            self._cache[key] = has_function_named(self._artifact, name)
        return self._cache[key]
    
    def has_class(self) -> bool:
        """Check if artifact has any class."""
        if "class" not in self._cache:
            self._cache["class"] = has_class(self._artifact)
        return self._cache["class"]
    
    def has_class_named(self, name: str) -> bool:
        """Check if artifact has class with name."""
        key = f"class:{name}"
        if key not in self._cache:
            self._cache[key] = has_class_named(self._artifact, name)
        return self._cache[key]
    
    def has_docstring(self) -> bool:
        """Check if artifact has docstring."""
        if "docstring" not in self._cache:
            self._cache["docstring"] = has_docstring(self._artifact)
        return self._cache["docstring"]
    
    def has_type_hints(self) -> bool:
        """Check if artifact has type hints."""
        if "type_hints" not in self._cache:
            self._cache["type_hints"] = has_type_hints(self._artifact)
        return self._cache["type_hints"]
    
    def has_method(self, name: str) -> bool:
        """Check if artifact has method with name."""
        key = f"method:{name}"
        if key not in self._cache:
            self._cache[key] = has_method(self._artifact, name)
        return self._cache[key]


# =============================================================================
# Violation Extraction Utilities
# =============================================================================

def extract_func_name_from_text(text: str) -> str:
    """Extract function name from violation or error text.
    
    Args:
        text: Text containing function name reference.
        
    Returns:
        Extracted function name or default 'function_name'.
    """
    match = re.search(r"function\s+['\"]?(\w+)['\"]?", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "function_name"


def extract_class_name_from_text(text: str) -> str:
    """Extract class name from violation or error text.
    
    Args:
        text: Text containing class name reference.
        
    Returns:
        Extracted class name or default 'ClassName'.
    """
    match = re.search(r"class\s+['\"]?(\w+)['\"]?", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "ClassName"


def extract_method_name_from_text(text: str) -> str:
    """Extract method name from violation or error text.
    
    Args:
        text: Text containing method name reference.
        
    Returns:
        Extracted method name or default 'method_name'.
    """
    match = re.search(r"method\s+['\"]?(\w+)['\"]?", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "method_name"


__all__ = [
    # Specification requirement checking
    "spec_requires_function",
    "spec_requires_class",
    "spec_requires_docstring",
    # Artifact checking
    "ArtifactParser",
    "has_function",
    "has_function_named",
    "has_async_function",
    "has_class",
    "has_class_named",
    "has_docstring",
    "has_type_hints",
    "has_method",
    # Name extraction
    "extract_function_name",
    "extract_class_name",
    "extract_entity_name",
    "extract_method_name",
    # Violation extraction
    "extract_func_name_from_text",
    "extract_class_name_from_text",
    "extract_method_name_from_text",
    # Constants
    "NON_NAME_WORDS",
]

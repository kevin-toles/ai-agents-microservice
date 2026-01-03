"""AST Parser Tool for Code Analysis.

WBS Reference: WBS-KB9 - AST Parser Tool
Gap: ast_parser Tool (not implemented)

Provides AST-based code structure extraction for:
- Function/method detection
- Class hierarchy analysis
- Import dependency tracking
- Complexity metrics

Anti-Patterns Avoided:
- #12: Cached parser instances
- S1192: String constants at module level
- S3776: Cognitive complexity < 15 via dispatch pattern
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Module Constants (S1192 Compliance)
# =============================================================================

_DEFAULT_MAX_DEPTH = 10
_DEFAULT_INCLUDE_DOCSTRINGS = True


# =============================================================================
# Enums
# =============================================================================


class NodeType(str, Enum):
    """Types of AST nodes we track."""
    
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    ASSIGN = "assign"
    DECORATOR = "decorator"


# =============================================================================
# Data Models
# =============================================================================


class FunctionInfo(BaseModel):
    """Information about a function or method."""
    
    name: str = Field(..., description="Function name")
    node_type: NodeType = Field(..., description="function or method")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    docstring: str | None = Field(None, description="Function docstring")
    parameters: list[str] = Field(default_factory=list, description="Parameter names")
    decorators: list[str] = Field(default_factory=list, description="Decorator names")
    is_async: bool = Field(False, description="Whether function is async")
    complexity: int = Field(1, description="Cyclomatic complexity estimate")


class ClassInfo(BaseModel):
    """Information about a class."""
    
    name: str = Field(..., description="Class name")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    docstring: str | None = Field(None, description="Class docstring")
    bases: list[str] = Field(default_factory=list, description="Base class names")
    methods: list[FunctionInfo] = Field(default_factory=list, description="Class methods")
    decorators: list[str] = Field(default_factory=list, description="Decorator names")


class ImportInfo(BaseModel):
    """Information about an import statement."""
    
    module: str = Field(..., description="Module being imported")
    names: list[str] = Field(default_factory=list, description="Names imported")
    is_from_import: bool = Field(False, description="Whether it's a 'from' import")
    line_number: int = Field(..., description="Line number")


class ASTParseResult(BaseModel):
    """Result of AST parsing."""
    
    file_path: str = Field(..., description="Source file path")
    language: str = Field("python", description="Programming language")
    classes: list[ClassInfo] = Field(default_factory=list, description="Class definitions")
    functions: list[FunctionInfo] = Field(default_factory=list, description="Top-level functions")
    imports: list[ImportInfo] = Field(default_factory=list, description="Import statements")
    total_lines: int = Field(0, description="Total lines of code")
    complexity_score: int = Field(0, description="Total cyclomatic complexity")
    parse_errors: list[str] = Field(default_factory=list, description="Any parse errors")


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class ASTParserProtocol(Protocol):
    """Protocol for AST parser tools."""
    
    def parse(self, code: str, file_path: str = "<string>") -> ASTParseResult:
        """Parse source code into AST structure."""
        ...
    
    def parse_file(self, file_path: str) -> ASTParseResult:
        """Parse a file into AST structure."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ASTParserConfig:
    """Configuration for AST parser.
    
    Attributes:
        max_depth: Maximum recursion depth for nested structures
        include_docstrings: Whether to extract docstrings
        calculate_complexity: Whether to calculate complexity metrics
    """
    
    max_depth: int = _DEFAULT_MAX_DEPTH
    include_docstrings: bool = _DEFAULT_INCLUDE_DOCSTRINGS
    calculate_complexity: bool = True


# =============================================================================
# Complexity Visitor
# =============================================================================


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that calculates cyclomatic complexity.
    
    Complexity is incremented by:
    - if/elif statements
    - for/while loops
    - try/except blocks
    - and/or operators
    - list/dict/set comprehensions
    """
    
    def __init__(self) -> None:
        """Initialize the visitor."""
        self.complexity = 1  # Base complexity
    
    def visit_If(self, node: ast.If) -> None:
        """Count if statements."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        """Count for loops."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        """Count while loops."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Count except handlers."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Count boolean operators."""
        # Each and/or adds to complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Count list comprehensions."""
        self.complexity += len(node.generators)
        self.generic_visit(node)
    
    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Count dict comprehensions."""
        self.complexity += len(node.generators)
        self.generic_visit(node)
    
    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Count set comprehensions."""
        self.complexity += len(node.generators)
        self.generic_visit(node)


# =============================================================================
# ASTParserTool Implementation
# =============================================================================


class ASTParserTool:
    """Tool for parsing Python code into AST structure.
    
    Extracts:
    - Classes with methods, bases, decorators
    - Functions with parameters, decorators
    - Import statements
    - Complexity metrics
    
    Example:
        >>> parser = ASTParserTool()
        >>> result = parser.parse('''
        ... class MyClass:
        ...     def my_method(self):
        ...         pass
        ... ''')
        >>> print(result.classes[0].name)
        MyClass
    """
    
    def __init__(self, config: ASTParserConfig | None = None) -> None:
        """Initialize the AST parser.
        
        Args:
            config: Optional configuration.
        """
        self._config = config or ASTParserConfig()
    
    def parse(self, code: str, file_path: str = "<string>") -> ASTParseResult:
        """Parse Python source code into structured result.
        
        Args:
            code: Python source code string
            file_path: Optional file path for error messages
            
        Returns:
            ASTParseResult with extracted structure.
        """
        result = ASTParseResult(
            file_path=file_path,
            language="python",
            total_lines=len(code.splitlines()),
        )
        
        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as e:
            result.parse_errors.append(f"Syntax error: {e}")
            return result
        
        # Extract structure
        result.imports = self._extract_imports(tree)
        result.classes = self._extract_classes(tree)
        result.functions = self._extract_functions(tree)
        
        # Calculate total complexity
        if self._config.calculate_complexity:
            result.complexity_score = sum(f.complexity for f in result.functions)
            for cls in result.classes:
                result.complexity_score += sum(m.complexity for m in cls.methods)
        
        return result
    
    def parse_file(self, file_path: str) -> ASTParseResult:
        """Parse a Python file into structured result.
        
        Args:
            file_path: Path to Python source file
            
        Returns:
            ASTParseResult with extracted structure.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                code = f.read()
            return self.parse(code, file_path)
        except OSError as e:
            return ASTParseResult(
                file_path=file_path,
                language="python",
                parse_errors=[f"File error: {e}"],
            )
    
    def _extract_imports(self, tree: ast.Module) -> list[ImportInfo]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_from_import=False,
                        line_number=node.lineno,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from_import=True,
                    line_number=node.lineno,
                ))
        
        return imports
    
    def _extract_classes(self, tree: ast.Module) -> list[ClassInfo]:
        """Extract class definitions from AST."""
        classes = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self._parse_class(node))
        
        return classes
    
    def _parse_class(self, node: ast.ClassDef) -> ClassInfo:
        """Parse a single class definition."""
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")
        
        # Extract decorators
        decorators = self._extract_decorators(node)
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._parse_function(item, is_method=True))
        
        # Get docstring
        docstring = None
        if self._config.include_docstrings:
            docstring = ast.get_docstring(node)
        
        return ClassInfo(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            bases=bases,
            methods=methods,
            decorators=decorators,
        )
    
    def _extract_functions(self, tree: ast.Module) -> list[FunctionInfo]:
        """Extract top-level functions from AST."""
        functions = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._parse_function(node, is_method=False))
        
        return functions
    
    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool = False,
    ) -> FunctionInfo:
        """Parse a single function definition."""
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            parameters.append(arg.arg)
        for arg in node.args.kwonlyargs:
            parameters.append(arg.arg)
        if node.args.vararg:
            parameters.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            parameters.append(f"**{node.args.kwarg.arg}")
        
        # Extract decorators
        decorators = self._extract_decorators(node)
        
        # Get docstring
        docstring = None
        if self._config.include_docstrings:
            docstring = ast.get_docstring(node)
        
        # Calculate complexity
        complexity = 1
        if self._config.calculate_complexity:
            visitor = ComplexityVisitor()
            visitor.visit(node)
            complexity = visitor.complexity
        
        return FunctionInfo(
            name=node.name,
            node_type=NodeType.METHOD if is_method else NodeType.FUNCTION,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            parameters=parameters,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            complexity=complexity,
        )
    
    def _extract_decorators(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> list[str]:
        """Extract decorator names from a node."""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attribute_name(dec))
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(self._get_attribute_name(dec.func))
        return decorators
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeASTParserTool:
    """Fake implementation for testing."""
    
    def __init__(self) -> None:
        """Initialize the fake parser."""
        self._results: dict[str, ASTParseResult] = {}
    
    def set_result(self, file_path: str, result: ASTParseResult) -> None:
        """Set a predefined result for a file path."""
        self._results[file_path] = result
    
    def parse(self, code: str, file_path: str = "<string>") -> ASTParseResult:
        """Return predefined result or empty result."""
        return self._results.get(
            file_path,
            ASTParseResult(file_path=file_path, language="python"),
        )
    
    def parse_file(self, file_path: str) -> ASTParseResult:
        """Return predefined result or empty result."""
        return self._results.get(
            file_path,
            ASTParseResult(file_path=file_path, language="python"),
        )


# =============================================================================
# Singleton Instance
# =============================================================================

_ast_parser_tool: ASTParserTool | None = None


def get_ast_parser_tool(config: ASTParserConfig | None = None) -> ASTParserTool:
    """Get or create singleton ASTParserTool instance.
    
    Args:
        config: Optional configuration. Only used on first call.
        
    Returns:
        Cached ASTParserTool instance.
    """
    global _ast_parser_tool
    if _ast_parser_tool is None:
        _ast_parser_tool = ASTParserTool(config)
    return _ast_parser_tool

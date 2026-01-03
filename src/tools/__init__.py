"""Tools for Agent Functions.

Tools provide specialized capabilities for agents to validate,
analyze, and process code artifacts.

Reference: WBS-KB7 - Code-Orchestrator Tool Integration
Reference: WBS-KB8 - Textbook Search Tool
Reference: WBS-KB9 - AST Parser Tool
Reference: WBS-KB10 - Template Engine Tool
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Tool Protocols
"""

from src.tools.ast_parser import (
    ASTParserConfig,
    ASTParserProtocol,
    ASTParserTool,
    ASTParseResult,
    ClassInfo,
    FakeASTParserTool,
    FunctionInfo,
    get_ast_parser_tool,
    ImportInfo,
    NodeType,
)
from src.tools.code_validation import (
    CodeValidationConfig,
    CodeValidationProtocol,
    CodeValidationResult,
    CodeValidationTool,
    FakeCodeValidationTool,
    StepResult,
    ValidationStep,
)
from src.tools.template_engine import (
    BUILTIN_TEMPLATES,
    FakeTemplateEngineTool,
    get_template_engine_tool,
    TemplateEngineConfig,
    TemplateEngineProtocol,
    TemplateEngineTool,
    TemplateInfo,
    TemplateRenderResult,
)
from src.tools.textbook_search import (
    FakeTextbookSearchTool,
    get_textbook_search_tool,
    TextbookChapter,
    TextbookSearchConfig,
    TextbookSearchProtocol,
    TextbookSearchResult,
    TextbookSearchTool,
)


__all__ = [
    # AST Parser
    "ASTParserConfig",
    "ASTParserProtocol",
    "ASTParserTool",
    "ASTParseResult",
    "ClassInfo",
    "FakeASTParserTool",
    "FunctionInfo",
    "get_ast_parser_tool",
    "ImportInfo",
    "NodeType",
    # Code Validation
    "CodeValidationConfig",
    "CodeValidationProtocol",
    "CodeValidationResult",
    "CodeValidationTool",
    "FakeCodeValidationTool",
    "StepResult",
    "ValidationStep",
    # Template Engine
    "BUILTIN_TEMPLATES",
    "FakeTemplateEngineTool",
    "get_template_engine_tool",
    "TemplateEngineConfig",
    "TemplateEngineProtocol",
    "TemplateEngineTool",
    "TemplateInfo",
    "TemplateRenderResult",
    # Textbook Search
    "FakeTextbookSearchTool",
    "get_textbook_search_tool",
    "TextbookChapter",
    "TextbookSearchConfig",
    "TextbookSearchProtocol",
    "TextbookSearchResult",
    "TextbookSearchTool",
]

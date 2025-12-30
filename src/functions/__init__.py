"""Agent Functions Package.

This package contains stateless agent functions that execute over cached artifacts.
Each function follows the AgentFunction ABC pattern and produces typed outputs.

Agent functions are NOT chat personas - they read from caches and write new state back.

Pattern: Stateless Executor Pattern
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md
"""

from src.functions.base import (
    AgentFunction,
    AgentFunctionProtocol,
    CONTEXT_BUDGET_DEFAULTS,
    ContextBudgetExceededError,
)
from src.functions.extract_structure import ExtractStructureFunction
from src.functions.summarize_content import SummarizeContentFunction
from src.functions.generate_code import GenerateCodeFunction
from src.functions.analyze_artifact import AnalyzeArtifactFunction
from src.functions.validate_against_spec import ValidateAgainstSpecFunction
from src.functions.decompose_task import DecomposeTaskFunction

__all__ = [
    # Base class and utilities
    "AgentFunction",
    "AgentFunctionProtocol",
    "CONTEXT_BUDGET_DEFAULTS",
    "ContextBudgetExceededError",
    # Agent functions
    "ExtractStructureFunction",
    "SummarizeContentFunction",
    "GenerateCodeFunction",
    "AnalyzeArtifactFunction",
    "ValidateAgainstSpecFunction",
    "DecomposeTaskFunction",
]

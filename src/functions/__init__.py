"""Agent Functions Package.

This package contains stateless agent functions that execute over cached artifacts.
Each function follows the AgentFunction ABC pattern and produces typed outputs.

Agent functions are NOT chat personas - they read from caches and write new state back.

Pattern: Stateless Executor Pattern
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md
"""

from src.core.constants import CONTEXT_BUDGET_DEFAULTS
from src.functions.analyze_artifact import AnalyzeArtifactFunction
from src.functions.base import (
    AgentFunction,
    AgentFunctionProtocol,
    ContextBudgetExceededError,
)
from src.functions.cross_reference import CrossReferenceFunction
from src.functions.decompose_task import DecomposeTaskFunction
from src.functions.extract_structure import ExtractStructureFunction
from src.functions.generate_code import GenerateCodeFunction
from src.functions.summarize_content import SummarizeContentFunction
from src.functions.synthesize_outputs import SynthesizeOutputsFunction
from src.functions.validate_against_spec import ValidateAgainstSpecFunction


__all__ = [
    "CONTEXT_BUDGET_DEFAULTS",
    # Base class and utilities
    "AgentFunction",
    "AgentFunctionProtocol",
    "AnalyzeArtifactFunction",
    "ContextBudgetExceededError",
    "CrossReferenceFunction",
    "DecomposeTaskFunction",
    # Agent functions
    "ExtractStructureFunction",
    "GenerateCodeFunction",
    "SummarizeContentFunction",
    "SynthesizeOutputsFunction",
    "ValidateAgainstSpecFunction",
]

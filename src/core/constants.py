"""Kitchen Brigade service constants and configuration values.

Provides centralized constants for the Kitchen Brigade architecture:
- Service ports and roles
- Default timeout values
- ADK state prefixes
- Context budget defaults

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md
"""

from enum import IntEnum


# =============================================================================
# Kitchen Brigade Service Ports
# =============================================================================
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Kitchen Brigade Architecture

class ServicePort(IntEnum):
    """Standard ports for Kitchen Brigade services.
    
    Port allocation:
    - 8080: llm-gateway (Router) - External API entry point
    - 8081: semantic-search-service (Cookbook) - Vector search
    - 8082: ai-agents (Expeditor) - THIS SERVICE - Workflow execution
    - 8083: code-orchestrator (Sous Chef) - Code analysis
    - 8084: audit-service (Auditor) - Citation tracking
    - 8085: inference-service (Line Cook) - LLM inference
    - 8086: code-reference-engine (Pantry) - Code reference data
    """
    LLM_GATEWAY = 8080
    SEMANTIC_SEARCH = 8081
    AI_AGENTS = 8082  # This service (Expeditor)
    CODE_ORCHESTRATOR = 8083
    AUDIT_SERVICE = 8084
    INFERENCE_SERVICE = 8085
    CODE_REFERENCE_ENGINE = 8086


# Service role mapping (Kitchen Brigade metaphor)
SERVICE_ROLES: dict[str, str] = {
    "llm-gateway": "Router",
    "semantic-search-service": "Cookbook",
    "ai-agents": "Expeditor",
    "code-orchestrator": "Sous Chef",
    "audit-service": "Auditor",
    "inference-service": "Line Cook",
    "code-reference-engine": "Pantry",
}


# =============================================================================
# Default Timeout Values
# =============================================================================

class Timeouts:
    """Default timeout values in seconds.
    
    These can be overridden via Settings or per-request.
    """
    # HTTP client defaults
    HTTP_DEFAULT: float = 30.0
    HTTP_INFERENCE: float = 120.0  # LLM calls can be slow
    HTTP_SEARCH: float = 10.0
    HTTP_AUDIT: float = 5.0
    
    # Pipeline execution
    PIPELINE_DEFAULT: float = 300.0  # 5 minutes
    PIPELINE_MAX: float = 600.0  # 10 minutes
    
    # Cache TTLs
    CACHE_USER_TTL: int = 86400  # 24 hours (user: prefix)
    CACHE_TEMP_TTL: int = 3600  # 1 hour (temp: prefix)


# =============================================================================
# ADK State Prefixes
# =============================================================================
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → ADK State Management

class StatePrefix:
    """ADK state key prefixes for different storage tiers.
    
    Storage tiers:
    - TEMP: Pipeline-local state, cleared after execution
    - USER: User session state, Redis with 24h TTL
    - APP: Permanent application state, Qdrant/Neo4j
    
    Example:
        ```python
        key = f"{StatePrefix.USER}{user_id}:{resource_type}"
        ```
    """
    TEMP = "temp:"  # Pipeline-local, in-memory
    USER = "user:"  # Redis, 24h TTL
    APP = "app:"    # Qdrant/Neo4j, permanent


# =============================================================================
# Token Estimation Constants (S1192 - No Duplicated Literals)
# =============================================================================
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Token Budget Allocation

# Characters per token for budget estimation
# Industry standard approximation: 1 token ≈ 4 characters
CHARS_PER_TOKEN: int = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    Uses the industry standard approximation of ~4 characters per token.
    
    Args:
        text: Text to estimate tokens for.
        
    Returns:
        Estimated token count.
        
    Example:
        >>> estimate_tokens("Hello world!")  # 12 chars
        3
    """
    return len(text) // CHARS_PER_TOKEN


# =============================================================================
# Context Budget Defaults (S1192 - Single Source of Truth)
# =============================================================================
# Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Token Budget Allocation
# WBS: AGT5, AGT6-13 define these budgets

# Token limits per agent function
# Format: {"input": input_tokens, "output": output_tokens}
CONTEXT_BUDGET_DEFAULTS: dict[str, dict[str, int]] = {
    # WBS-AGT6: extract_structure (AC-6.3)
    "extract_structure": {"input": 16384, "output": 2048},
    # WBS-AGT7: summarize_content (AC-7.3)
    "summarize_content": {"input": 8192, "output": 4096},
    # WBS-AGT8: generate_code (AC-8.3)
    "generate_code": {"input": 4096, "output": 8192},
    # WBS-AGT9: analyze_artifact (AC-9.3)
    "analyze_artifact": {"input": 16384, "output": 2048},
    # WBS-AGT10: validate_against_spec (AC-10.3)
    "validate_against_spec": {"input": 4096, "output": 1024},
    # WBS-AGT11: synthesize_outputs (AC-11.3)
    "synthesize_outputs": {"input": 8192, "output": 4096},
    # WBS-AGT12: decompose_task (AC-12.3)
    "decompose_task": {"input": 4096, "output": 2048},
    # WBS-AGT13: cross_reference (AC-13.3)
    "cross_reference": {"input": 2048, "output": 4096},
}

# Default budget for unknown/new functions
DEFAULT_CONTEXT_BUDGET: dict[str, int] = {"input": 4096, "output": 2048}


def get_context_budget(function_name: str) -> dict[str, int]:
    """Get the context budget for an agent function.
    
    Args:
        function_name: Name of the agent function.
    
    Returns:
        Dict with 'input' and 'output' token limits.
    
    Example:
        >>> budget = get_context_budget("extract_structure")
        >>> budget
        {'input': 16384, 'output': 2048}
    """
    return CONTEXT_BUDGET_DEFAULTS.get(function_name, DEFAULT_CONTEXT_BUDGET)


# =============================================================================
# API Versioning
# =============================================================================

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# =============================================================================
# Health Check Constants
# =============================================================================

HEALTH_CHECK_PATH = "/health"
READINESS_CHECK_PATH = "/ready"
LIVENESS_CHECK_PATH = "/live"


# =============================================================================
# Default Configuration Values
# =============================================================================

DEFAULT_MAX_PIPELINE_STAGES = 10
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ENVIRONMENT = "development"

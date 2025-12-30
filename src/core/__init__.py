"""Core module - Configuration, logging, HTTP clients, and shared utilities.

Kitchen Brigade Architecture:
    ai-agents acts as the Expeditor (:8082), orchestrating workflow
    execution and coordinating with downstream services.

Exports:
    - Settings, get_settings: Pydantic Settings configuration
    - configure_logging, get_logger: Structured logging (structlog)
    - HTTPClientFactory, ServiceName, get_http_client_factory: HTTP clients
    - ServicePort, StatePrefix, Timeouts: Kitchen Brigade constants
    - Exception classes: AgentError, AgentConnectionError, etc.

Reference: WBS-AGT2, AGENT_FUNCTIONS_ARCHITECTURE.md
"""

from src.core.config import Settings, get_settings
from src.core.constants import (
    ServicePort,
    StatePrefix,
    Timeouts,
    SERVICE_ROLES,
    CONTEXT_BUDGET_DEFAULTS,
    DEFAULT_CONTEXT_BUDGET,
    get_context_budget,
    API_VERSION,
    API_PREFIX,
)
from src.core.exceptions import (
    AgentError,
    AgentConnectionError,
    AgentExecutionError,
    AgentValidationError,
    AgentTimeoutError,
    ToolExecutionError,
    PlanningError,
)
from src.core.http import (
    HTTPClientFactory,
    ServiceName,
    get_http_client_factory,
)
from src.core.logging import configure_logging, get_logger

__all__ = [
    # Configuration
    "Settings",
    "get_settings",
    # Logging
    "configure_logging",
    "get_logger",
    # HTTP Clients
    "HTTPClientFactory",
    "ServiceName",
    "get_http_client_factory",
    # Constants
    "ServicePort",
    "StatePrefix",
    "Timeouts",
    "SERVICE_ROLES",
    "CONTEXT_BUDGET_DEFAULTS",
    "DEFAULT_CONTEXT_BUDGET",
    "get_context_budget",
    "API_VERSION",
    "API_PREFIX",
    # Exceptions
    "AgentError",
    "AgentConnectionError",
    "AgentExecutionError",
    "AgentValidationError",
    "AgentTimeoutError",
    "ToolExecutionError",
    "PlanningError",
]

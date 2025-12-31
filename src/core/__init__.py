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
    API_PREFIX,
    API_VERSION,
    CONTEXT_BUDGET_DEFAULTS,
    DEFAULT_CONTEXT_BUDGET,
    SERVICE_ROLES,
    ServicePort,
    StatePrefix,
    Timeouts,
    get_context_budget,
)
from src.core.exceptions import (
    AgentConnectionError,
    AgentError,
    AgentExecutionError,
    AgentTimeoutError,
    AgentValidationError,
    PlanningError,
    ToolExecutionError,
)
from src.core.http import (
    HTTPClientFactory,
    ServiceName,
    get_http_client_factory,
)
from src.core.logging import configure_logging, get_logger


__all__ = [
    "API_PREFIX",
    "API_VERSION",
    "CONTEXT_BUDGET_DEFAULTS",
    "DEFAULT_CONTEXT_BUDGET",
    "SERVICE_ROLES",
    "AgentConnectionError",
    # Exceptions
    "AgentError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentValidationError",
    # HTTP Clients
    "HTTPClientFactory",
    "PlanningError",
    "ServiceName",
    # Constants
    "ServicePort",
    # Configuration
    "Settings",
    "StatePrefix",
    "Timeouts",
    "ToolExecutionError",
    # Logging
    "configure_logging",
    "get_context_budget",
    "get_http_client_factory",
    "get_logger",
    "get_settings",
]

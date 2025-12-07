"""Custom exceptions for AI Agents service.

Follows exception naming conventions from Comp_Static_Analysis_Report issues #7, #13.
All exceptions are namespaced to avoid shadowing Python builtins.

Pattern: Namespaced Custom Exceptions
Source: CODING_PATTERNS_ANALYSIS.md Anti-Pattern #7, #13
"""

from typing import Any


class AgentError(Exception):
    """Base exception for all agent-related errors.
    
    All agent exceptions inherit from this class to enable
    catching any agent error with a single except clause.
    """
    
    def __init__(self, message: str, agent_name: str | None = None) -> None:
        """Initialize agent error.
        
        Args:
            message: Error description
            agent_name: Name of the agent that raised the error
        """
        self.agent_name = agent_name
        super().__init__(message)


class AgentConnectionError(AgentError):
    """Raised when connection to an external service fails.
    
    Distinct from Python's built-in ConnectionError to avoid
    exception shadowing (Comp_Static_Analysis #7).
    """
    
    def __init__(
        self,
        message: str,
        service: str,
        host: str | None = None,
        port: int | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize connection error.
        
        Args:
            message: Error description
            service: Name of the service that failed
            host: Host that was unreachable
            port: Port that was unreachable
            agent_name: Name of the agent if applicable
        """
        self.service = service
        self.host = host
        self.port = port
        super().__init__(message, agent_name)


class AgentExecutionError(AgentError):
    """Raised when agent execution fails.
    
    This covers general execution failures that don't fit
    more specific error categories.
    """
    
    def __init__(
        self,
        message: str,
        step: str,
        cause: Exception | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize execution error.
        
        Args:
            message: Error description
            step: The workflow step that failed
            cause: Original exception that caused this error
            agent_name: Name of the agent that failed
        """
        self.step = step
        self.cause = cause
        if cause:
            self.__cause__ = cause
        super().__init__(message, agent_name)


class AgentValidationError(AgentError):
    """Raised when input validation fails.
    
    Distinct from Python's built-in ValueError to provide
    structured error information for API responses.
    """
    
    def __init__(
        self,
        message: str,
        field: str,
        value: Any | None = None,
        errors: list[dict[str, str]] | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize validation error.
        
        Args:
            message: Error description
            field: The field that failed validation
            value: The invalid value
            errors: List of validation errors for multiple fields
            agent_name: Name of the agent if applicable
        """
        self.field = field
        self.value = value
        self.errors = errors
        super().__init__(message, agent_name)


class AgentTimeoutError(AgentError):
    """Raised when agent execution exceeds timeout.
    
    Distinct from Python's built-in TimeoutError to avoid
    exception shadowing (Comp_Static_Analysis #7).
    """
    
    def __init__(
        self,
        message: str,
        operation: str,
        timeout_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize timeout error.
        
        Args:
            message: Error description
            operation: The operation that timed out
            timeout_seconds: The timeout that was exceeded
            elapsed_seconds: Actual time elapsed
            agent_name: Name of the agent that timed out
        """
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        super().__init__(message, agent_name)


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails.
    
    Tools are the functions that agents can invoke to
    interact with external systems (search, graph, etc.).
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        agent_name: str | None = None,
    ) -> None:
        """Initialize tool error.
        
        Args:
            message: Error description
            tool_name: Name of the tool that failed
            agent_name: Name of the agent that invoked the tool
        """
        self.tool_name = tool_name
        super().__init__(message, agent_name)


class PlanningError(AgentError):
    """Raised when agent planning fails.
    
    Planning errors occur when the agent cannot create
    a valid execution plan for the given task.
    """
    
    def __init__(
        self,
        message: str,
        task: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Initialize planning error.
        
        Args:
            message: Error description
            task: The task that failed planning
            agent_name: Name of the agent that failed to plan
        """
        self.task = task
        super().__init__(message, agent_name)


class ClientError(AgentError):
    """Raised when an external service client fails.
    
    This is the base class for service-specific client errors.
    """
    
    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: int | None = None,
    ) -> None:
        """Initialize client error.
        
        Args:
            message: Error description
            service_name: Name of the external service
            status_code: HTTP status code if applicable
        """
        self.service_name = service_name
        self.status_code = status_code
        super().__init__(message)


class GatewayClientError(ClientError):
    """Raised when LLM Gateway client fails."""
    
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message, "llm-gateway", status_code)


class SearchClientError(ClientError):
    """Raised when Semantic Search client fails."""
    
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message, "semantic-search", status_code)


class GraphClientError(ClientError):
    """Raised when Graph (Neo4j) client fails."""
    
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message, "neo4j", status_code)

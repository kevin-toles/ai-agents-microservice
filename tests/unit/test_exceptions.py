"""Unit tests for custom exceptions.

TDD Phase: RED - These tests define expected behavior.
Pattern: Custom exception hierarchy
Source: Comp_Static_Analysis_Report - Exception patterns
"""

import pytest

from src.core.exceptions import (
    AgentError,
    AgentConnectionError,
    AgentExecutionError,
    AgentValidationError,
    AgentTimeoutError,
)


class TestAgentError:
    """Tests for base AgentError exception."""
    
    def test_agent_error_is_exception(self) -> None:
        """Test that AgentError inherits from Exception."""
        assert issubclass(AgentError, Exception)
    
    def test_agent_error_message(self) -> None:
        """Test that AgentError stores message."""
        error = AgentError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_agent_error_can_be_raised(self) -> None:
        """Test that AgentError can be raised and caught."""
        with pytest.raises(AgentError):
            raise AgentError("Test")
    
    def test_agent_error_caught_as_exception(self) -> None:
        """Test that AgentError can be caught as Exception."""
        with pytest.raises(Exception):
            raise AgentError("Test")


class TestAgentConnectionError:
    """Tests for AgentConnectionError exception."""
    
    def test_connection_error_inherits_agent_error(self) -> None:
        """Test that AgentConnectionError inherits from AgentError."""
        assert issubclass(AgentConnectionError, AgentError)
    
    def test_connection_error_stores_service(self) -> None:
        """Test that AgentConnectionError stores service name."""
        error = AgentConnectionError(
            message="Connection failed",
            service="neo4j",
        )
        
        assert error.service == "neo4j"
        assert "Connection failed" in str(error)
    
    def test_connection_error_stores_host(self) -> None:
        """Test that AgentConnectionError stores host info."""
        error = AgentConnectionError(
            message="Connection refused",
            service="qdrant",
            host="localhost",
            port=6333,
        )
        
        assert error.host == "localhost"
        assert error.port == 6333
    
    def test_connection_error_without_optional_fields(self) -> None:
        """Test AgentConnectionError with only required fields."""
        error = AgentConnectionError(
            message="Generic connection error",
            service="llm-gateway",
        )
        
        assert error.service == "llm-gateway"
        assert error.host is None
        assert error.port is None


class TestAgentExecutionError:
    """Tests for AgentExecutionError exception."""
    
    def test_execution_error_inherits_agent_error(self) -> None:
        """Test that AgentExecutionError inherits from AgentError."""
        assert issubclass(AgentExecutionError, AgentError)
    
    def test_execution_error_stores_step(self) -> None:
        """Test that AgentExecutionError stores workflow step."""
        error = AgentExecutionError(
            message="Step failed",
            step="analyze_source",
        )
        
        assert error.step == "analyze_source"
    
    def test_execution_error_stores_cause(self) -> None:
        """Test that AgentExecutionError stores original cause."""
        original = ValueError("Invalid input")
        error = AgentExecutionError(
            message="Execution failed",
            step="search_taxonomy",
            cause=original,
        )
        
        assert error.cause is original
    
    def test_execution_error_chains_exceptions(self) -> None:
        """Test that cause is properly chained."""
        original = ValueError("Root cause")
        error = AgentExecutionError(
            message="Wrapper",
            step="traverse_graph",
            cause=original,
        )
        
        # Should be able to access the chain via cause attribute
        assert error.cause is original


class TestAgentValidationError:
    """Tests for AgentValidationError exception."""
    
    def test_validation_error_inherits_agent_error(self) -> None:
        """Test that AgentValidationError inherits from AgentError."""
        assert issubclass(AgentValidationError, AgentError)
    
    def test_validation_error_stores_field(self) -> None:
        """Test that AgentValidationError stores field name."""
        error = AgentValidationError(
            message="Invalid value",
            field="chapter_number",
        )
        
        assert error.field == "chapter_number"
    
    def test_validation_error_stores_value(self) -> None:
        """Test that AgentValidationError stores invalid value."""
        error = AgentValidationError(
            message="Must be positive",
            field="chapter_number",
            value=-1,
        )
        
        assert error.value == -1
    
    def test_validation_error_for_multiple_fields(self) -> None:
        """Test AgentValidationError with multiple field errors."""
        error = AgentValidationError(
            message="Multiple validation errors",
            field="request",
            errors=[
                {"field": "book", "error": "required"},
                {"field": "chapter", "error": "must be positive"},
            ],
        )
        
        assert error.errors is not None
        assert len(error.errors) == 2


class TestAgentTimeoutError:
    """Tests for AgentTimeoutError exception."""
    
    def test_timeout_error_inherits_agent_error(self) -> None:
        """Test that AgentTimeoutError inherits from AgentError."""
        assert issubclass(AgentTimeoutError, AgentError)
    
    def test_timeout_error_stores_operation(self) -> None:
        """Test that AgentTimeoutError stores operation name."""
        error = AgentTimeoutError(
            message="Operation timed out",
            operation="search_taxonomy",
        )
        
        assert error.operation == "search_taxonomy"
    
    def test_timeout_error_stores_timeout_seconds(self) -> None:
        """Test that AgentTimeoutError stores timeout duration."""
        error = AgentTimeoutError(
            message="Timed out after 30 seconds",
            operation="traverse_graph",
            timeout_seconds=30.0,
        )
        
        assert error.timeout_seconds == pytest.approx(30.0)
    
    def test_timeout_error_stores_elapsed(self) -> None:
        """Test that AgentTimeoutError can store elapsed time."""
        error = AgentTimeoutError(
            message="Exceeded timeout",
            operation="llm_synthesis",
            timeout_seconds=60.0,
            elapsed_seconds=62.5,
        )
        
        assert error.elapsed_seconds == pytest.approx(62.5)


class TestExceptionHierarchy:
    """Tests for the overall exception hierarchy."""
    
    def test_all_exceptions_inherit_from_agent_error(self) -> None:
        """Test that all custom exceptions inherit from AgentError."""
        exceptions = [
            AgentConnectionError,
            AgentExecutionError,
            AgentValidationError,
            AgentTimeoutError,
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, AgentError), f"{exc_class} should inherit from AgentError"
    
    def test_exceptions_do_not_shadow_builtins(self) -> None:
        """Test that exceptions don't shadow builtin names.
        
        Anti-Pattern: S5717 (exception shadowing)
        Source: Comp_Static_Analysis_Report
        """
        # These should NOT be the same as builtins
        assert AgentConnectionError is not ConnectionError
        assert AgentTimeoutError is not TimeoutError
        assert AgentError is not Exception
    
    def test_catching_agent_error_catches_all_custom(self) -> None:
        """Test that catching AgentError catches all custom exceptions."""
        exceptions_to_test = [
            AgentConnectionError("test", service="test"),
            AgentExecutionError("test", step="test"),
            AgentValidationError("test", field="test"),
            AgentTimeoutError("test", operation="test"),
        ]
        
        for exc in exceptions_to_test:
            try:
                raise exc
            except AgentError as e:
                # Should be caught
                assert e is exc
            except Exception:
                pytest.fail(f"{type(exc).__name__} was not caught by AgentError handler")

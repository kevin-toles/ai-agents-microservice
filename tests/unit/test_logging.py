"""Unit tests for src/core/logging module.

Tests structured logging configuration and logger creation.

Reference: WBS-AGT2 AC-2.2
"""

import pytest
import structlog
from unittest.mock import patch, MagicMock

from src.core.logging import (
    configure_logging,
    get_logger,
    add_service_context,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""
    
    def test_configure_logging_development(self) -> None:
        """Test logging configuration for development environment."""
        with patch("src.core.logging.get_settings") as mock_settings:
            mock_settings.return_value.environment = "development"
            mock_settings.return_value.log_level = "DEBUG"
            mock_settings.return_value.service_name = "ai-agents"
            
            with patch("src.core.logging.structlog.configure") as mock_configure:
                # Reset configured flag for test
                import src.core.logging as logging_module
                logging_module._configured = False
                
                configure_logging()
                
                mock_configure.assert_called_once()
    
    def test_configure_logging_production(self) -> None:
        """Test logging configuration for production environment."""
        with patch("src.core.logging.get_settings") as mock_settings:
            mock_settings.return_value.environment = "production"
            mock_settings.return_value.log_level = "INFO"
            mock_settings.return_value.service_name = "ai-agents"
            
            with patch("src.core.logging.structlog.configure") as mock_configure:
                # Reset configured flag for test
                import src.core.logging as logging_module
                logging_module._configured = False
                
                configure_logging()
                
                mock_configure.assert_called_once()
    
    def test_configure_logging_idempotent(self) -> None:
        """Test that configure_logging can be called multiple times safely."""
        with patch("src.core.logging.get_settings") as mock_settings:
            mock_settings.return_value.environment = "development"
            mock_settings.return_value.log_level = "INFO"
            mock_settings.return_value.service_name = "ai-agents"
            
            # Reset the configured flag
            import src.core.logging as logging_module
            logging_module._configured = False
            
            # Calling multiple times should not raise errors
            configure_logging()
            configure_logging()
            
            # If we get here without exception, the test passes
            assert True


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_with_name(self) -> None:
        """Test getting a named logger."""
        with patch("src.core.logging.structlog.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            
            logger = get_logger("test_module")
            
            mock_get.assert_called_once_with("test_module")
    
    def test_get_logger_without_name(self) -> None:
        """Test getting logger without explicit name."""
        with patch("src.core.logging.structlog.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            
            logger = get_logger()
            
            mock_get.assert_called_once_with(None)


class TestAddServiceContext:
    """Tests for add_service_context processor."""
    
    def test_adds_service_context(self) -> None:
        """Test that service context is added to log events."""
        with patch("src.core.logging.get_settings") as mock_settings:
            mock_settings.return_value.service_name = "ai-agents"
            mock_settings.return_value.environment = "test"
            
            event_dict = {"event": "test_event"}
            result = add_service_context(None, "info", event_dict)
            
            assert result["service"] == "ai-agents"
            assert result["environment"] == "test"
            assert result["role"] == "Expeditor"
    
    def test_preserves_existing_fields(self) -> None:
        """Test that existing fields are preserved."""
        with patch("src.core.logging.get_settings") as mock_settings:
            mock_settings.return_value.service_name = "ai-agents"
            mock_settings.return_value.environment = "test"
            
            event_dict = {
                "event": "test_event",
                "user_id": "123",
                "custom_field": "value",
            }
            result = add_service_context(None, "info", event_dict)
            
            assert result["user_id"] == "123"
            assert result["custom_field"] == "value"

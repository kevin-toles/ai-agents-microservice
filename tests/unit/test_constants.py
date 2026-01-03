"""Unit tests for src/core/constants module.

Tests Kitchen Brigade service constants and configuration values.

Reference: WBS-AGT2 AC-2.5
"""

import pytest

from src.core.constants import (
    ServicePort,
    SERVICE_ROLES,
    Timeouts,
    StatePrefix,
    CONTEXT_BUDGET_DEFAULTS,
    get_context_budget,
    API_VERSION,
    API_PREFIX,
    HEALTH_CHECK_PATH,
    DEFAULT_MAX_PIPELINE_STAGES,
)


class TestServicePort:
    """Tests for ServicePort enum."""
    
    def test_kitchen_brigade_ports(self) -> None:
        """Test all Kitchen Brigade service ports are defined correctly."""
        assert ServicePort.LLM_GATEWAY == 8080
        assert ServicePort.SEMANTIC_SEARCH == 8081
        assert ServicePort.AI_AGENTS == 8082
        assert ServicePort.CODE_ORCHESTRATOR == 8083
        assert ServicePort.AUDIT_SERVICE == 8084
        assert ServicePort.INFERENCE_SERVICE == 8085
        assert ServicePort.CODE_REFERENCE_ENGINE == 8086
    
    def test_service_port_is_int(self) -> None:
        """Test ServicePort values are integers."""
        assert isinstance(ServicePort.AI_AGENTS.value, int)
        # IntEnum allows direct integer comparison
        assert ServicePort.AI_AGENTS == 8082


class TestServiceRoles:
    """Tests for SERVICE_ROLES mapping."""
    
    def test_all_roles_defined(self) -> None:
        """Test all Kitchen Brigade roles are mapped."""
        assert SERVICE_ROLES["llm-gateway"] == "Router"
        assert SERVICE_ROLES["semantic-search-service"] == "Cookbook"
        assert SERVICE_ROLES["ai-agents"] == "Expeditor"
        assert SERVICE_ROLES["code-orchestrator"] == "Sous Chef"
        assert SERVICE_ROLES["audit-service"] == "Auditor"
        assert SERVICE_ROLES["inference-service"] == "Line Cook"
        assert SERVICE_ROLES["code-reference-engine"] == "Pantry"
    
    def test_ai_agents_is_expeditor(self) -> None:
        """Test ai-agents has Expeditor role."""
        assert SERVICE_ROLES["ai-agents"] == "Expeditor"


class TestTimeouts:
    """Tests for Timeouts class."""
    
    def test_http_timeouts_defined(self) -> None:
        """Test HTTP timeout values are defined."""
        assert Timeouts.HTTP_DEFAULT == pytest.approx(30.0)
        assert Timeouts.HTTP_INFERENCE == pytest.approx(120.0)
        assert Timeouts.HTTP_SEARCH == pytest.approx(10.0)
        assert Timeouts.HTTP_AUDIT == pytest.approx(5.0)
    
    def test_pipeline_timeouts_defined(self) -> None:
        """Test pipeline timeout values are defined."""
        assert Timeouts.PIPELINE_DEFAULT == pytest.approx(300.0)
        assert Timeouts.PIPELINE_MAX == pytest.approx(600.0)
    
    def test_cache_ttls_defined(self) -> None:
        """Test cache TTL values are defined."""
        assert Timeouts.CACHE_USER_TTL == 86400  # 24 hours
        assert Timeouts.CACHE_TEMP_TTL == 3600   # 1 hour


class TestStatePrefix:
    """Tests for StatePrefix class."""
    
    def test_adk_state_prefixes(self) -> None:
        """Test ADK state prefixes are defined correctly."""
        assert StatePrefix.TEMP == "temp:"
        assert StatePrefix.USER == "user:"
        assert StatePrefix.APP == "app:"
    
    def test_prefix_format(self) -> None:
        """Test prefixes end with colon for key building."""
        assert StatePrefix.TEMP.endswith(":")
        assert StatePrefix.USER.endswith(":")
        assert StatePrefix.APP.endswith(":")


class TestContextBudgetDefaults:
    """Tests for CONTEXT_BUDGET_DEFAULTS and get_context_budget."""
    
    def test_defaults_defined(self) -> None:
        """Test context budget defaults are defined."""
        assert "extract_structure" in CONTEXT_BUDGET_DEFAULTS
        assert "summarize_content" in CONTEXT_BUDGET_DEFAULTS
        assert "cross_reference" in CONTEXT_BUDGET_DEFAULTS
    
    def test_budget_format(self) -> None:
        """Test budget values are dicts with input/output keys."""
        budget = CONTEXT_BUDGET_DEFAULTS["extract_structure"]
        assert isinstance(budget, dict)
        assert "input" in budget
        assert "output" in budget
        assert budget == {"input": 16384, "output": 2048}
    
    def test_get_context_budget_known_function(self) -> None:
        """Test get_context_budget returns correct budget for known function."""
        budget = get_context_budget("extract_structure")
        
        assert budget["input"] == 16384
        assert budget["output"] == 2048
    
    def test_get_context_budget_unknown_function(self) -> None:
        """Test get_context_budget returns default for unknown function."""
        budget = get_context_budget("unknown_function")
        
        assert budget["input"] == 4096  # default input
        assert budget["output"] == 2048  # default output


class TestAPIConstants:
    """Tests for API versioning constants."""
    
    def test_api_version(self) -> None:
        """Test API version is defined."""
        assert API_VERSION == "v1"
    
    def test_api_prefix(self) -> None:
        """Test API prefix is correctly formatted."""
        assert API_PREFIX == "/api/v1"


class TestHealthCheckConstants:
    """Tests for health check path constants."""
    
    def test_health_check_path(self) -> None:
        """Test health check path is defined."""
        assert HEALTH_CHECK_PATH == "/health"


class TestDefaultConstants:
    """Tests for default configuration constants."""
    
    def test_max_pipeline_stages(self) -> None:
        """Test default max pipeline stages."""
        assert DEFAULT_MAX_PIPELINE_STAGES == 10

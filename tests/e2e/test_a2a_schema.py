"""A2A Agent Card Schema Validation E2E Tests.

WBS-PI7: End-to-End Protocol Testing
AC-PI7.1: A2A Agent Card validates against official A2A JSON schema

Validates that the Agent Card conforms to the A2A Protocol v0.3.0 schema.

Reference: https://a2a-protocol.org/latest/specification/
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# -----------------------------------------------------------------------------
# AC-PI7.1: A2A Agent Card Schema Validation
# -----------------------------------------------------------------------------


class TestAgentCardSchemaValidation:
    """Validate Agent Card against A2A schema requirements."""
    
    def test_agent_card_has_required_fields(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card must have all required A2A fields."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required top-level fields per A2A spec
        assert "protocolVersion" in data
        assert "name" in data
        assert "description" in data
        assert "version" in data
        assert "capabilities" in data
        assert "skills" in data
    
    def test_agent_card_protocol_version(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card must declare protocol version 0.3.0."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        assert data["protocolVersion"] == "0.3.0"
    
    def test_agent_card_capabilities_structure(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card capabilities must have correct structure."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        capabilities = data["capabilities"]
        
        # Capabilities should be a dict with boolean flags
        assert isinstance(capabilities, dict)
        
        # Check for expected capability keys
        if "streaming" in capabilities:
            assert isinstance(capabilities["streaming"], bool)
        if "pushNotifications" in capabilities:
            assert isinstance(capabilities["pushNotifications"], bool)
    
    def test_agent_card_skills_structure(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card skills must have correct structure."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        skills = data["skills"]
        
        # Skills should be a list
        assert isinstance(skills, list)
        assert len(skills) > 0
        
        # Each skill should have required fields
        for skill in skills:
            assert "id" in skill
            assert "name" in skill
            assert "description" in skill
            assert isinstance(skill["id"], str)
            assert isinstance(skill["name"], str)
            assert isinstance(skill["description"], str)
    
    def test_agent_card_skills_have_modes(
        self, test_client_enabled: TestClient
    ) -> None:
        """Each skill should declare input and output modes."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        for skill in data["skills"]:
            # inputModes and outputModes are optional per spec but recommended
            if "inputModes" in skill:
                assert isinstance(skill["inputModes"], list)
                assert len(skill["inputModes"]) > 0
                for mode in skill["inputModes"]:
                    assert isinstance(mode, str)
            
            if "outputModes" in skill:
                assert isinstance(skill["outputModes"], list)
                assert len(skill["outputModes"]) > 0
    
    def test_agent_card_skill_count_matches_functions(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card should expose all 8 agent functions as skills."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        skills = data["skills"]
        
        # We have 8 agent functions
        assert len(skills) >= 8
        
        # Check for expected skill IDs (actual implementation names)
        skill_ids = {skill["id"] for skill in skills}
        expected_skills = {
            "extract_structure",
            "summarize_content",
            "generate_code",
            "analyze_artifact",
            "validate_against_spec",
            "decompose_task",
            "synthesize_outputs",
            "cross_reference",
        }
        
        # All expected skills should be present
        assert expected_skills.issubset(skill_ids)


# -----------------------------------------------------------------------------
# Agent Card Content Validation
# -----------------------------------------------------------------------------


class TestAgentCardContent:
    """Validate Agent Card content correctness."""
    
    def test_agent_card_name_is_descriptive(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card name should be descriptive."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        name = data["name"]
        assert len(name) > 0
        assert len(name) <= 100  # Reasonable length limit
    
    def test_agent_card_description_is_meaningful(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card description should be meaningful."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        description = data["description"]
        assert len(description) > 10  # More than trivial
        assert len(description) <= 500  # Not excessive
    
    def test_agent_card_version_is_semver(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card version should be semver-like."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        version = data["version"]
        
        # Should contain at least one dot (e.g., "1.0" or "1.0.0")
        assert "." in version
        
        # First part should be numeric
        major = version.split(".")[0]
        assert major.isdigit()
    
    def test_skill_descriptions_are_helpful(
        self, test_client_enabled: TestClient
    ) -> None:
        """Each skill should have a helpful description."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        data = response.json()
        
        for skill in data["skills"]:
            description = skill["description"]
            
            # Description should be non-empty
            assert len(description) > 0
            
            # Description should be meaningful (not just placeholder)
            assert description.lower() != "todo"
            assert description.lower() != "description"


# -----------------------------------------------------------------------------
# Agent Card Discoverability
# -----------------------------------------------------------------------------


class TestAgentCardDiscoverability:
    """Test Agent Card discoverability per A2A spec."""
    
    def test_well_known_path_correct(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card must be at /.well-known/agent-card.json."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        
        assert response.status_code == 200
    
    def test_content_type_is_json(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card response must be application/json."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type
    
    def test_agent_card_is_valid_json(
        self, test_client_enabled: TestClient
    ) -> None:
        """Agent Card must be parseable as valid JSON."""
        response = test_client_enabled.get("/.well-known/agent-card.json")
        
        # This will raise if not valid JSON
        data = response.json()
        
        assert data is not None
        assert isinstance(data, dict)

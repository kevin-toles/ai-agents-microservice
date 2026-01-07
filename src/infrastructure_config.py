"""
Infrastructure-aware configuration for ai-agents.

This module provides dynamic URL resolution based on INFRASTRUCTURE_MODE.
NO HARDCODED VALUES - URLs are determined at runtime based on mode.

Modes:
  - docker: All services in Docker (uses Docker DNS names)
  - hybrid: All infra in Docker, Python services native (uses localhost)
  - native: All native (uses localhost)

Usage:
    from src.infrastructure_config import get_platform_config, get_service_url
    
    config = get_platform_config()
    semantic_url = config.services["semantic-search"]
    
    # Or direct access:
    gateway_url = get_service_url("llm-gateway")

Reference: ARCHITECTURE_DECISION_RECORD.md - Explicit mode declaration
Reference: ARCHITECTURE_ROUNDTABLE_FINDINGS.md - Mode parity contracts
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Docker DNS Names (used when running inside Docker containers)
# =============================================================================

DOCKER_HOSTNAMES = {
    # Databases
    "neo4j": "ai-platform-neo4j",
    "qdrant": "ai-platform-qdrant",
    "redis": "ai-platform-redis",
    # Services
    "llm-gateway": "llm-gateway",
    "semantic-search": "semantic-search-service",
    "code-orchestrator": "code-orchestrator-service",
    "inference-service": "inference-service",
    "ai-agents": "ai-agents",
    "audit-service": "audit-service",
    "task-orchestrator": "task-orchestrator",  # Future: D6 from UNIFIED_KITCHEN_BRIGADE_ARCHITECTURE
}

# =============================================================================
# Default Ports
# =============================================================================

DEFAULT_PORTS = {
    # Databases
    "neo4j_bolt": 7687,
    "neo4j_http": 7474,
    "qdrant": 6333,
    "redis": 6379,
    # Services
    "llm-gateway": 8080,
    "semantic-search": 8081,
    "ai-agents": 8082,
    "code-orchestrator": 8083,
    "audit-service": 8084,
    "inference-service": 8085,
    "task-orchestrator": 8086,  # Future: D6
}

# =============================================================================
# Default Credentials
# =============================================================================

DEFAULT_CREDENTIALS = {
    "neo4j_user": "neo4j",
    "neo4j_password": "devpassword",  # Matches existing volume data
}

# =============================================================================
# Data Directories (for cross-reference Stage 2)
# =============================================================================

# These paths are used for direct file access in local development
# In production, would be accessed via ai-platform-data API
DEFAULT_DATA_PATHS = {
    "ai_platform_data": Path("/Users/kevintoles/POC/ai-platform-data"),
    "textbooks": Path("/Users/kevintoles/POC/textbooks"),
}


# =============================================================================
# Platform Configuration Dataclass
# =============================================================================

@dataclass
class PlatformConfig:
    """Complete platform configuration based on infrastructure mode.
    
    Provides:
    - Service URLs (llm-gateway, semantic-search, etc.)
    - Database URLs (Neo4j, Qdrant, Redis)
    - Data paths (for cross-reference Stage 2)
    - Mode information
    
    Reference: ARCHITECTURE_DECISION_RECORD.md - Generated config artifact
    """
    
    mode: str
    services: dict[str, str] = field(default_factory=dict)
    databases: dict[str, str] = field(default_factory=dict)
    data_paths: dict[str, Path] = field(default_factory=dict)
    credentials: dict[str, str] = field(default_factory=dict)
    
    def get_service_url(self, service_name: str) -> str | None:
        """Get URL for a service by name."""
        return self.services.get(service_name)
    
    def get_database_url(self, db_name: str) -> str | None:
        """Get URL for a database by name."""
        return self.databases.get(db_name)
    
    def get_data_path(self, path_name: str) -> Path | None:
        """Get data path by name."""
        return self.data_paths.get(path_name)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "mode": self.mode,
            "services": self.services,
            "databases": self.databases,
            "data_paths": {k: str(v) for k, v in self.data_paths.items()},
            "credentials": dict.fromkeys(self.credentials, "***"),  # Mask credentials
        }
    
    @classmethod
    def from_mode(cls, mode: str) -> "PlatformConfig":
        """Create config from infrastructure mode.
        
        Args:
            mode: One of 'docker', 'hybrid', 'native'
            
        Returns:
            Configured PlatformConfig instance
        """
        if mode == "docker":
            return cls._create_docker_config()
        elif mode in ("hybrid", "native"):
            return cls._create_localhost_config(mode)
        else:
            raise ValueError(f"Unknown infrastructure mode: {mode}")
    
    @classmethod
    def _create_docker_config(cls) -> "PlatformConfig":
        """Create config for full Docker mode (all services in containers)."""
        return cls(
            mode="docker",
            services={
                "llm-gateway": f"http://{DOCKER_HOSTNAMES['llm-gateway']}:{DEFAULT_PORTS['llm-gateway']}",
                "semantic-search": f"http://{DOCKER_HOSTNAMES['semantic-search']}:{DEFAULT_PORTS['semantic-search']}",
                "code-orchestrator": f"http://{DOCKER_HOSTNAMES['code-orchestrator']}:{DEFAULT_PORTS['code-orchestrator']}",
                "inference-service": f"http://{DOCKER_HOSTNAMES['inference-service']}:{DEFAULT_PORTS['inference-service']}",
                "ai-agents": f"http://{DOCKER_HOSTNAMES['ai-agents']}:{DEFAULT_PORTS['ai-agents']}",
                "audit-service": f"http://{DOCKER_HOSTNAMES['audit-service']}:{DEFAULT_PORTS['audit-service']}",
                "task-orchestrator": f"http://{DOCKER_HOSTNAMES['task-orchestrator']}:{DEFAULT_PORTS['task-orchestrator']}",
            },
            databases={
                "qdrant": f"http://{DOCKER_HOSTNAMES['qdrant']}:{DEFAULT_PORTS['qdrant']}",
                "neo4j_bolt": f"bolt://{DOCKER_HOSTNAMES['neo4j']}:{DEFAULT_PORTS['neo4j_bolt']}",
                "neo4j_http": f"http://{DOCKER_HOSTNAMES['neo4j']}:{DEFAULT_PORTS['neo4j_http']}",
                "redis": f"redis://{DOCKER_HOSTNAMES['redis']}:{DEFAULT_PORTS['redis']}",
            },
            data_paths=DEFAULT_DATA_PATHS.copy(),
            credentials=DEFAULT_CREDENTIALS.copy(),
        )
    
    @classmethod
    def _create_localhost_config(cls, mode: str) -> "PlatformConfig":
        """Create config for hybrid/native mode (services connect via localhost)."""
        return cls(
            mode=mode,
            services={
                "llm-gateway": f"http://localhost:{DEFAULT_PORTS['llm-gateway']}",
                "semantic-search": f"http://localhost:{DEFAULT_PORTS['semantic-search']}",
                "code-orchestrator": f"http://localhost:{DEFAULT_PORTS['code-orchestrator']}",
                "inference-service": f"http://localhost:{DEFAULT_PORTS['inference-service']}",
                "ai-agents": f"http://localhost:{DEFAULT_PORTS['ai-agents']}",
                "audit-service": f"http://localhost:{DEFAULT_PORTS['audit-service']}",
                "task-orchestrator": f"http://localhost:{DEFAULT_PORTS['task-orchestrator']}",
            },
            databases={
                "qdrant": f"http://localhost:{DEFAULT_PORTS['qdrant']}",
                "neo4j_bolt": f"bolt://localhost:{DEFAULT_PORTS['neo4j_bolt']}",
                "neo4j_http": f"http://localhost:{DEFAULT_PORTS['neo4j_http']}",
                "redis": f"redis://localhost:{DEFAULT_PORTS['redis']}",
            },
            data_paths=DEFAULT_DATA_PATHS.copy(),
            credentials=DEFAULT_CREDENTIALS.copy(),
        )


# =============================================================================
# Module-level Functions
# =============================================================================

def _detect_running_in_docker() -> bool:
    """Detect if we're running inside a Docker container.
    
    Checks for /.dockerenv file or cgroup hints.
    """
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except FileNotFoundError:
        return False


def get_infrastructure_mode() -> str:
    """Get the current infrastructure mode.
    
    Priority:
    1. INFRASTRUCTURE_MODE environment variable (set by Platform Control)
    2. PLATFORM_MODE environment variable (alias)
    3. Auto-detect if running in Docker
    4. Default to 'hybrid' (most common development mode)
    
    Reference: ARCHITECTURE_DECISION_RECORD.md - Explicit mode declaration
    
    Returns:
        One of: 'docker', 'hybrid', 'native'
    """
    # Check explicit env vars first
    mode = os.environ.get("INFRASTRUCTURE_MODE", "").lower()
    if mode in ("docker", "hybrid", "native"):
        logger.info("Infrastructure mode from INFRASTRUCTURE_MODE: %s", mode)
        return mode
    
    mode = os.environ.get("PLATFORM_MODE", "").lower()
    if mode in ("docker", "hybrid", "native"):
        logger.info("Infrastructure mode from PLATFORM_MODE: %s", mode)
        return mode
    
    # Auto-detect Docker container
    if _detect_running_in_docker():
        logger.info("Auto-detected: running in Docker container")
        return "docker"
    
    # Default to hybrid for local development
    logger.info("Using default infrastructure mode: hybrid")
    return "hybrid"


def get_platform_config(mode: str | None = None) -> PlatformConfig:
    """Get the complete platform configuration.
    
    Args:
        mode: Optional mode override. If None, auto-detects.
        
    Returns:
        PlatformConfig instance with all URLs and paths
    """
    current_mode = mode or get_infrastructure_mode()
    config = PlatformConfig.from_mode(current_mode)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    logger.info("Platform config loaded for mode '%s'", current_mode)
    return config


def _apply_env_overrides(config: PlatformConfig) -> PlatformConfig:
    """Apply environment variable overrides to config.
    
    Allows individual URL overrides via environment variables.
    Example: LLM_GATEWAY_URL=http://custom:8080
    """
    # Service URL overrides
    service_env_map = {
        "LLM_GATEWAY_URL": "llm-gateway",
        "SEMANTIC_SEARCH_URL": "semantic-search",
        "CODE_ORCHESTRATOR_URL": "code-orchestrator",
        "INFERENCE_SERVICE_URL": "inference-service",
        "AI_AGENTS_URL": "ai-agents",
        "AUDIT_SERVICE_URL": "audit-service",
    }
    
    for env_var, service_name in service_env_map.items():
        override = os.environ.get(env_var)
        if override:
            config.services[service_name] = override
            logger.info("Service %s overridden via %s: %s", service_name, env_var, override)
    
    # Database URL overrides
    db_env_map = {
        "QDRANT_URL": "qdrant",
        "NEO4J_URI": "neo4j_bolt",
        "REDIS_URL": "redis",
    }
    
    for env_var, db_name in db_env_map.items():
        override = os.environ.get(env_var)
        if override:
            config.databases[db_name] = override
            logger.info("Database %s overridden via %s: %s", db_name, env_var, override)
    
    # Data path overrides
    ai_platform_data_override = os.environ.get("AI_PLATFORM_DATA_PATH")
    if ai_platform_data_override:
        config.data_paths["ai_platform_data"] = Path(ai_platform_data_override)
        logger.info("Data path ai_platform_data overridden: %s", ai_platform_data_override)
    
    textbooks_override = os.environ.get("TEXTBOOKS_PATH")
    if textbooks_override:
        config.data_paths["textbooks"] = Path(textbooks_override)
        logger.info("Data path textbooks overridden: %s", textbooks_override)
    
    return config


def get_service_url(service_name: str, mode: str | None = None) -> str:
    """Convenience function to get a specific service URL.
    
    Args:
        service_name: Name of service (e.g., 'llm-gateway', 'semantic-search')
        mode: Optional mode override
        
    Returns:
        Service URL string
        
    Raises:
        ValueError: If service name is unknown
    """
    config = get_platform_config(mode)
    url = config.services.get(service_name)
    if not url:
        raise ValueError(f"Unknown service: {service_name}. Valid: {list(config.services.keys())}")
    return url


def get_database_url(db_name: str, mode: str | None = None) -> str:
    """Convenience function to get a specific database URL.
    
    Args:
        db_name: Name of database (e.g., 'qdrant', 'neo4j_bolt')
        mode: Optional mode override
        
    Returns:
        Database URL string
        
    Raises:
        ValueError: If database name is unknown
    """
    config = get_platform_config(mode)
    url = config.databases.get(db_name)
    if not url:
        raise ValueError(f"Unknown database: {db_name}. Valid: {list(config.databases.keys())}")
    return url


def get_data_paths(mode: str | None = None) -> dict[str, Path]:
    """Get all data paths for cross-reference Stage 2.
    
    Returns paths for:
    - ai_platform_data: Books, metadata, enriched content
    - textbooks: Technical reference books (JSON)
    
    These paths are used by Kitchen Brigade cross-reference.
    """
    config = get_platform_config(mode)
    return config.data_paths


# =============================================================================
# Kitchen Brigade Specific Helpers
# =============================================================================

def get_cross_reference_config() -> dict[str, Any]:
    """Get configuration specifically for Kitchen Brigade cross-reference (Stage 2).
    
    Returns a dict with:
    - Service endpoints for retrieval
    - Data paths for direct file access
    - Database connections
    
    Reference: UNIFIED_KITCHEN_BRIGADE_ARCHITECTURE.md - Stage 2 retrieval
    """
    config = get_platform_config()
    
    ai_platform_data = config.data_paths.get("ai_platform_data", Path("."))
    textbooks_dir = config.data_paths.get("textbooks", Path("."))
    
    return {
        # Service endpoints
        "semantic_search_url": config.services.get("semantic-search"),
        "code_orchestrator_url": config.services.get("code-orchestrator"),
        "llm_gateway_url": config.services.get("llm-gateway"),
        "inference_service_url": config.services.get("inference-service"),
        
        # Database endpoints (for direct access if needed)
        "qdrant_url": config.databases.get("qdrant"),
        "neo4j_bolt_url": config.databases.get("neo4j_bolt"),
        
        # Data paths for direct file access
        "textbooks_dir": textbooks_dir / "JSON Texts",
        "books_raw_dir": ai_platform_data / "books" / "raw",
        "books_enriched_dir": ai_platform_data / "books" / "enriched",
        "books_metadata_dir": ai_platform_data / "books" / "metadata",
        
        # Mode info
        "mode": config.mode,
    }


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    mode = get_infrastructure_mode()
    print(f"Auto-detected mode: {mode}")
    print()
    
    for test_mode in ["docker", "hybrid", "native"]:
        config = get_platform_config(test_mode)
        print(f"=== Configuration for '{test_mode}' mode ===")
        print("Services:")
        for name, url in config.services.items():
            print(f"  {name}: {url}")
        print("Databases:")
        for name, url in config.databases.items():
            print(f"  {name}: {url}")
        print("Data Paths:")
        for name, path in config.data_paths.items():
            print(f"  {name}: {path}")
        print()
    
    print("=== Cross-Reference Config ===")
    xref_config = get_cross_reference_config()
    for k, v in xref_config.items():
        print(f"  {k}: {v}")

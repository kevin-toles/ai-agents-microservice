"""Configuration module for ai-agents service.

This module contains:
- models.py: LLM model configuration loader
- feature_flags.py: Phase 2 Protocol Integration feature flags

Reference: WBS-PI1, PROTOCOL_INTEGRATION_ARCHITECTURE.md
"""

from src.config.feature_flags import ProtocolFeatureFlags, get_feature_flags
from src.config.models import (
    get_model_config,
    get_provider_config,
    is_model_enabled,
    load_config,
    reload_config,
)

__all__ = [
    "ProtocolFeatureFlags",
    "get_feature_flags",
    "get_model_config",
    "get_provider_config",
    "is_model_enabled",
    "load_config",
    "reload_config",
]

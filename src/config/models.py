"""
LLM Model Configuration Loader

Loads and provides access to LLM model configurations from config/llm_models.yaml.
This enables runtime configuration of which models are available and their parameters.

Usage:
    from src.config.models import get_model_config, get_enabled_models, get_poc_participants
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "llm_models.yaml"

# Cached configuration
_config: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    """Load the LLM model configuration from YAML file.
    
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    global _config
    
    if _config is not None:
        return _config
    
    if not CONFIG_PATH.exists():
        logger.warning("LLM config file not found at %s, using defaults", CONFIG_PATH)
        _config = {"providers": {}, "poc": {}, "error_handling": {}}
        return _config
    
    with open(CONFIG_PATH) as f:
        _config = yaml.safe_load(f)
        
    logger.info("Loaded LLM model configuration from %s", CONFIG_PATH)
    return _config


def reload_config() -> dict[str, Any]:
    """Force reload the configuration from disk.
    
    Returns:
        Fresh configuration dictionary.
    """
    global _config
    _config = None
    return load_config()


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific provider.
    
    Args:
        provider: Provider name (openai, anthropic, openrouter).
        
    Returns:
        Provider configuration dict, or empty dict if not found.
    """
    config = load_config()
    return config.get("providers", {}).get(provider, {})


def get_model_config(provider: str, model: str) -> dict[str, Any]:
    """Get configuration for a specific model.
    
    Args:
        provider: Provider name.
        model: Model identifier.
        
    Returns:
        Model configuration dict, or empty dict if not found.
    """
    provider_config = get_provider_config(provider)
    return provider_config.get("models", {}).get(model, {})


def is_model_enabled(provider: str, model: str) -> bool:
    """Check if a model is enabled.
    
    Args:
        provider: Provider name.
        model: Model identifier.
        
    Returns:
        True if enabled, False otherwise.
    """
    model_config = get_model_config(provider, model)
    return model_config.get("enabled", False)


def get_enabled_models(provider: str | None = None) -> list[dict[str, Any]]:
    """Get all enabled models, optionally filtered by provider.
    
    Args:
        provider: Optional provider to filter by.
        
    Returns:
        List of enabled model configs with provider and model_id added.
    """
    config = load_config()
    enabled = []
    
    providers = config.get("providers", {})
    if provider:
        providers = {provider: providers.get(provider, {})}
    
    for prov_name, prov_config in providers.items():
        if not prov_config.get("enabled", False):
            continue
            
        for model_id, model_config in prov_config.get("models", {}).items():
            if model_config.get("enabled", False):
                enabled.append({
                    "provider": prov_name,
                    "model_id": model_id,
                    **model_config,
                })
    
    return enabled


def get_poc_participants() -> list[dict[str, Any]]:
    """Get the POC participant configurations.
    
    Returns:
        List of participant configurations for the Inter-AI POC.
    """
    config = load_config()
    return config.get("poc", {}).get("participants", [])


def get_fallback_models(provider: str) -> list[dict[str, str]]:
    """Get fallback models for a provider.
    
    Args:
        provider: Provider that failed.
        
    Returns:
        List of fallback provider/model pairs.
    """
    config = load_config()
    fallbacks = config.get("poc", {}).get("fallbacks", {})
    return fallbacks.get(provider, [])


def get_error_handling_config() -> dict[str, Any]:
    """Get error handling configuration.
    
    Returns:
        Error handling configuration dict.
    """
    config = load_config()
    return config.get("error_handling", {})


def get_max_tokens(provider: str, model: str) -> int:
    """Get max tokens for a model.
    
    Args:
        provider: Provider name.
        model: Model identifier.
        
    Returns:
        Max tokens value, defaulting to 4096.
    """
    model_config = get_model_config(provider, model)
    # GPT-5.x uses max_completion_tokens
    return model_config.get("max_completion_tokens") or model_config.get("max_tokens", 4096)

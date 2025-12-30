"""Token estimation and budget utilities.

Provides centralized token estimation to eliminate CHARS_PER_TOKEN duplication
across function modules (S1192 compliance).

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Token Budget Allocation
"""

from src.core.constants import CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    Uses the industry standard approximation of ~4 characters per token.
    This is a heuristic - actual tokenization varies by model.
    
    Args:
        text: Text to estimate tokens for.
        
    Returns:
        Estimated token count (integer division).
        
    Example:
        >>> estimate_tokens("Hello world!")  # 12 chars
        3
        >>> estimate_tokens("")
        0
    """
    return len(text) // CHARS_PER_TOKEN


def check_budget(text: str, budget_tokens: int) -> tuple[bool, int]:
    """Check if text fits within token budget.
    
    Args:
        text: Text to check.
        budget_tokens: Maximum allowed tokens.
        
    Returns:
        Tuple of (within_budget, estimated_tokens).
        
    Example:
        >>> within, tokens = check_budget("short text", 1000)
        >>> within
        True
    """
    estimated = estimate_tokens(text)
    return estimated <= budget_tokens, estimated


def tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count.
    
    Args:
        tokens: Number of tokens.
        
    Returns:
        Approximate character count.
        
    Example:
        >>> tokens_to_chars(100)
        400
    """
    return tokens * CHARS_PER_TOKEN


__all__ = [
    "CHARS_PER_TOKEN",
    "estimate_tokens",
    "check_budget",
    "tokens_to_chars",
]

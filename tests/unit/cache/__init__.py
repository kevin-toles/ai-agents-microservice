"""Test package for cache management.

Contains unit tests for:
- State prefix conventions (temp:, user:, app:)
- build_cache_key() function
- HandoffCache (pipeline-local)
- CompressionCache (Redis)
- Artifact dataclass

Pattern: TDD (RED → GREEN → REFACTOR)
Reference: WBS-AGT3
"""

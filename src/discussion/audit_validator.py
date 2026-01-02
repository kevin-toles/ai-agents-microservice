"""Audit Service Validator for Citation Validation.

WBS Reference: WBS-KB5 - Provenance & Audit Integration
Tasks: KB5.4, KB5.5, KB5.7

Acceptance Criteria:
- AC-KB5.2: Citations validated via audit-service:8084/v1/validate
- AC-KB5.3: Invalid citations rejected, trigger additional evidence gathering
- AC-KB5.5: Audit trail includes discussion_history with all cycles

Exit Criteria:
- audit-service receives validation request with citations
- Invalid citation (fake source) triggers loop retry

Anti-Patterns Avoided:
- S1192: Constants at module level
- S3776: Cognitive complexity via helper functions
- Frozen dataclasses for immutability
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from src.discussion.models import DiscussionCycle


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

_CONST_DEFAULT_SERVICE_URL = "http://audit-service:8084"
_CONST_DEFAULT_TIMEOUT = 30
_CONST_DEFAULT_RETRY_COUNT = 3
_CONST_VALIDATE_ENDPOINT = "/v1/audit/cross-reference"

logger = logging.getLogger(__name__)


# =============================================================================
# ValidationResult
# =============================================================================


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of citation validation.
    
    Attributes:
        is_valid: Whether all citations passed validation
        valid_citations: List of valid citation markers
        invalid_citations: List of invalid citation markers
        errors: List of error messages for invalid citations
    """

    is_valid: bool
    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "valid_citations": list(self.valid_citations),
            "invalid_citations": list(self.invalid_citations),
            "errors": list(self.errors),
        }


# =============================================================================
# ValidationConfig
# =============================================================================


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for audit service validation.
    
    Attributes:
        service_url: Base URL for audit-service
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
    """

    service_url: str = _CONST_DEFAULT_SERVICE_URL
    timeout: int = _CONST_DEFAULT_TIMEOUT
    retry_count: int = _CONST_DEFAULT_RETRY_COUNT


# =============================================================================
# AuditServiceValidator (AC-KB5.2, AC-KB5.3, AC-KB5.5)
# =============================================================================


class AuditServiceValidator:
    """Validates citations via audit-service.
    
    AC-KB5.2: Citations validated via audit-service:8084/v1/validate
    AC-KB5.3: Invalid citations rejected, trigger additional evidence gathering
    AC-KB5.5: Audit trail includes discussion_history with all cycles
    
    Example:
        >>> validator = AuditServiceValidator()
        >>> result = await validator.validate(
        ...     content="The cache uses Redis. [^1]",
        ...     citations=[{"marker": 1, "source": "cache.py"}],
        ... )
        >>> print(result.is_valid)
        True
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize the audit service validator.
        
        Args:
            config: Optional configuration for validation behavior.
        """
        self._config = config or ValidationConfig()

    @property
    def config(self) -> ValidationConfig:
        """Get the validator configuration."""
        return self._config

    async def validate(
        self,
        content: str,
        citations: list[dict[str, Any]],
    ) -> ValidationResult:
        """Validate citations against audit-service.
        
        AC-KB5.2: Citations validated via audit-service:8084/v1/validate
        
        Args:
            content: The content containing citation markers
            citations: List of citation dictionaries with marker and source
            
        Returns:
            ValidationResult indicating which citations are valid/invalid.
        """
        # Handle empty citations - all valid by default
        if not citations:
            return ValidationResult(
                is_valid=True,
                valid_citations=[],
                invalid_citations=[],
                errors=[],
            )

        try:
            return await self._call_audit_service(content, citations)
        except asyncio.TimeoutError:
            logger.error("Audit service request timed out")
            return ValidationResult(
                is_valid=False,
                valid_citations=[],
                invalid_citations=[c.get("marker", 0) for c in citations],
                errors=["Timeout: audit-service request timed out"],
            )
        except Exception as e:
            logger.error("Audit service error: %s", str(e))
            return ValidationResult(
                is_valid=False,
                valid_citations=[],
                invalid_citations=[c.get("marker", 0) for c in citations],
                errors=[f"Connection error: {str(e)}"],
            )

    async def _call_audit_service(
        self,
        content: str,
        citations: list[dict[str, Any]],
    ) -> ValidationResult:
        """Make the actual HTTP call to audit-service.
        
        Args:
            content: Content to validate
            citations: Citations to check
            
        Returns:
            ValidationResult from audit-service response.
        """
        url = f"{self._config.service_url}{_CONST_VALIDATE_ENDPOINT}"

        # Build request payload
        payload = {
            "code": content,
            "references": [
                {
                    "chapter_id": str(c.get("marker", 0)),
                    "title": c.get("source", ""),
                    "content": c.get("content", content),
                }
                for c in citations
            ],
            "threshold": 0.5,
        }

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            response = await client.post(url, json=payload)

            if response.status_code != 200:
                return ValidationResult(
                    is_valid=False,
                    valid_citations=[],
                    invalid_citations=[c.get("marker", 0) for c in citations],
                    errors=[f"HTTP {response.status_code}: Service error"],
                )

            try:
                data = response.json()
            except ValueError:
                return ValidationResult(
                    is_valid=False,
                    valid_citations=[],
                    invalid_citations=[c.get("marker", 0) for c in citations],
                    errors=["Invalid JSON response from audit-service"],
                )

            return self._parse_response(data, citations)

    def _parse_response(
        self,
        data: dict[str, Any],
        citations: list[dict[str, Any]],
    ) -> ValidationResult:
        """Parse audit-service response into ValidationResult.
        
        Args:
            data: JSON response from audit-service
            citations: Original citation list for marker extraction
            
        Returns:
            Parsed ValidationResult.
        """
        passed = data.get("passed", False)
        findings = data.get("findings", [])

        valid_citations = []
        invalid_citations = []
        errors = []

        for finding in findings:
            marker = finding.get("marker", 0)
            is_valid = finding.get("valid", False)
            reason = finding.get("reason", "")

            if is_valid:
                valid_citations.append(marker)
            else:
                invalid_citations.append(marker)
                if reason:
                    errors.append(f"Citation {marker}: {reason}")

        # If no findings, use passed status
        if not findings:
            if passed:
                valid_citations = [c.get("marker", 0) for c in citations]
            else:
                invalid_citations = [c.get("marker", 0) for c in citations]

        return ValidationResult(
            is_valid=passed and len(invalid_citations) == 0,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            errors=errors,
        )

    async def validate_citations(
        self,
        content: str,
        citations: list[dict[str, Any]],
    ) -> ValidationResult:
        """Alias for validate() for API clarity.
        
        Args:
            content: Content containing citations
            citations: List of citation dictionaries
            
        Returns:
            ValidationResult.
        """
        return await self.validate(content, citations)

    async def validate_with_history(
        self,
        content: str,
        citations: list[dict[str, Any]],
        discussion_history: list["DiscussionCycle"],
    ) -> ValidationResult:
        """Validate with discussion history included in audit payload.
        
        AC-KB5.5: Audit trail includes discussion_history with all cycles.
        
        Args:
            content: Content to validate
            citations: Citations to check
            discussion_history: Complete discussion history
            
        Returns:
            ValidationResult from audit-service.
        """
        # Build payload with history
        payload = self.build_audit_payload(content, citations, discussion_history)

        try:
            url = f"{self._config.service_url}{_CONST_VALIDATE_ENDPOINT}"
            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                response = await client.post(url, json=payload)

                if response.status_code != 200:
                    return ValidationResult(
                        is_valid=False,
                        valid_citations=[],
                        invalid_citations=[c.get("marker", 0) for c in citations],
                        errors=[f"HTTP {response.status_code}"],
                    )

                data = response.json()
                return self._parse_response(data, citations)

        except Exception as e:
            logger.error("Validation with history failed: %s", str(e))
            return ValidationResult(
                is_valid=False,
                valid_citations=[],
                invalid_citations=[c.get("marker", 0) for c in citations],
                errors=[str(e)],
            )

    def build_audit_payload(
        self,
        content: str,
        citations: list[dict[str, Any]],
        discussion_history: list["DiscussionCycle"],
    ) -> dict[str, Any]:
        """Build the audit request payload with discussion history.
        
        AC-KB5.5: Audit trail includes discussion_history with all cycles.
        
        Args:
            content: Content being validated
            citations: Citations to validate
            discussion_history: Complete discussion history
            
        Returns:
            Payload dictionary for audit-service.
        """
        return {
            "code": content,
            "references": [
                {
                    "chapter_id": str(c.get("marker", 0)),
                    "title": c.get("source", ""),
                    "content": c.get("content", content),
                }
                for c in citations
            ],
            "threshold": 0.5,
            "discussion_history": [
                cycle.to_dict() for cycle in discussion_history
            ],
        }

    def should_retry(self, result: ValidationResult) -> bool:
        """Determine if validation should trigger a retry.
        
        AC-KB5.3: Invalid citations rejected, trigger additional evidence gathering.
        
        Args:
            result: The validation result to check.
            
        Returns:
            True if invalid citations exist and retry is warranted.
        """
        return not result.is_valid and len(result.invalid_citations) > 0


__all__ = [
    "AuditServiceValidator",
    "ValidationConfig",
    "ValidationResult",
]

"""Audit Service HTTP Client.

WBS: MSE-8.2 - Audit Service Client
WBS: MSE-8.3 - Fake Audit Client
WBS: AGT-17.3 - Citation Audit Client

Async client for calling Audit-Service API endpoints.
Implements connection pooling and retry logic per anti-pattern #12.

Reference: MULTI_STAGE_ENRICHMENT_PIPELINE_WBS.md - MSE-8
Reference: AGENT_FUNCTIONS_ARCHITECTURE.md - Citation Flow
Pattern: Protocol duck typing (CODING_PATTERNS_ANALYSIS.md)
Anti-Pattern Mitigation: #12 (Connection Pooling via shared client)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from src.schemas.audit import CitationAuditBatch, CitationAuditRecord

from src.agents.msep.constants import (
    ENDPOINT_AUDIT_CROSS_REF,
    SERVICE_AUDIT_SERVICE,
    SERVICE_AUDIT_URL,
)
from src.agents.msep.exceptions import AuditServiceUnavailableError


logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5
DEFAULT_TIMEOUT = 30.0


class AuditServiceClient:
    """HTTP client for Audit-Service.

    WBS: MSE-8.2 - Audit Service Client

    Provides async methods for cross-reference auditing via CodeBERT.
    Uses connection pooling (single httpx.AsyncClient) and implements
    retry logic for transient errors.

    Attributes:
        base_url: Base URL for Audit-Service
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts on transient errors

    Example:
        >>> client = AuditServiceClient(base_url="http://localhost:8084")
        >>> result = await client.audit_cross_references(code, refs, 0.5)
        >>> await client.close()
    """

    def __init__(
        self,
        base_url: str = SERVICE_AUDIT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the Audit-Service client.

        Args:
            base_url: Base URL for Audit-Service
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialization).

        Returns:
            Shared httpx.AsyncClient instance (connection pooling)
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            await asyncio.sleep(0)  # Yield to event loop on first init
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _is_retryable_error(self, e: httpx.HTTPStatusError) -> bool:
        """Check if HTTP error is retryable (5xx server errors).

        Args:
            e: HTTP status error

        Returns:
            True if error is retryable (5xx), False otherwise
        """
        return e.response.status_code >= 500

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff between retries.

        Args:
            attempt: Current attempt number (0-indexed)
        """
        delay = RETRY_BACKOFF_FACTOR * (2**attempt)
        await asyncio.sleep(delay)

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP request with retry logic for transient errors.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            json: Request body as JSON

        Returns:
            Parsed JSON response

        Raises:
            AuditServiceUnavailableError: After exhausting retries
        """
        client = await self._get_client()
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "POST":
                    response = await client.post(endpoint, json=json)
                else:
                    response = await client.get(endpoint)

                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result

            except httpx.HTTPStatusError as e:
                if not self._is_retryable_error(e):
                    raise
                last_exception = e

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e

            if attempt < self.max_retries and last_exception:
                await self._backoff(attempt)
                logger.warning(
                    "Retrying %s %s (attempt %d/%d): %s",
                    method,
                    endpoint,
                    attempt + 1,
                    self.max_retries,
                    str(last_exception),
                )

        raise AuditServiceUnavailableError(
            message=f"{SERVICE_AUDIT_SERVICE} unavailable after {self.max_retries} retries",
            cause=last_exception,
            url=f"{self.base_url}{endpoint}",
        )

    async def audit_cross_references(
        self,
        code: str,
        references: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Audit code against reference chapters using CodeBERT similarity.

        WBS: MSE-8.2.5 - Calls POST /v1/audit/cross-reference

        Args:
            code: Source code/content to audit
            references: List of reference chapter dicts with chapter_id, content
            threshold: Similarity threshold for passing audit

        Returns:
            Dict with passed, status, findings, best_similarity

        Raises:
            AuditServiceUnavailableError: When service is unavailable
        """
        payload = {
            "code": code,
            "references": references,
            "threshold": threshold,
        }

        return await self._request_with_retry(
            method="POST",
            endpoint=ENDPOINT_AUDIT_CROSS_REF,
            json=payload,
        )

    async def _post(
        self,
        endpoint: str,
        json: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute POST request with retry logic.

        This is exposed for testing purposes.

        Args:
            endpoint: API endpoint path
            json: Request body as JSON

        Returns:
            Parsed JSON response
        """
        return await self._request_with_retry(
            method="POST",
            endpoint=endpoint,
            json=json,
        )

    async def send_citation_audit(
        self,
        record: CitationAuditRecord,
    ) -> dict[str, Any]:
        """Send a citation audit record to the audit service.

        WBS: AGT-17.3 - Audit record sent to audit-service:8084

        Args:
            record: Citation audit record to send

        Returns:
            Dict with status and created record ID

        Raises:
            ConnectionError: After exhausting retries
        """

        endpoint = "/v1/audit/citations"
        payload = record.model_dump(mode="json")

        try:
            return await self._post(endpoint, payload)
        except Exception as e:
            # Convert to ConnectionError for consistent interface
            if "unavailable" in str(e).lower():
                raise ConnectionError(str(e)) from e
            raise

    async def send_citation_audit_batch(
        self,
        batch: CitationAuditBatch,
    ) -> dict[str, Any]:
        """Send a batch of citation audit records to the audit service.

        WBS: AGT-17.3 - Batch citation audit

        Args:
            batch: Batch of citation audit records to send

        Returns:
            Dict with status and count of created records

        Raises:
            ConnectionError: After exhausting retries
        """

        endpoint = "/v1/audit/citations/batch"
        payload = batch.model_dump(mode="json")

        try:
            return await self._post(endpoint, payload)
        except Exception as e:
            if "unavailable" in str(e).lower():
                raise ConnectionError(str(e)) from e
            raise


class FakeAuditServiceClient:
    """Fake Audit-Service client for unit testing.

    WBS: MSE-8.3 - Fake Audit Client
    WBS: AGT-17.3 - Fake Citation Audit Client

    Returns configurable responses without making HTTP calls.
    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md

    Attributes:
        should_pass: Whether audit should pass
        best_similarity: Similarity score to return
        should_raise_error: Whether to raise an error
        should_fail: Whether to simulate connection failures
        citation_audit_records: Recorded citation audit records
    """

    def __init__(
        self,
        should_pass: bool = True,
        best_similarity: float = 0.8,
        should_raise_error: bool = False,
        should_fail: bool = False,
    ) -> None:
        """Initialize fake client.

        Args:
            should_pass: Whether audit should pass
            best_similarity: Similarity score to return
            should_raise_error: Whether to raise AuditServiceUnavailableError
            should_fail: Whether to raise ConnectionError for citation audits
        """
        self.should_pass = should_pass
        self.best_similarity = best_similarity
        self.should_raise_error = should_raise_error
        self.should_fail = should_fail
        self._client: httpx.AsyncClient | None = None  # Always None for fake
        self.citation_audit_records: list[CitationAuditRecord] = []

    async def audit_cross_references(
        self,
        code: str,
        references: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Return fake audit result.

        The code parameter is intentionally unused - FakeAuditServiceClient
        returns deterministic responses based on its configuration, not the input.

        Args:
            code: Source code/content to audit (unused in fake implementation)
            references: List of reference chapter dicts
            threshold: Similarity threshold for response

        Returns:
            Configured fake response

        Raises:
            AuditServiceUnavailableError: If should_raise_error is True
        """
        # code parameter intentionally unused - fake returns configured response
        del code  # silence ARG002 linter warning
        if self.should_raise_error:
            raise AuditServiceUnavailableError(
                message="Fake audit service unavailable",
                cause=None,
                url="http://fake-audit:8084",
            )

        # Build findings from references
        findings = []
        for ref in references:
            findings.append({
                "chapter_id": ref.get("chapter_id", "unknown"),
                "similarity": self.best_similarity,
                "matched_chapter": ref.get("chapter_id", "unknown"),
            })

        return {
            "passed": self.should_pass,
            "status": "verified" if self.should_pass else "suspicious",
            "findings": findings,
            "best_similarity": self.best_similarity,
            "threshold": threshold,
            "theory_impl_count": 0,
        }

    async def close(self) -> None:
        """No-op close for fake client."""
        pass

    async def send_citation_audit(
        self,
        record: CitationAuditRecord,
    ) -> dict[str, Any]:
        """Record citation audit for testing.

        WBS: AGT-17.3 - Fake citation audit

        Args:
            record: Citation audit record to record

        Returns:
            Fake success response

        Raises:
            ConnectionError: If should_fail is True
        """
        if self.should_fail:
            raise ConnectionError("Fake connection failure")

        self.citation_audit_records.append(record)
        return {"status": "created", "id": f"fake-audit-{len(self.citation_audit_records)}"}

    async def send_citation_audit_batch(
        self,
        batch: CitationAuditBatch,
    ) -> dict[str, Any]:
        """Record batch citation audit for testing.

        Args:
            batch: Batch of citation audit records

        Returns:
            Fake success response

        Raises:
            ConnectionError: If should_fail is True
        """
        if self.should_fail:
            raise ConnectionError("Fake connection failure")

        self.citation_audit_records.extend(batch.records)
        return {"status": "created", "count": len(batch.records)}

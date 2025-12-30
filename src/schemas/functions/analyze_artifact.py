"""Schemas for analyze_artifact function.

WBS-AGT9: analyze_artifact Function schemas.

Purpose: Analyze code/document for patterns, issues, quality.

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 4

From Architecture Doc:
```yaml
analyze_artifact:
  description: "Analyze artifact for quality, security, patterns"
  
  input:
    artifact: str             # Code or document
    artifact_type: enum       # code | document | config
    analysis_type: enum       # quality | security | patterns | dependencies
    checklist: list[str]      # Optional specific checks
  
  output:
    findings: list[Finding]   # Issues with severity, location, fix hint
    metrics: dict             # CC, LOC, etc. for code
    pass: bool                # Overall gate
    compressed_report: str    # For downstream
  
  context_budget:
    input: 16384 tokens
    output: 2048 tokens
```
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ArtifactKind(str, Enum):
    """Type of artifact being analyzed.
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - code: Source code files
    - document: Documentation/markdown
    - config: Configuration files (YAML, JSON, etc.)
    """
    CODE = "code"
    DOCUMENT = "document"
    CONFIG = "config"


class AnalysisType(str, Enum):
    """Type of analysis to perform.
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md (AC-9.5):
    - quality: Code quality checks (complexity, style, docstrings)
    - security: Security vulnerability detection
    - patterns: Design pattern identification
    - dependencies: Dependency analysis
    """
    QUALITY = "quality"
    SECURITY = "security"
    PATTERNS = "patterns"
    DEPENDENCIES = "dependencies"


class Severity(str, Enum):
    """Severity level for findings.
    
    Severity levels from low to critical:
    - INFO: Informational, not an issue
    - LOW: Minor issue, no immediate action needed
    - MEDIUM: Should be addressed
    - HIGH: Important issue, should be fixed
    - CRITICAL: Must be fixed immediately
    """
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalyzeArtifactInput(BaseModel):
    """Input schema for analyze_artifact function.
    
    Reference: AC-9.1, AC-9.5
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - artifact: Code or document content
    - artifact_type: code | document | config
    - analysis_type: quality | security | patterns | dependencies
    - checklist: Optional specific checks to perform
    """
    artifact: str = Field(
        ...,
        description="Code or document content to analyze",
    )
    artifact_type: ArtifactKind = Field(
        default=ArtifactKind.CODE,
        description="Type of artifact: code | document | config",
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.QUALITY,
        description="Type of analysis: quality | security | patterns | dependencies",
    )
    checklist: list[str] = Field(
        default_factory=list,
        description="Optional list of specific checks to perform",
    )


class Finding(BaseModel):
    """Single finding from artifact analysis.
    
    Exit Criteria: Each Finding has severity, category, description, location.
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - findings: list[Finding] # Issues with severity, location, fix hint
    """
    severity: Severity = Field(
        ...,
        description="Severity level: critical | high | medium | low | info",
    )
    category: str = Field(
        ...,
        description="Category of finding (e.g., security, code-quality, design-pattern)",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the finding",
    )
    location: str = Field(
        ...,
        description="Location in artifact (e.g., 'line 42', 'func:calculate', 'class:User')",
    )
    fix_hint: Optional[str] = Field(
        default=None,
        description="Optional hint for how to fix the issue",
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Optional precise line number for the finding",
    )


class AnalysisResult(BaseModel):
    """Output schema for analyze_artifact function.
    
    Reference: AC-9.2
    
    From AGENT_FUNCTIONS_ARCHITECTURE.md:
    - findings: list[Finding] # Issues with severity, location, fix hint
    - metrics: dict # CC, LOC, etc. for code
    - pass: bool # Overall gate
    - compressed_report: str # For downstream
    """
    findings: list[Finding] = Field(
        default_factory=list,
        description="List of findings from the analysis",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Code metrics (LOC, cyclomatic complexity, etc.)",
    )
    passed: bool = Field(
        default=True,
        description="Overall analysis gate (True if no critical/high issues)",
    )
    compressed_report: Optional[str] = Field(
        default=None,
        description="Compressed summary for downstream functions",
    )

"""analyze_artifact agent function.

WBS-AGT9: analyze_artifact Function implementation.

Purpose: Analyze code/document for patterns, issues, quality.
- Analyzes code/docs for quality, patterns, issues (AC-9.1)
- Returns AnalysisResult with findings list (AC-9.2)
- Context budget: 16384 input / 2048 output (AC-9.3)
- Default preset: D4 (Standard) (AC-9.4)
- Supports analysis_type parameter (quality/security/patterns) (AC-9.5)

Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 4

REFACTOR Phase:
- Extracted CHARS_PER_TOKEN to shared utilities (S1192)
- Using estimate_tokens() from src/functions/utils/token_utils.py
"""

import re
from typing import Any

from src.functions.base import AgentFunction, ContextBudgetExceededError
from src.functions.utils.token_utils import estimate_tokens
from src.schemas.functions.analyze_artifact import (
    AnalysisResult,
    AnalysisType,
    ArtifactKind,
    Finding,
    Severity,
)


# Context budget for analyze_artifact (AC-9.3)
INPUT_BUDGET_TOKENS = 16384
OUTPUT_BUDGET_TOKENS = 2048


class AnalyzeArtifactFunction(AgentFunction):
    """Analyze code/document for patterns, issues, quality.
    
    Reference: AGENT_FUNCTIONS_ARCHITECTURE.md → Agent Function 4
    
    Acceptance Criteria:
    - AC-9.1: Analyzes code/docs for quality, patterns, issues
    - AC-9.2: Returns AnalysisResult with findings list
    - AC-9.3: Context budget: 16384 input / 2048 output
    - AC-9.4: Default preset: D4 (Standard)
    - AC-9.5: Supports analysis_type parameter (quality/security/patterns)
    
    Exit Criteria:
    - Each Finding has severity, category, description, location
    - analysis_type="security" flags common vulnerabilities
    - analysis_type="patterns" identifies design patterns
    """
    
    name: str = "analyze_artifact"
    
    # AC-9.4: Default preset D4 (Standard)
    default_preset: str = "D4"
    
    # Preset options from architecture doc
    available_presets: dict[str, str] = {
        "code": "D4",      # Think + Code critique
        "security": "D3",  # Debate for high-stakes
        "quick": "S3",     # Qwen solo for speed
    }
    
    # =========================================================================
    # Security Patterns (for analysis_type="security")
    # =========================================================================
    
    SECURITY_PATTERNS: dict[str, dict[str, Any]] = {
        "hardcoded_password": {
            "patterns": [
                r'password\s*=\s*["\'][^"\']+["\']',  # Python: password = "secret"
                r'passwd\s*=\s*["\'][^"\']+["\']',
                r'pwd\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'password\s*:\s*\S+',  # YAML/Config: password: secret123
                r'passwd\s*:\s*\S+',
                r'secret\s*:\s*\S+',
            ],
            "severity": Severity.HIGH,
            "category": "security-credentials",
            "description": "Hardcoded credential detected - use environment variables or secrets manager",
            "fix_hint": "Move credentials to environment variables or a secrets manager",
        },
        "api_key_exposure": {
            "patterns": [
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'apikey\s*=\s*["\'][^"\']+["\']',
                r'api[-_]?secret\s*=\s*["\'][^"\']+["\']',
                r'["\']sk_live_[a-zA-Z0-9]+["\']',
                r'["\']sk_test_[a-zA-Z0-9]+["\']',
            ],
            "severity": Severity.HIGH,
            "category": "security-credentials",
            "description": "API key exposure detected - use secure key management",
            "fix_hint": "Store API keys in environment variables or a secrets vault",
        },
        "sql_injection": {
            "patterns": [
                r'f["\']SELECT\s+.*{',
                r'f["\']INSERT\s+.*{',
                r'f["\']UPDATE\s+.*{',
                r'f["\']DELETE\s+.*{',
                r'["\']SELECT\s+.*\+\s*\w+',
                r'\.format\(.*SELECT',
                r'%\s*.*SELECT',
            ],
            "severity": Severity.CRITICAL,
            "category": "security-injection",
            "description": "Potential SQL injection vulnerability - use parameterized queries",
            "fix_hint": "Use parameterized queries or an ORM to prevent SQL injection",
        },
        "eval_usage": {
            "patterns": [
                r'\beval\s*\(',
                r'\bexec\s*\(',
            ],
            "severity": Severity.HIGH,
            "category": "security-code-execution",
            "description": "Dangerous eval/exec usage - can execute arbitrary code",
            "fix_hint": "Avoid eval/exec with user input; use safer alternatives like ast.literal_eval",
        },
        "shell_injection": {
            "patterns": [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\([^,]+\+',
                r'subprocess\.run\s*\([^,]+\+',
                r'shell\s*=\s*True',
            ],
            "severity": Severity.HIGH,
            "category": "security-injection",
            "description": "Potential shell command injection - sanitize input",
            "fix_hint": "Use subprocess with shell=False and pass arguments as a list",
        },
        "weak_crypto": {
            "patterns": [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\.',
                r'RC4\.',
            ],
            "severity": Severity.MEDIUM,
            "category": "security-crypto",
            "description": "Weak cryptographic algorithm detected",
            "fix_hint": "Use SHA-256 or stronger algorithms for hashing; use AES for encryption",
        },
    }
    
    # =========================================================================
    # Design Pattern Signatures (for analysis_type="patterns")
    # =========================================================================
    
    DESIGN_PATTERNS: dict[str, dict[str, Any]] = {
        "singleton": {
            "patterns": [
                r'_instance\s*=\s*None',
                r'def\s+__new__\s*\([^)]*\).*_instance',
                r'@classmethod.*def\s+get_instance',
            ],
            "required_matches": 1,
            "category": "design-pattern",
            "description": "Singleton pattern detected - ensures single instance",
        },
        "factory": {
            "patterns": [
                r'class\s+\w*Factory\w*',
                r'def\s+create_\w+\s*\(',
                r'if\s+.*type.*==.*:.*return\s+\w+\(',
            ],
            "required_matches": 1,
            "category": "design-pattern",
            "description": "Factory pattern detected - encapsulates object creation",
        },
        "repository": {
            "patterns": [
                r'class\s+\w*Repository\w*',
                r'def\s+get\s*\(\s*self.*id',
                r'def\s+save\s*\(\s*self',
                r'def\s+delete\s*\(\s*self',
            ],
            "required_matches": 2,
            "category": "design-pattern",
            "description": "Repository pattern detected - abstracts data access",
        },
        "strategy": {
            "patterns": [
                r'class\s+\w*Strategy\w*',
                r'def\s+execute\s*\(\s*self',
                r'self\._strategy',
                r'set_strategy\s*\(',
            ],
            "required_matches": 2,
            "category": "design-pattern",
            "description": "Strategy pattern detected - encapsulates algorithms",
        },
        "observer": {
            "patterns": [
                r'def\s+subscribe\s*\(',
                r'def\s+notify\s*\(',
                r'self\._observers',
                r'def\s+attach\s*\(',
            ],
            "required_matches": 2,
            "category": "design-pattern",
            "description": "Observer pattern detected - implements pub/sub",
        },
        "decorator_pattern": {
            "patterns": [
                r'class\s+\w*Decorator\w*',
                r'def\s+__init__\s*\(\s*self\s*,\s*\w+\s*:.*\w+',
                r'self\._wrapped',
                r'self\._component',
            ],
            "required_matches": 2,
            "category": "design-pattern",
            "description": "Decorator pattern detected - adds behavior dynamically",
        },
    }
    
    # =========================================================================
    # Quality Check Thresholds
    # =========================================================================
    
    QUALITY_THRESHOLDS: dict[str, int] = {
        "max_function_lines": 20,
        "max_complexity": 10,
        "max_class_methods": 10,
        "max_parameters": 5,
    }
    
    async def run(self, **kwargs: Any) -> AnalysisResult:
        """Analyze artifact for quality, security, or patterns.
        
        Args:
            artifact: Code or document content to analyze
            artifact_type: Type of artifact (code/document/config)
            analysis_type: Type of analysis (quality/security/patterns/dependencies)
            checklist: Optional list of specific checks to perform
            
        Returns:
            AnalysisResult with findings, metrics, pass status, and compressed report
            
        Raises:
            ContextBudgetExceededError: If input exceeds budget
        """
        artifact: str = kwargs.get("artifact", "")
        artifact_type = kwargs.get("artifact_type", ArtifactKind.CODE)
        analysis_type = kwargs.get("analysis_type", AnalysisType.QUALITY)
        checklist: list[str] = kwargs.get("checklist", [])
        
        # Convert enum if passed as string
        if isinstance(artifact_type, str):
            artifact_type = ArtifactKind(artifact_type)
        if isinstance(analysis_type, str):
            analysis_type = AnalysisType(analysis_type)
        
        # AC-9.3: Enforce input budget - using shared utility
        input_tokens = estimate_tokens(artifact)
        if input_tokens > INPUT_BUDGET_TOKENS:
            raise ContextBudgetExceededError(
                function_name=self.name,
                actual=input_tokens,
                limit=INPUT_BUDGET_TOKENS,
            )
        
        # Handle empty artifact
        if not artifact.strip():
            return AnalysisResult(
                findings=[],
                metrics={},
                passed=True,
                compressed_report="Empty artifact - no analysis performed",
            )
        
        # Dispatch to appropriate analyzer
        if analysis_type == AnalysisType.QUALITY:
            return self._analyze_quality(artifact, artifact_type, checklist)
        elif analysis_type == AnalysisType.SECURITY:
            return self._analyze_security(artifact, artifact_type)
        elif analysis_type == AnalysisType.PATTERNS:
            return self._analyze_patterns(artifact, artifact_type)
        elif analysis_type == AnalysisType.DEPENDENCIES:
            return self._analyze_dependencies(artifact, artifact_type)
        else:
            # Default to quality analysis
            return self._analyze_quality(artifact, artifact_type, checklist)
    
    # =========================================================================
    # Quality Analysis
    # =========================================================================
    
    def _analyze_quality(
        self,
        artifact: str,
        artifact_type: ArtifactKind,
        checklist: list[str],
    ) -> AnalysisResult:
        """Analyze artifact for code quality issues.
        
        Checks for:
        - Long functions
        - Missing docstrings
        - High complexity
        - Too many parameters
        """
        findings: list[Finding] = []
        metrics: dict[str, Any] = {}
        
        # Calculate basic metrics
        lines = artifact.split("\n")
        metrics["loc"] = len(lines)
        metrics["blank_lines"] = sum(1 for line in lines if not line.strip())
        
        if artifact_type == ArtifactKind.CODE:
            # Find functions
            functions = self._extract_functions(artifact)
            metrics["functions"] = len(functions)
            
            # Find classes
            classes = self._extract_classes(artifact)
            metrics["classes"] = len(classes)
            
            # Check for long functions
            findings.extend(self._check_long_functions(artifact, functions))
            
            # Check for missing docstrings
            if not checklist or "docstrings" in checklist:
                findings.extend(self._check_missing_docstrings(artifact, functions, classes))
            
            # Check for too many parameters
            findings.extend(self._check_parameter_count(artifact, functions))
        
        # Calculate passed status (no HIGH or CRITICAL issues)
        passed = not any(
            f.severity in [Severity.HIGH, Severity.CRITICAL]
            for f in findings
        )
        
        # Generate compressed report
        compressed_report = self._generate_quality_report(findings, metrics)
        
        return AnalysisResult(
            findings=findings,
            metrics=metrics,
            passed=passed,
            compressed_report=compressed_report,
        )
    
    def _extract_functions(self, code: str) -> list[dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        pattern = r'^\s*(async\s+)?def\s+(\w+)\s*\(([^)]*)\):'
        
        for i, line in enumerate(code.split("\n"), 1):
            match = re.match(pattern, line)
            if match:
                functions.append({
                    "name": match.group(2),
                    "line": i,
                    "params": match.group(3),
                    "is_async": bool(match.group(1)),
                })
        
        return functions
    
    def _extract_classes(self, code: str) -> list[dict[str, Any]]:
        """Extract class definitions from code."""
        classes = []
        pattern = r'^\s*class\s+(\w+)\s*[:\(]'
        
        for i, line in enumerate(code.split("\n"), 1):
            match = re.match(pattern, line)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i,
                })
        
        return classes
    
    def _check_long_functions(
        self,
        code: str,
        functions: list[dict[str, Any]],
    ) -> list[Finding]:
        """Check for functions that are too long."""
        findings = []
        lines = code.split("\n")
        
        for i, func in enumerate(functions):
            start_line = func["line"]
            
            # Find function end (next function or class at same/lower indent, or EOF)
            end_line = len(lines)
            if i + 1 < len(functions):
                end_line = min(end_line, functions[i + 1]["line"] - 1)
            
            # Count function lines
            func_lines = 0
            for j in range(start_line - 1, min(end_line, len(lines))):
                line = lines[j]
                if line.strip() and not line.strip().startswith("#"):
                    func_lines += 1
            
            if func_lines > self.QUALITY_THRESHOLDS["max_function_lines"]:
                findings.append(Finding(
                    severity=Severity.MEDIUM,
                    category="code-quality",
                    description=f"Function '{func['name']}' is {func_lines} lines long (max {self.QUALITY_THRESHOLDS['max_function_lines']})",
                    location=f"func:{func['name']}",
                    line_number=func["line"],
                    fix_hint="Consider breaking into smaller functions",
                ))
        
        return findings
    
    def _check_missing_docstrings(
        self,
        code: str,
        functions: list[dict[str, Any]],
        classes: list[dict[str, Any]],
    ) -> list[Finding]:
        """Check for missing docstrings on functions and classes."""
        findings = []
        lines = code.split("\n")
        
        # Check functions
        for func in functions:
            if func["name"].startswith("_") and func["name"] != "__init__":
                continue  # Skip private functions except __init__
            
            line_idx = func["line"]  # 1-indexed
            if line_idx < len(lines):
                next_line = lines[line_idx].strip() if line_idx < len(lines) else ""
                has_docstring = next_line.startswith('"""') or next_line.startswith("'''")
                
                if not has_docstring:
                    findings.append(Finding(
                        severity=Severity.LOW,
                        category="documentation",
                        description=f"Function '{func['name']}' is missing a docstring",
                        location=f"func:{func['name']}",
                        line_number=func["line"],
                        fix_hint="Add a docstring describing the function's purpose",
                    ))
        
        # Check classes
        for cls in classes:
            line_idx = cls["line"]
            if line_idx < len(lines):
                next_line = lines[line_idx].strip() if line_idx < len(lines) else ""
                has_docstring = next_line.startswith('"""') or next_line.startswith("'''")
                
                if not has_docstring:
                    findings.append(Finding(
                        severity=Severity.LOW,
                        category="documentation",
                        description=f"Class '{cls['name']}' is missing a docstring",
                        location=f"class:{cls['name']}",
                        line_number=cls["line"],
                        fix_hint="Add a docstring describing the class's purpose",
                    ))
        
        return findings
    
    def _check_parameter_count(
        self,
        code: str,
        functions: list[dict[str, Any]],
    ) -> list[Finding]:
        """Check for functions with too many parameters."""
        findings = []
        
        for func in functions:
            params_str = func.get("params", "")
            if not params_str:
                continue
            
            # Count parameters (excluding self, cls, *args, **kwargs)
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            params = [p for p in params if p not in ("self", "cls") 
                      and not p.startswith("*")]
            
            if len(params) > self.QUALITY_THRESHOLDS["max_parameters"]:
                findings.append(Finding(
                    severity=Severity.LOW,
                    category="code-quality",
                    description=f"Function '{func['name']}' has {len(params)} parameters (max {self.QUALITY_THRESHOLDS['max_parameters']})",
                    location=f"func:{func['name']}",
                    line_number=func["line"],
                    fix_hint="Consider using a configuration object or data class",
                ))
        
        return findings
    
    def _generate_quality_report(
        self,
        findings: list[Finding],
        metrics: dict[str, Any],
    ) -> str:
        """Generate compressed quality report for downstream."""
        critical_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == Severity.HIGH)
        medium_count = sum(1 for f in findings if f.severity == Severity.MEDIUM)
        low_count = sum(1 for f in findings if f.severity == Severity.LOW)
        
        parts = [f"Quality analysis: {len(findings)} issues found"]
        
        if metrics:
            parts.append(f"LOC: {metrics.get('loc', 'N/A')}")
            if "functions" in metrics:
                parts.append(f"Functions: {metrics['functions']}")
            if "classes" in metrics:
                parts.append(f"Classes: {metrics['classes']}")
        
        if findings:
            severity_summary = []
            if critical_count:
                severity_summary.append(f"{critical_count} critical")
            if high_count:
                severity_summary.append(f"{high_count} high")
            if medium_count:
                severity_summary.append(f"{medium_count} medium")
            if low_count:
                severity_summary.append(f"{low_count} low")
            parts.append(f"Severity: {', '.join(severity_summary)}")
        
        return " | ".join(parts)
    
    # =========================================================================
    # Security Analysis
    # =========================================================================
    
    def _analyze_security(
        self,
        artifact: str,
        artifact_type: ArtifactKind,
    ) -> AnalysisResult:
        """Analyze artifact for security vulnerabilities.
        
        Exit Criteria: analysis_type="security" flags common vulnerabilities
        """
        findings: list[Finding] = []
        metrics: dict[str, Any] = {"checks_performed": 0}
        
        # Run all security pattern checks
        for check_name, check_config in self.SECURITY_PATTERNS.items():
            metrics["checks_performed"] += 1
            
            for pattern in check_config["patterns"]:
                matches = list(re.finditer(pattern, artifact, re.IGNORECASE | re.MULTILINE))
                
                for match in matches:
                    # Calculate line number
                    line_number = artifact[:match.start()].count("\n") + 1
                    
                    findings.append(Finding(
                        severity=check_config["severity"],
                        category=check_config["category"],
                        description=check_config["description"],
                        location=f"line {line_number}",
                        line_number=line_number,
                        fix_hint=check_config.get("fix_hint"),
                    ))
        
        # Remove duplicate findings on same line
        seen: set[tuple[str, int]] = set()
        unique_findings: list[Finding] = []
        for f in findings:
            key = (f.category, f.line_number or 0)
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)
        
        findings = unique_findings
        
        # Calculate passed status
        passed = not any(
            f.severity in [Severity.HIGH, Severity.CRITICAL]
            for f in findings
        )
        
        # Generate compressed report
        compressed_report = self._generate_security_report(findings, metrics)
        
        return AnalysisResult(
            findings=findings,
            metrics=metrics,
            passed=passed,
            compressed_report=compressed_report,
        )
    
    def _generate_security_report(
        self,
        findings: list[Finding],
        metrics: dict[str, Any],
    ) -> str:
        """Generate compressed security report."""
        if not findings:
            return f"Security analysis: No vulnerabilities found ({metrics.get('checks_performed', 0)} checks performed)"
        
        critical_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == Severity.HIGH)
        
        parts = [f"Security analysis: {len(findings)} vulnerabilities found"]
        
        if critical_count:
            parts.append(f"CRITICAL: {critical_count}")
        if high_count:
            parts.append(f"HIGH: {high_count}")
        
        categories = set(f.category for f in findings)
        parts.append(f"Categories: {', '.join(categories)}")
        
        return " | ".join(parts)
    
    # =========================================================================
    # Pattern Detection
    # =========================================================================
    
    def _analyze_patterns(
        self,
        artifact: str,
        artifact_type: ArtifactKind,
    ) -> AnalysisResult:
        """Analyze artifact for design patterns.
        
        Exit Criteria: analysis_type="patterns" identifies design patterns
        
        Note: Pattern findings have INFO severity (informational, not issues).
        """
        findings: list[Finding] = []
        metrics: dict[str, Any] = {"patterns_detected": []}
        
        for pattern_name, pattern_config in self.DESIGN_PATTERNS.items():
            match_count = 0
            
            for pattern in pattern_config["patterns"]:
                if re.search(pattern, artifact, re.IGNORECASE | re.MULTILINE):
                    match_count += 1
            
            required_matches = pattern_config.get("required_matches", 1)
            
            if match_count >= required_matches:
                metrics["patterns_detected"].append(pattern_name)
                
                findings.append(Finding(
                    severity=Severity.INFO,  # Patterns are informational
                    category=pattern_config["category"],
                    description=f"{pattern_name.replace('_', ' ').title()} pattern: {pattern_config['description']}",
                    location="code structure",
                ))
        
        # Pattern analysis always passes (it's informational)
        passed = True
        
        # Generate compressed report
        compressed_report = self._generate_patterns_report(findings, metrics)
        
        return AnalysisResult(
            findings=findings,
            metrics=metrics,
            passed=passed,
            compressed_report=compressed_report,
        )
    
    def _generate_patterns_report(
        self,
        findings: list[Finding],
        metrics: dict[str, Any],
    ) -> str:
        """Generate compressed patterns report."""
        patterns = metrics.get("patterns_detected", [])
        
        if not patterns:
            return "Pattern analysis: No design patterns detected"
        
        pattern_names = ", ".join(p.replace("_", " ").title() for p in patterns)
        return f"Pattern analysis: Detected {len(patterns)} patterns ({pattern_names})"
    
    # =========================================================================
    # Dependency Analysis
    # =========================================================================
    
    def _analyze_dependencies(
        self,
        artifact: str,
        artifact_type: ArtifactKind,
    ) -> AnalysisResult:
        """Analyze artifact for dependencies."""
        findings: list[Finding] = []
        metrics: dict[str, Any] = {
            "imports": [],
            "stdlib_imports": 0,
            "third_party_imports": 0,
        }
        
        # Extract import statements
        import_patterns = [
            r'^import\s+(\w+)',
            r'^from\s+(\w+)',
        ]
        
        stdlib_modules = {
            "os", "sys", "re", "json", "typing", "abc", "asyncio", "collections",
            "datetime", "functools", "itertools", "math", "pathlib", "subprocess",
            "tempfile", "threading", "time", "unittest", "uuid", "logging",
            "dataclasses", "enum", "contextlib", "copy", "hashlib", "base64",
        }
        
        lines = artifact.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    metrics["imports"].append(module)
                    
                    if module in stdlib_modules:
                        metrics["stdlib_imports"] += 1
                    else:
                        metrics["third_party_imports"] += 1
                        
                        # Flag certain risky imports
                        if module in ("pickle", "marshal"):
                            findings.append(Finding(
                                severity=Severity.MEDIUM,
                                category="dependency-risk",
                                description=f"Import of '{module}' can be risky with untrusted data",
                                location=f"line {i}",
                                line_number=i,
                                fix_hint="Ensure data is from trusted sources",
                            ))
        
        # Generate compressed report
        compressed_report = (
            f"Dependency analysis: {len(metrics['imports'])} imports "
            f"({metrics['stdlib_imports']} stdlib, {metrics['third_party_imports']} third-party)"
        )
        
        return AnalysisResult(
            findings=findings,
            metrics=metrics,
            passed=True,
            compressed_report=compressed_report,
        )

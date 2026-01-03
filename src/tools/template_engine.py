"""Template Engine Tool for Code Generation.

WBS Reference: WBS-KB10 - Template Engine Tool
Gap: template_engine Tool (not implemented)

Provides Jinja2-based template rendering for:
- Code scaffolding
- Documentation generation
- Configuration file generation
- Response formatting

Anti-Patterns Avoided:
- #12: Cached environment instances
- S1192: String constants at module level
- S3776: Cognitive complexity < 15 via dispatch pattern
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    TemplateSyntaxError,
    select_autoescape,
)
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Module Constants (S1192 Compliance)
# =============================================================================

_DEFAULT_TEMPLATES_DIR = "/templates"
_DEFAULT_AUTOESCAPE = True


# =============================================================================
# Data Models
# =============================================================================


class TemplateRenderResult(BaseModel):
    """Result of template rendering."""
    
    template_name: str = Field(..., description="Name of rendered template")
    rendered_content: str = Field(..., description="Rendered content")
    success: bool = Field(True, description="Whether rendering succeeded")
    error_message: str | None = Field(None, description="Error message if failed")


class TemplateInfo(BaseModel):
    """Information about a template."""
    
    name: str = Field(..., description="Template name")
    path: str = Field(..., description="Template file path")
    description: str | None = Field(None, description="Template description from metadata")
    required_variables: list[str] = Field(
        default_factory=list,
        description="Variables required by template",
    )


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class TemplateEngineProtocol(Protocol):
    """Protocol for template engine tools."""
    
    def render(self, template_name: str, context: dict[str, Any]) -> TemplateRenderResult:
        """Render a template with context."""
        ...
    
    def render_string(self, template_str: str, context: dict[str, Any]) -> TemplateRenderResult:
        """Render a template string with context."""
        ...
    
    def list_templates(self) -> list[TemplateInfo]:
        """List available templates."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TemplateEngineConfig:
    """Configuration for template engine.
    
    Attributes:
        templates_dir: Directory containing templates
        autoescape: Whether to enable autoescaping
        strict_undefined: Whether to error on undefined variables
    """
    
    templates_dir: str = _DEFAULT_TEMPLATES_DIR
    autoescape: bool = _DEFAULT_AUTOESCAPE
    strict_undefined: bool = True


# =============================================================================
# Built-in Templates
# =============================================================================

# These templates are available even without filesystem templates
BUILTIN_TEMPLATES = {
    "function_stub": '''"""{{ docstring }}"""

def {{ function_name }}({{ parameters | join(", ") }}):
    {% if return_type %}# Returns: {{ return_type }}{% endif %}
    {% for line in body %}
    {{ line }}
    {% endfor %}
''',
    
    "class_stub": '''"""{{ docstring }}"""

class {{ class_name }}{% if bases %}({{ bases | join(", ") }}){% endif %}:
    {% for method in methods %}
    def {{ method.name }}(self{% if method.params %}, {{ method.params | join(", ") }}{% endif %}):
        """{{ method.docstring }}"""
        {% if method.body %}{{ method.body }}{% else %}pass{% endif %}
    {% endfor %}
''',
    
    "docstring": '''"""{{ summary }}

{% if args %}
Args:
{% for arg in args %}
    {{ arg.name }}: {{ arg.description }}
{% endfor %}
{% endif %}
{% if returns %}
Returns:
    {{ returns }}
{% endif %}
{% if raises %}
Raises:
{% for exc in raises %}
    {{ exc.type }}: {{ exc.description }}
{% endfor %}
{% endif %}
"""
''',
    
    "footnote": '''[^{{ footnote_id }}]: {{ source_type }}: {{ source_name }}
{% if page is defined and page %}  - Page: {{ page }}{% endif %}
{% if chapter is defined and chapter %}  - Chapter: {{ chapter }}{% endif %}
{% if url is defined and url %}  - URL: {{ url }}{% endif %}
{% if confidence is defined and confidence %}  - Confidence: {{ confidence }}{% endif %}
''',
    
    "api_response": '''{
    "success": {{ success | lower }},
    "data": {{ data | tojson }},
    {% if error %}"error": "{{ error }}",{% endif %}
    "metadata": {
        "timestamp": "{{ timestamp }}",
        "version": "{{ version }}"
    }
}
''',
    
    "test_case": '''def test_{{ test_name }}():
    """{{ docstring }}"""
    # Arrange
    {% for setup in arrange %}
    {{ setup }}
    {% endfor %}
    
    # Act
    {% for action in act %}
    {{ action }}
    {% endfor %}
    
    # Assert
    {% for assertion in assertions %}
    {{ assertion }}
    {% endfor %}
''',
    
    "markdown_section": '''## {{ title }}

{{ content }}

{% if subsections %}
{% for sub in subsections %}
### {{ sub.title }}

{{ sub.content }}

{% endfor %}
{% endif %}
{% if code_example %}
```{{ language | default("python") }}
{{ code_example }}
```
{% endif %}
''',
}


# =============================================================================
# TemplateEngineTool Implementation
# =============================================================================


class TemplateEngineTool:
    """Tool for rendering Jinja2 templates.
    
    Provides:
    - File-based templates from templates directory
    - Built-in templates for common patterns
    - String template rendering
    - Template listing and introspection
    
    Example:
        >>> engine = TemplateEngineTool()
        >>> result = engine.render("function_stub", {
        ...     "function_name": "process_data",
        ...     "docstring": "Process input data",
        ...     "parameters": ["data", "options=None"],
        ...     "body": ["return data"]
        ... })
        >>> print(result.rendered_content)
    """
    
    def __init__(self, config: TemplateEngineConfig | None = None) -> None:
        """Initialize the template engine.
        
        Args:
            config: Optional configuration.
        """
        self._config = config or TemplateEngineConfig()
        self._env = self._create_environment()
        self._builtin_env = self._create_builtin_environment()
    
    def _create_environment(self) -> Environment | None:
        """Create Jinja2 environment from filesystem templates."""
        templates_path = Path(self._config.templates_dir)
        
        if not templates_path.exists():
            logger.info(f"Templates directory not found: {templates_path}")
            return None
        
        loader = FileSystemLoader(str(templates_path))
        return Environment(
            loader=loader,
            autoescape=select_autoescape() if self._config.autoescape else False,
            undefined=StrictUndefined if self._config.strict_undefined else None,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    
    def _create_builtin_environment(self) -> Environment:
        """Create Jinja2 environment for built-in templates."""
        env = Environment(
            autoescape=False,
            undefined=StrictUndefined if self._config.strict_undefined else None,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        return env
    
    def render(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> TemplateRenderResult:
        """Render a template with context.
        
        Checks built-in templates first, then filesystem templates.
        
        Args:
            template_name: Name of template to render
            context: Variables to pass to template
            
        Returns:
            TemplateRenderResult with rendered content.
        """
        # Try built-in templates first
        if template_name in BUILTIN_TEMPLATES:
            return self._render_builtin(template_name, context)
        
        # Try filesystem templates
        if self._env is not None:
            return self._render_filesystem(template_name, context)
        
        return TemplateRenderResult(
            template_name=template_name,
            rendered_content="",
            success=False,
            error_message=f"Template not found: {template_name}",
        )
    
    def _render_builtin(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> TemplateRenderResult:
        """Render a built-in template."""
        try:
            template_str = BUILTIN_TEMPLATES[template_name]
            template = self._builtin_env.from_string(template_str)
            rendered = template.render(**context)
            return TemplateRenderResult(
                template_name=template_name,
                rendered_content=rendered.strip(),
                success=True,
            )
        except Exception as e:
            logger.error(f"Error rendering builtin template {template_name}: {e}")
            return TemplateRenderResult(
                template_name=template_name,
                rendered_content="",
                success=False,
                error_message=str(e),
            )
    
    def _render_filesystem(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> TemplateRenderResult:
        """Render a filesystem template."""
        try:
            template = self._env.get_template(template_name)
            rendered = template.render(**context)
            return TemplateRenderResult(
                template_name=template_name,
                rendered_content=rendered.strip(),
                success=True,
            )
        except TemplateNotFound:
            return TemplateRenderResult(
                template_name=template_name,
                rendered_content="",
                success=False,
                error_message=f"Template not found: {template_name}",
            )
        except TemplateSyntaxError as e:
            return TemplateRenderResult(
                template_name=template_name,
                rendered_content="",
                success=False,
                error_message=f"Template syntax error: {e}",
            )
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return TemplateRenderResult(
                template_name=template_name,
                rendered_content="",
                success=False,
                error_message=str(e),
            )
    
    def render_string(
        self,
        template_str: str,
        context: dict[str, Any],
    ) -> TemplateRenderResult:
        """Render a template string directly.
        
        Args:
            template_str: Jinja2 template string
            context: Variables to pass to template
            
        Returns:
            TemplateRenderResult with rendered content.
        """
        try:
            template = self._builtin_env.from_string(template_str)
            rendered = template.render(**context)
            return TemplateRenderResult(
                template_name="<string>",
                rendered_content=rendered.strip(),
                success=True,
            )
        except Exception as e:
            logger.error(f"Error rendering template string: {e}")
            return TemplateRenderResult(
                template_name="<string>",
                rendered_content="",
                success=False,
                error_message=str(e),
            )
    
    def list_templates(self) -> list[TemplateInfo]:
        """List all available templates.
        
        Returns:
            List of TemplateInfo for built-in and filesystem templates.
        """
        templates = []
        
        # Add built-in templates
        builtin_descriptions = {
            "function_stub": "Generate a function stub with docstring",
            "class_stub": "Generate a class stub with methods",
            "docstring": "Generate a docstring block",
            "footnote": "Generate a footnote reference",
            "api_response": "Generate an API response JSON",
            "test_case": "Generate a test case with arrange/act/assert",
            "markdown_section": "Generate a markdown section with subsections",
        }
        
        for name in BUILTIN_TEMPLATES:
            templates.append(TemplateInfo(
                name=name,
                path="<builtin>",
                description=builtin_descriptions.get(name),
                required_variables=self._get_template_variables(BUILTIN_TEMPLATES[name]),
            ))
        
        # Add filesystem templates
        if self._env is not None:
            for name in self._env.list_templates():
                templates.append(TemplateInfo(
                    name=name,
                    path=str(Path(self._config.templates_dir) / name),
                    description=None,
                    required_variables=[],
                ))
        
        return templates
    
    def _get_template_variables(self, template_str: str) -> list[str]:
        """Extract variable names from a template string."""
        # Simple regex-free parsing for {{ variable }}
        variables = set()
        i = 0
        while i < len(template_str):
            if template_str[i:i+2] == "{{":
                end = template_str.find("}}", i)
                if end != -1:
                    var_expr = template_str[i+2:end].strip()
                    # Extract variable name (before any filters or operators)
                    var_name = var_expr.split("|")[0].split(".")[0].strip()
                    if var_name and not var_name.startswith("%") and var_name.isidentifier():
                        variables.add(var_name)
                    i = end + 2
                else:
                    i += 1
            else:
                i += 1
        return sorted(variables)
    
    def get_builtin_template(self, name: str) -> str | None:
        """Get the raw content of a built-in template.
        
        Args:
            name: Name of built-in template
            
        Returns:
            Template string or None if not found.
        """
        return BUILTIN_TEMPLATES.get(name)


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeTemplateEngineTool:
    """Fake implementation for testing."""
    
    def __init__(self) -> None:
        """Initialize the fake engine."""
        self._results: dict[str, TemplateRenderResult] = {}
        self._templates: list[TemplateInfo] = []
    
    def set_result(self, template_name: str, result: TemplateRenderResult) -> None:
        """Set a predefined result for a template."""
        self._results[template_name] = result
    
    def set_templates(self, templates: list[TemplateInfo]) -> None:
        """Set the list of available templates."""
        self._templates = templates
    
    def render(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> TemplateRenderResult:
        """Return predefined result or success result."""
        return self._results.get(
            template_name,
            TemplateRenderResult(
                template_name=template_name,
                rendered_content=f"Rendered {template_name}",
                success=True,
            ),
        )
    
    def render_string(
        self,
        template_str: str,
        context: dict[str, Any],
    ) -> TemplateRenderResult:
        """Return success result for string rendering."""
        return TemplateRenderResult(
            template_name="<string>",
            rendered_content="Rendered string template",
            success=True,
        )
    
    def list_templates(self) -> list[TemplateInfo]:
        """Return predefined templates list."""
        return self._templates


# =============================================================================
# Singleton Instance
# =============================================================================

_template_engine_tool: TemplateEngineTool | None = None


def get_template_engine_tool(
    config: TemplateEngineConfig | None = None,
) -> TemplateEngineTool:
    """Get or create singleton TemplateEngineTool instance.
    
    Args:
        config: Optional configuration. Only used on first call.
        
    Returns:
        Cached TemplateEngineTool instance.
    """
    global _template_engine_tool
    if _template_engine_tool is None:
        _template_engine_tool = TemplateEngineTool(config)
    return _template_engine_tool

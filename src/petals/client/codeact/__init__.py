"""
CodeAct - Python AST to ToolCall DAG Parser

This package provides tools for parsing Python source code and extracting
tool calls to build a ToolCallDAG with proper dependency resolution.

Example:
    >>> from petals.client.codeact import CodeActParser
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> # Register tools
    >>> registry = ToolRegistry()
    >>> registry.register("echo", lambda text: text)
    >>> registry.register("uppercase", lambda text: text.upper())
    >>>
    >>> # Parse Python source
    >>> parser = CodeActParser(registry)
    >>> source = '''
    ... result1 = echo(text="hello")
    ... result2 = uppercase(text="${result1}")
    ... '''
    >>> dag, var_map = parser.parse(source)
    >>> # dag: ToolCallDAG with 2 nodes
    >>> # var_map: {"result1": "echo_1", "result2": "uppercase_1"}
"""

from .exceptions import (
    CodeActCycleError,
    CodeActDependencyError,
    CodeActParserError,
    CodeActSyntaxError,
    CodeActUnknownToolError,
)
from .parser import CodeActParser, ToolCallInfo

__all__ = [
    # Main parser
    "CodeActParser",
    # Data classes
    "ToolCallInfo",
    # Exceptions
    "CodeActParserError",
    "CodeActSyntaxError",
    "CodeActUnknownToolError",
    "CodeActDependencyError",
    "CodeActCycleError",
]

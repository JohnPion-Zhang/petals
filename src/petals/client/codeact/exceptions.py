"""
CodeAct Parser Exceptions

Custom exceptions raised by the CodeActParser for error handling.
"""


class CodeActParserError(Exception):
    """Base exception for CodeAct parser errors."""

    def __init__(self, message: str, line: int | None = None):
        self.message = message
        self.line = line
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with line number if available."""
        if self.line is not None:
            return f"{self.message} (line {self.line})"
        return self.message


class CodeActSyntaxError(CodeActParserError):
    """Raised when Python source code has syntax errors.

    This exception is raised when the AST parser encounters invalid
    Python syntax in the source code.

    Example:
        >>> parser.parse("x = 1 +")  # Incomplete expression
        CodeActSyntaxError: Syntax error in source: unexpected EOF
    """

    def __init__(self, message: str, line: int | None = None):
        super().__init__(f"Syntax error in source: {message}", line)


class CodeActUnknownToolError(CodeActParserError):
    """Raised when a tool call references an unregistered tool.

    This exception is raised when the parser encounters a function call
    that matches a tool name but the tool is not registered in the registry.

    Example:
        >>> parser.parse('unknown_tool(text="hello")')
        CodeActUnknownToolError: Unknown tool: unknown_tool

    Attributes:
        tool_name: The name of the unknown tool.
    """

    def __init__(
        self,
        tool_name: str,
        line: int | None = None,
        available_tools: list[str] | None = None,
    ):
        self.tool_name = tool_name
        self.available_tools = available_tools or []

        message = f"Unknown tool: {tool_name!r}"
        if available_tools:
            message += f". Available tools: {', '.join(available_tools)}"

        super().__init__(message, line)


class CodeActDependencyError(CodeActParserError):
    """Raised when a dependency reference cannot be resolved.

    This exception is raised when a tool call references a variable
    that doesn't exist or hasn't been assigned yet.

    Example:
        >>> parser.parse('result = uppercase(text="${undefined_var}")')
        CodeActDependencyError: Unresolved dependency: undefined_var
    """

    def __init__(self, variable_name: str, line: int | None = None):
        self.variable_name = variable_name
        super().__init__(f"Unresolved dependency: {variable_name!r}", line)


class CodeActCycleError(CodeActParserError):
    """Raised when circular dependencies are detected in the DAG.

    This exception is raised when the parsed tool calls form a cycle,
    which would cause infinite loops during execution.

    Example:
        >>> parser.parse('''
        ...     a = echo(text="${b}")
        ...     b = uppercase(text="${a}")
        ... ''')
        CodeActCycleError: Circular dependency detected: a -> b -> a
    """

    def __init__(self, cycle_path: list[str]):
        self.cycle_path = cycle_path
        cycle_str = " -> ".join(cycle_path)
        super().__init__(f"Circular dependency detected: {cycle_str}")

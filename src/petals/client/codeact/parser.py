"""
CodeActParser - Python AST to ToolCall DAG Parser

Parses Python source code using the ast module and extracts tool calls,
building a ToolCallDAG with proper dependency resolution.

Architecture:
    Python Source Code
           |
           v
    CodeActParser.parse()
           |
           v
    ast.parse() -> AST tree
           |
           v
    ASTVisitor (walk the tree)
           |
           v
    Extract: Call name, args, line numbers
           |
           v
    Build: ToolCallNode + ToolCallDAG
           |
           v
    Return: (dag, variable_map)

Example:
    >>> from petals.client.codeact import CodeActParser
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> registry = ToolRegistry()
    >>> registry.register("echo", lambda text: text)
    >>> registry.register("uppercase", lambda text: text.upper())
    >>>
    >>> parser = CodeActParser(registry)
    >>> source = '''
    ... result1 = echo(text="hello")
    ... result2 = uppercase(text="${result1}")
    ... '''
    >>> dag, var_map = parser.parse(source)
    >>> # dag: ToolCallDAG with 2 nodes
    >>> # var_map: {"result1": "echo_1", "result2": "uppercase_1"}
"""
import ast
import re
from dataclasses import dataclass, field
from typing import Any

from petals.client.dag import ToolCallDAG
from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.tool_registry import ToolRegistry

from .exceptions import (
    CodeActCycleError,
    CodeActDependencyError,
    CodeActSyntaxError,
    CodeActUnknownToolError,
)


# Pattern for ${variable} syntax
DEPENDENCY_PATTERN = re.compile(r"^\$\{(\w+)\}$")

# Built-in Python functions that should be silently skipped (not treated as tools)
# These are commonly used for control flow and data manipulation
BUILTIN_FUNCTIONS: set[str] = {
    # Type conversion
    "str", "int", "float", "bool", "list", "dict", "tuple", "set", "frozenset",
    "bytes", "bytearray", "memoryview", "complex", "range",
    # Object inspection
    "type", "isinstance", "issubclass", "callable", "hasattr", "getattr", "setattr", "delattr",
    # Iterables
    "len", "iter", "next", "enumerate", "zip", "map", "filter", "sorted", "reversed", "any", "all",
    "sum", "min", "max", "abs", "divmod", "pow", "round", "format",
    # I/O
    "print", "input", "open", "file",  # noqa: F401
    # Utilities
    "repr", "hash", "id", "vars", "dir", "help", "locals", "globals",
    # Math
    "abs", "round", "pow", "divmod",
    # Other
    "slice", "property", "classmethod", "staticmethod", "super",
    "object", "classmethod", "type", "vars",
    # Async
    "await",
    # Special
    "__import__",
}

# Counter for generating unique node IDs
_node_counter: dict[str, int] = {}


def _generate_node_id(tool_name: str) -> str:
    """Generate a unique node ID for a tool.

    Args:
        tool_name: Name of the tool.

    Returns:
        Unique node ID in format: {tool_name}_{counter}
    """
    global _node_counter
    if tool_name not in _node_counter:
        _node_counter[tool_name] = 0
    _node_counter[tool_name] += 1
    return f"{tool_name}_{_node_counter[tool_name]}"


def _reset_node_counter() -> None:
    """Reset the node ID counter. Useful for testing."""
    global _node_counter
    _node_counter = {}


@dataclass
class ToolCallInfo:
    """Information extracted from a tool call in Python source.

    Attributes:
        node_id: Unique ID for this node in the DAG.
        tool_name: Name of the tool being called.
        arguments: Dictionary of arguments to pass to the tool.
        line: Source line number where this call appears.
        variable_name: Variable name this call is assigned to (if any).
        dependencies: List of variable names this call depends on.
    """

    node_id: str
    tool_name: str
    arguments: dict[str, Any]
    line: int
    variable_name: str | None = None
    dependencies: list[str] = field(default_factory=list)


class CodeActParser:
    """Parse Python source code to ToolCallDAG.

    This parser extracts tool calls from Python source code using the AST
    module and builds a DAG with proper dependency resolution.

    The parser supports:
    - Simple tool calls: `echo(text="hello")`
    - Variable assignments: `result = echo(text="hello")`
    - Dependencies with ${var} syntax: `uppercase(text="${result}")`
    - Dependencies with from_dep syntax: `uppercase(text={"from_dep": "result"})`
    - Complex expressions as arguments

    Attributes:
        registry: ToolRegistry containing registered tools.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register("echo", lambda t: t)
        >>> parser = CodeActParser(registry)
        >>> dag, var_map = parser.parse('result = echo(text="hello")')
    """

    def __init__(self, registry: ToolRegistry):
        """Initialize the parser with a tool registry.

        Args:
            registry: ToolRegistry containing available tools.
        """
        self.registry = registry

    def parse(self, source: str) -> tuple[ToolCallDAG, dict[str, str]]:
        """Parse Python source to ToolCallDAG.

        Args:
            source: Python source code to parse.

        Returns:
            Tuple of (ToolCallDAG, variable_map) where:
            - ToolCallDAG contains all extracted tool calls with dependencies
            - variable_map maps variable names to node IDs

        Raises:
            CodeActSyntaxError: If source has invalid Python syntax.
            CodeActUnknownToolError: If an unknown tool is called.
            CodeActCycleError: If circular dependencies are detected.
        """
        # Reset counter for consistent IDs within this parse
        _reset_node_counter()

        # Parse the source code
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise CodeActSyntaxError(str(e), line=e.lineno)

        # Extract tool calls from AST
        tool_calls = self._extract_calls(tree)

        # Build the DAG
        dag = self._build_dag(tool_calls)

        # Build variable map
        var_map = {
            call.variable_name: call.node_id
            for call in tool_calls
            if call.variable_name is not None
        }

        return dag, var_map

    def _extract_calls(self, tree: ast.AST) -> list[ToolCallInfo]:
        """Extract tool calls from AST.

        Args:
            tree: Parsed AST tree.

        Returns:
            List of ToolCallInfo for each tool call found.
        """
        visitor = ToolCallExtractor(self.registry)
        visitor.visit(tree)
        return visitor.tool_calls

    def _build_dag(self, calls: list[ToolCallInfo]) -> ToolCallDAG:
        """Build DAG from extracted tool calls.

        Args:
            calls: List of extracted tool calls.

        Returns:
            ToolCallDAG with all nodes and edges.

        Raises:
            CodeActCycleError: If circular dependencies detected.
        """
        dag = ToolCallDAG()

        # Create variable -> node_id mapping for dependency resolution
        var_to_node: dict[str, str] = {}

        # First pass: create all nodes
        for call in calls:
            # Resolve dependencies from variable names to node IDs
            resolved_deps = []
            for dep_var in call.dependencies:
                if dep_var in var_to_node:
                    resolved_deps.append(var_to_node[dep_var])

            node = ToolCallNode(
                id=call.node_id,
                name=call.tool_name,
                arguments=call.arguments,
                dependencies=resolved_deps,
            )
            dag.add_node(node)

            # Track variable assignment
            if call.variable_name:
                var_to_node[call.variable_name] = call.node_id

        # Check for cycles
        cycle = dag.detect_cycle()
        if cycle:
            raise CodeActCycleError(cycle)

        return dag


class ToolCallExtractor(ast.NodeVisitor):
    """AST visitor that extracts tool calls from Python source.

    This visitor walks the AST and identifies function calls that
    match registered tool names, extracting their arguments and
    tracking variable assignments.
    """

    def __init__(self, registry: ToolRegistry):
        """Initialize the extractor.

        Args:
            registry: ToolRegistry containing available tools.
        """
        self.registry = registry
        self.tool_calls: list[ToolCallInfo] = []
        self._tool_names: set[str] = set(registry._tools.keys())  # noqa: SLF001

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment node.

        Extracts tool calls from the right-hand side and tracks
        the variable name for dependency resolution.

        Args:
            node: AST Assign node.
        """
        # Get variable name (only handle simple single-target assignments)
        var_name = None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

        # Visit the value (which might be a Call)
        self.visit(node.value)

        # If the last tool call was assigned to a variable, track it
        if var_name and self.tool_calls:
            self.tool_calls[-1].variable_name = var_name

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a function call node.

        Checks if the function is a registered tool and extracts
        its information if so. Built-in Python functions are silently skipped.

        Args:
            node: AST Call node.
        """
        # Get the function name
        func_name = self._get_func_name(node.func)

        if func_name is None:
            # Not a simple name call (e.g., method call), skip
            return

        # Skip built-in Python functions - they are not tools
        if func_name in BUILTIN_FUNCTIONS:
            return

        # Check if this is a registered tool
        if func_name not in self._tool_names:
            raise CodeActUnknownToolError(
                tool_name=func_name,
                line=node.lineno,
                available_tools=list(self._tool_names),
            )

        # Extract arguments
        arguments, dependencies = self._extract_arguments(node)

        # Create ToolCallInfo
        call_info = ToolCallInfo(
            node_id=_generate_node_id(func_name),
            tool_name=func_name,
            arguments=arguments,
            line=node.lineno or 0,
            dependencies=dependencies,
        )

        self.tool_calls.append(call_info)

    def _get_func_name(self, node: ast.AST) -> str | None:
        """Get the function name from a function node.

        Args:
            node: AST node representing a function.

        Returns:
            Function name if it's a simple Name node, None otherwise.
        """
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _extract_arguments(
        self, node: ast.Call
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract arguments from a function call.

        Handles:
        - Keyword arguments: `echo(text="hello")`
        - Positional arguments: `concat("hello", "world")`
        - Dependency markers: `echo(text="${var}")` or `echo(text={"from_dep": "var"})`

        Args:
            node: AST Call node.

        Returns:
            Tuple of (arguments dict, list of dependency variable names).
        """
        arguments = {}
        dependencies = []

        # Process keyword arguments
        for kw in node.keywords:
            if kw.arg is None:
                # **kwargs - skip for now
                continue

            value, deps = self._extract_value(kw.value)
            arguments[kw.arg] = value
            dependencies.extend(deps)

        # Process positional arguments
        # For simplicity, we'll convert them to arg_0, arg_1, etc.
        # or use the function's parameter names if available
        for i, arg in enumerate(node.args):
            value, deps = self._extract_value(arg)
            arguments[f"arg_{i}"] = value
            dependencies.extend(deps)

        return arguments, dependencies

    def _extract_value(self, node: ast.AST) -> tuple[Any, list[str]]:
        """Extract a Python value from an AST node.

        Handles literal values, dependency markers, and expressions.

        Args:
            node: AST node representing a value.

        Returns:
            Tuple of (extracted value, list of dependency variable names).
        """
        dependencies = []

        # String literal with dependency marker: "${var}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            match = DEPENDENCY_PATTERN.match(value)
            if match:
                dependencies.append(match.group(1))
                # Replace with None as placeholder - will be resolved at runtime
                value = {"from_dep": match.group(1)}
            return value, dependencies

        # Dict with from_dep marker
        if isinstance(node, ast.Dict):
            # Check if it has a "from_dep" key
            for key, val in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "from_dep":
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        dependencies.append(val.value)
                        return {"from_dep": val.value}, dependencies

            # Regular dict - process all values
            result = {}
            for key, val in zip(node.keys, node.values):
                if isinstance(key, ast.Constant):
                    k = key.value
                    v, deps = self._extract_value(val)
                    result[k] = v
                    dependencies.extend(deps)
                else:
                    # Dynamic key - can't process
                    k, deps = self._extract_value(key)
                    dependencies.extend(deps)
            return result, dependencies

        # List literal
        if isinstance(node, ast.List):
            result = []
            for elt in node.elts:
                val, deps = self._extract_value(elt)
                result.append(val)
                dependencies.extend(deps)
            return result, dependencies

        # Tuple literal
        if isinstance(node, ast.Tuple):
            result = []
            for elt in node.elts:
                val, deps = self._extract_value(elt)
                result.append(val)
                dependencies.extend(deps)
            return tuple(result), dependencies

        # Constant (number, bool, None, etc.)
        if isinstance(node, ast.Constant):
            return node.value, dependencies

        # Name (variable reference)
        if isinstance(node, ast.Name):
            # This is a variable reference - treat as dependency
            dependencies.append(node.id)
            return {"from_dep": node.id}, dependencies

        # Binary operations (e.g., string concatenation at parse time)
        if isinstance(node, ast.BinOp):
            # Try to evaluate at parse time
            try:
                left_val, left_deps = self._extract_value(node.left)
                right_val, right_deps = self._extract_value(node.right)
                dependencies.extend(left_deps)
                dependencies.extend(right_deps)

                if not dependencies:
                    # Can evaluate at parse time
                    op_map = {
                        ast.Add: lambda a, b: a + b,
                        ast.Mult: lambda a, b: a * b,
                        ast.Sub: lambda a, b: a - b,
                        ast.Div: lambda a, b: a / b,
                        ast.FloorDiv: lambda a, b: a // b,
                        ast.Mod: lambda a, b: a % b,
                        ast.Pow: lambda a, b: a**b,
                    }
                    op = type(node.op)
                    if op in op_map:
                        return op_map[op](left_val, right_val), dependencies
            except (TypeError, ValueError, AttributeError):
                pass

            # Can't evaluate - return placeholder
            return {"from_dep": f"expr_{id(node)}"}, dependencies

        # Call expression (e.g., len(), str(), etc.)
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node.func)
            if func_name in ("str", "int", "float", "bool", "len", "list", "dict", "tuple", "set"):
                # Built-in functions - try to evaluate
                try:
                    args, deps = self._extract_arguments(node)
                    dependencies.extend(deps)
                    if not dependencies:
                        # Evaluate at parse time
                        if func_name == "str":
                            return str(args.get("arg_0", "")), dependencies
                        elif func_name == "int":
                            return int(args.get("arg_0", 0)), dependencies
                        elif func_name == "float":
                            return float(args.get("arg_0", 0.0)), dependencies
                        elif func_name == "bool":
                            return bool(args.get("arg_0", False)), dependencies
                        elif func_name == "len":
                            return len(args.get("arg_0", [])), dependencies
                        elif func_name == "list":
                            return list(args.get("arg_0", [])), dependencies
                        elif func_name == "dict":
                            return dict(args.get("arg_0", {})), dependencies
                        elif func_name == "tuple":
                            return tuple(args.get("arg_0", ())), dependencies
                        elif func_name == "set":
                            return set(args.get("arg_0", [])), dependencies
                except (TypeError, ValueError):
                    pass

            # Can't evaluate - mark as dependency
            return {"from_dep": f"call_{id(node)}"}, dependencies

        # List comprehension, generator expression, etc.
        # For simplicity, treat as non-evaluable
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
            return {"from_dep": f"comp_{id(node)}"}, dependencies

        # Subscript (e.g., list[0])
        if isinstance(node, ast.Subscript):
            try:
                val, deps = self._extract_value(node.value)
                dependencies.extend(deps)

                if isinstance(node.slice, ast.Constant):
                    index = node.slice.value
                    return val[index], dependencies
            except (TypeError, IndexError, KeyError):
                pass
            return {"from_dep": f"subscript_{id(node)}"}, dependencies

        # Attribute access (e.g., obj.attr)
        if isinstance(node, ast.Attribute):
            try:
                val, deps = self._extract_value(node.value)
                dependencies.extend(deps)
                if not dependencies:
                    return getattr(val, node.attr), dependencies
            except AttributeError:
                pass
            return {"from_dep": f"attr_{id(node)}"}, dependencies

        # Unknown node type - return placeholder
        return {"from_dep": f"unknown_{id(node)}"}, dependencies

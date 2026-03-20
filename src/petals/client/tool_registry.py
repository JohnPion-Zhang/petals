"""
ToolRegistry - Manages available tools and their schemas for the Agent Tool Call Layer.

This module provides a registry for storing, retrieving, and executing tools
that can be called by agents during inference. Tools can be synchronous or
asynchronous functions, and each tool can optionally have an associated
JSON schema for validation and documentation purposes.
"""
from typing import Any, Callable, Dict, Optional

import asyncio


class ToolRegistry:
    """
    A registry for managing available tools and their execution.

    Tools can be registered with optional JSON schemas for documentation
    and validation. Both synchronous and asynchronous functions are supported.

    Example:
        >>> registry = ToolRegistry()
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> registry.register("add", add, schema={"name": "add", "parameters": {...}})
        >>> schema = registry.get_schema()
        >>> result = await registry.execute("add", {"a": 1, "b": 2})
        >>> print(result)
        3
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a tool with the registry.

        Args:
            name: Unique identifier for the tool.
            func: Callable function (sync or async) to register.
            schema: Optional JSON schema describing the tool's parameters.

        Raises:
            ValueError: If a tool with the given name is already registered.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = {"func": func, "schema": schema or {}}

    def get_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Get schemas for all registered tools.

        Returns:
            Dictionary mapping tool names to their schemas.
        """
        return {name: info["schema"] for name, info in self._tools.items()}

    def get_tool(self, name: str) -> Callable:
        """
        Retrieve a registered tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The registered callable function.

        Raises:
            ValueError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name]["func"]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a registered tool with the given arguments.

        Async functions are automatically awaited. If the tool raises an
        exception, it is caught and returned as an error dictionary.

        Args:
            name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            The result of the tool execution, or an error dictionary
            if execution failed.

        Raises:
            ValueError: If no tool with the given name is registered.
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        try:
            func = tool["func"]
            if asyncio.iscoroutinefunction(func):
                return await func(**arguments)
            return func(**arguments)
        except Exception as e:
            return {"error": str(e)}

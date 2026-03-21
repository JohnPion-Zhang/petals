"""
DAG - Directed Acyclic Graph for ToolCall Orchestration

This module provides the core data structures and execution engine for
managing tool call dependencies as a DAG with wave-based parallel execution.

Example:
    >>> from petals.client.dag import ToolCallNode, ToolCallDAG, WaveExecutor
    >>> from petals.client.tool_registry import ToolRegistry
    >>>
    >>> # Create nodes
    >>> search = ToolCallNode(id="search", name="web_search", arguments={"query": "AI"})
    >>> process = ToolCallNode(id="process", name="process", arguments={}, dependencies=["search"])
    >>>
    >>> # Build DAG
    >>> dag = ToolCallDAG()
    >>> dag.add_node(search)
    >>> dag.add_node(process)
    >>>
    >>> # Execute
    >>> executor = WaveExecutor(registry)
    >>> async for completed in executor.execute_dag(dag):
    ...     print(f"Completed: {completed.id}")
"""

from petals.client.dag.dag import ToolCallDAG
from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.dag.wave_executor import WaveExecutor

__all__ = [
    "ToolCallDAG",
    "ToolCallNode",
    "WaveExecutor",
]

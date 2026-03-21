"""
ToolCallNode - DAG Node for ToolCall Orchestration

A node in the ToolCall DAG representing a single tool call.

Note: ToolCallNode represents an ASYNC LLM task to fill parameters - NOT execution.
Tools execute client-side. See WaveExecutor for the execution flow.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from petals.data_structures import CallStatus


@dataclass
class ToolCallNode:
    """A node in the ToolCall DAG representing a single tool call.

    This class represents the planning/parameter-filling phase of a tool call.
    The actual execution happens client-side via ToolRegistry.

    Attributes:
        id: Unique node identifier within the DAG.
        name: Name of the tool to be called.
        arguments: Dictionary of tool arguments (may include dependency references).
        dependencies: List of parent node IDs that must complete before this node.
        dependents: List of child node IDs that depend on this node (auto-managed).
        execution_key: Optional key for deduplicating identical tool calls.
        status: Current execution status of the node.
        result: Result of tool execution (None until completed).
        error: Error message if execution failed.
        execution_trace: stdout/stderr from CodeAct pattern execution.
        error_feedback: Formatted error context for LLM correction.
        retry_count: Number of retry attempts made.
        requires_verification: Whether RLM verification is required before proceeding.

    Example:
        >>> node = ToolCallNode(
        ...     id="search_1",
        ...     name="web_search",
        ...     arguments={"query": "latest AI news"},
        ...     dependencies=[],
        ...     execution_key="web_search:latest AI news"
        ... )
        >>> node.status = CallStatus.DONE
        >>> node.result = {"articles": [...]}
    """

    # Identity
    id: str
    name: str
    arguments: Dict[str, Any]

    # DAG Structure
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    execution_key: Optional[str] = None

    # Execution State
    status: CallStatus = CallStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    # CodeAct/RLM Extensions
    execution_trace: Optional[str] = None
    error_feedback: Optional[str] = None
    retry_count: int = 0
    requires_verification: bool = False

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no dependents).

        Returns:
            True if this node has no dependent nodes, False otherwise.
        """
        return len(self.dependents) == 0

    @property
    def is_root(self) -> bool:
        """Check if this node is a root (has no dependencies).

        Returns:
            True if this node has no dependencies, False otherwise.
        """
        return len(self.dependencies) == 0

    @property
    def is_done(self) -> bool:
        """Check if node execution is complete (success or failure).

        Returns:
            True if status is DONE or FAILED, False otherwise.
        """
        return self.status in (CallStatus.DONE, CallStatus.FAILED)

    @property
    def is_pending(self) -> bool:
        """Check if node is waiting for execution.

        Returns:
            True if status is PENDING, False otherwise.
        """
        return self.status == CallStatus.PENDING

    def mark_running(self) -> None:
        """Mark node as currently running."""
        self.status = CallStatus.RUNNING

    def mark_done(self, result: Any) -> None:
        """Mark node as successfully completed with result.

        Args:
            result: The execution result to store.
        """
        self.status = CallStatus.DONE
        self.result = result

    def mark_failed(self, error: str, error_feedback: Optional[str] = None) -> None:
        """Mark node as failed with error information.

        Args:
            error: Human-readable error message.
            error_feedback: Detailed error context (e.g., traceback) for LLM.
        """
        self.status = CallStatus.FAILED
        self.error = error
        if error_feedback:
            self.error_feedback = error_feedback

    def increment_retry(self) -> None:
        """Increment retry count after a failed attempt."""
        self.retry_count += 1

    def can_retry(self, max_retries: int) -> bool:
        """Check if node can be retried.

        Args:
            max_retries: Maximum number of retries allowed.

        Returns:
            True if retry_count < max_retries, False otherwise.
        """
        return self.retry_count < max_retries

    def add_dependency(self, node_id: str) -> None:
        """Add a dependency on another node.

        Args:
            node_id: ID of the node this depends on.
        """
        if node_id not in self.dependencies:
            self.dependencies.append(node_id)

    def remove_dependency(self, node_id: str) -> None:
        """Remove a dependency on another node.

        Args:
            node_id: ID of the dependency to remove.
        """
        if node_id in self.dependencies:
            self.dependencies.remove(node_id)

    def add_dependent(self, node_id: str) -> None:
        """Add a dependent node (child).

        Args:
            node_id: ID of the dependent node.
        """
        if node_id not in self.dependents:
            self.dependents.append(node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "execution_key": self.execution_key,
            "status": self.status.value if isinstance(self.status, CallStatus) else self.status,
            "result": self.result,
            "error": self.error,
            "execution_trace": self.execution_trace,
            "error_feedback": self.error_feedback,
            "retry_count": self.retry_count,
            "requires_verification": self.requires_verification,
            "is_leaf": self.is_leaf,
            "is_root": self.is_root,
        }

    def __repr__(self) -> str:
        """String representation of the node."""
        status_str = self.status.value if isinstance(self.status, CallStatus) else self.status
        return (
            f"ToolCallNode(id={self.id!r}, name={self.name!r}, "
            f"status={status_str}, deps={len(self.dependencies)}, "
            f"dependents={len(self.dependents)})"
        )

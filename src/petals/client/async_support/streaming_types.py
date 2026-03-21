"""
Streaming Types - Stream event types and base class for SSE support.

This module provides the core streaming event types used throughout
the async streaming layer, including SSE (Server-Sent Events) formatting.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class StreamEventType(Enum):
    """Types of events in the SSE stream.

    These event types represent the lifecycle of LLM output and tool execution.

    Example:
        >>> event_type = StreamEventType.TEXT_CHUNK
        >>> print(event_type.value)
        'text_chunk'
    """

    TEXT_CHUNK = "text_chunk"
    """LLM text output chunk (partial generation)."""

    TOOL_CALL_PENDING = "tool_call_pending"
    """Tool call detected but parameters not yet resolved."""

    TOOL_CALL_READY = "tool_call_ready"
    """Tool parameters resolved, ready to execute."""

    TOOL_EXECUTING = "tool_executing"
    """Tool execution has started."""

    TOOL_RESULT = "tool_result"
    """Tool execution completed with result."""

    FINAL = "final"
    """All generation and tool calls complete."""

    ERROR = "error"
    """An error occurred during processing."""


@dataclass
class StreamEvent:
    """A single event in the SSE stream.

    Events represent discrete units of streaming output, such as text
    chunks, tool calls, or results.

    Attributes:
        type: The type of event (StreamEventType).
        data: Dictionary containing event-specific data.
        timestamp: Event timestamp (defaults to current time).

    Example:
        >>> event = StreamEvent(
        ...     type=StreamEventType.TEXT_CHUNK,
        ...     data={"text": "Hello, ", "is_final": False}
        ... )
        >>> print(event.to_sse())
        'event: text_chunk\\ndata: {"text": "Hello, ", "is_final": false}\\n\\n'
    """

    type: StreamEventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as SSE data string.

        Returns a properly formatted Server-Sent Events string
        that can be sent to clients.

        Returns:
            SSE-formatted string with event type and JSON data.

        Example:
            >>> event = StreamEvent(
            ...     type=StreamEventType.TOOL_RESULT,
            ...     data={"tool": "search", "result": "found 42 results"}
            ... )
            >>> print(event.to_sse())
            event: tool_result
            data: {"tool": "search", "result": "found 42 results"}

        """
        return f"event: {self.type.value}\ndata: {json.dumps(self.data)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with type, data, and timestamp fields.

        Example:
            >>> event = StreamEvent(
            ...     type=StreamEventType.FINAL,
            ...     data={"total_chunks": 10}
            ... )
            >>> d = event.to_dict()
            >>> print(d["type"])
            'final'
        """
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def text_chunk(
        cls,
        text: str,
        is_final: bool = False,
        chunk_index: Optional[int] = None
    ) -> "StreamEvent":
        """Create a text chunk event.

        Args:
            text: The text content of the chunk.
            is_final: Whether this is the final chunk.
            chunk_index: Optional index of this chunk.

        Returns:
            A new StreamEvent for a text chunk.
        """
        data: Dict[str, Any] = {"text": text, "is_final": is_final}
        if chunk_index is not None:
            data["chunk_index"] = chunk_index
        return cls(type=StreamEventType.TEXT_CHUNK, data=data)

    @classmethod
    def tool_pending(
        cls,
        tool_name: str,
        node_id: str,
        partial_args: Optional[Dict[str, Any]] = None
    ) -> "StreamEvent":
        """Create a tool call pending event.

        Args:
            tool_name: Name of the tool.
            node_id: DAG node ID.
            partial_args: Partially resolved arguments (may have dependencies).

        Returns:
            A new StreamEvent for a pending tool call.
        """
        return cls(
            type=StreamEventType.TOOL_CALL_PENDING,
            data={
                "tool_name": tool_name,
                "node_id": node_id,
                "partial_args": partial_args or {},
            }
        )

    @classmethod
    def tool_ready(
        cls,
        tool_name: str,
        node_id: str,
        resolved_args: Dict[str, Any]
    ) -> "StreamEvent":
        """Create a tool call ready event.

        Args:
            tool_name: Name of the tool.
            node_id: DAG node ID.
            resolved_args: Fully resolved arguments.

        Returns:
            A new StreamEvent for a ready tool call.
        """
        return cls(
            type=StreamEventType.TOOL_CALL_READY,
            data={
                "tool_name": tool_name,
                "node_id": node_id,
                "resolved_args": resolved_args,
            }
        )

    @classmethod
    def tool_executing(
        cls,
        tool_name: str,
        node_id: str,
        execution_key: Optional[str] = None
    ) -> "StreamEvent":
        """Create a tool executing event.

        Args:
            tool_name: Name of the tool.
            node_id: DAG node ID.
            execution_key: Optional deduplication key.

        Returns:
            A new StreamEvent for a tool starting execution.
        """
        data: Dict[str, Any] = {
            "tool_name": tool_name,
            "node_id": node_id,
        }
        if execution_key:
            data["execution_key"] = execution_key
        return cls(type=StreamEventType.TOOL_EXECUTING, data=data)

    @classmethod
    def tool_result(
        cls,
        tool_name: str,
        node_id: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None
    ) -> "StreamEvent":
        """Create a tool result event.

        Args:
            tool_name: Name of the tool.
            node_id: DAG node ID.
            result: The result from tool execution.
            success: Whether execution succeeded.
            error: Error message if execution failed.

        Returns:
            A new StreamEvent for a tool result.
        """
        data: Dict[str, Any] = {
            "tool_name": tool_name,
            "node_id": node_id,
            "result": result,
            "success": success,
        }
        if error:
            data["error"] = error
        return cls(type=StreamEventType.TOOL_RESULT, data=data)

    @classmethod
    def final(
        cls,
        total_chunks: int = 0,
        total_tools: int = 0,
        final_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "StreamEvent":
        """Create a final event.

        Args:
            total_chunks: Total number of text chunks.
            total_tools: Total number of tool calls.
            final_text: Complete aggregated text.
            metadata: Additional metadata.

        Returns:
            A new StreamEvent for the final state.
        """
        data: Dict[str, Any] = {
            "total_chunks": total_chunks,
            "total_tools": total_tools,
        }
        if final_text is not None:
            data["final_text"] = final_text
        if metadata:
            data["metadata"] = metadata
        return cls(type=StreamEventType.FINAL, data=data)

    @classmethod
    def error(
        cls,
        message: str,
        error_type: str = "general",
        recoverable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> "StreamEvent":
        """Create an error event.

        Args:
            message: Human-readable error message.
            error_type: Type/category of error.
            recoverable: Whether the error might be recoverable.
            context: Additional error context.

        Returns:
            A new StreamEvent for an error.
        """
        data: Dict[str, Any] = {
            "message": message,
            "error_type": error_type,
            "recoverable": recoverable,
        }
        if context:
            data["context"] = context
        return cls(type=StreamEventType.ERROR, data=data)

    def __repr__(self) -> str:
        """String representation of the event."""
        return f"StreamEvent(type={self.type.value}, data_keys={list(self.data.keys())})"

"""
Async/Streaming Module for Petals Client

This module provides async and streaming support for the petals client,
including task pooling, streaming event types, aggregation, and structured
output enforcement.

Example:
    >>> from petals.client.async_support import TaskPool, StreamingExecutor
    >>> from petals.client.async_support.streaming_types import StreamEvent, StreamEventType
    >>>
    >>> # Create async task pool
    >>> pool = TaskPool(max_concurrency=10)
    >>>
    >>> # Execute streaming DAG
    >>> executor = StreamingExecutor(registry)
    >>> async for event in executor.execute_streaming(dag):
    ...     print(f"Event: {event.type}")
"""

from petals.client.async_support.task_pool import TaskPool
from petals.client.async_support.streaming_types import StreamEvent, StreamEventType
from petals.client.async_support.streaming_aggregator import StreamingAggregator, AggregationResult
from petals.client.async_support.structured_output import (
    StructuredOutputEnforcer,
    OutputSchema,
    ValidationResult,
    ValidationStatus,
)
from petals.client.async_support.streaming_executor import StreamingExecutor

__all__ = [
    # Task Pool
    "TaskPool",
    # Streaming Types
    "StreamEvent",
    "StreamEventType",
    # Streaming Aggregator
    "StreamingAggregator",
    "AggregationResult",
    # Structured Output
    "StructuredOutputEnforcer",
    "OutputSchema",
    "ValidationResult",
    "ValidationStatus",
    # Streaming Executor
    "StreamingExecutor",
]

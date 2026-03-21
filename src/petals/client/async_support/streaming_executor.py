"""
StreamingExecutor - Combines DAG execution with streaming output support.

This module provides a streaming executor that combines wave-based DAG
execution with SSE event generation, structured output validation,
and concurrent task management.
"""
import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from petals.client.async_support.streaming_aggregator import AggregationResult, StreamingAggregator
from petals.client.async_support.streaming_types import StreamEvent, StreamEventType
from petals.client.async_support.structured_output import OutputSchema, StructuredOutputEnforcer
from petals.client.async_support.task_pool import TaskPool
from petals.client.dag.dag import ToolCallDAG
from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.dag.wave_executor import WaveExecutor
from petals.client.tool_registry import ToolRegistry
from petals.data_structures import CallStatus

logger = logging.getLogger(__name__)


class StreamingExecutor:
    """Combines DAG execution with streaming output support.

    Features:
    - Wave-based DAG execution (via WaveExecutor)
    - SSE event generation
    - Structured output validation
    - Concurrent task management (via TaskPool)
    - Streaming result aggregation (via StreamingAggregator)

    Example:
        >>> from petals.client.async import StreamingExecutor
        >>> from petals.client.dag import ToolCallDAG, ToolCallNode
        >>>
        >>> executor = StreamingExecutor(registry, max_concurrency=10)
        >>>
        >>> # Execute with streaming
        >>> async for event in executor.execute_streaming(dag):
        ...     print(f"Event: {event.type.value}")
        >>>
        >>> # Graceful shutdown
        >>> await executor.shutdown()
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_concurrency: int = 10,
        timeout: float = 30.0,
        enable_structured_output: bool = True
    ) -> None:
        """Initialize the streaming executor.

        Args:
            registry: ToolRegistry containing available tools.
            max_concurrency: Maximum concurrent executions.
            timeout: Maximum time in seconds for tool execution.
            enable_structured_output: Whether to enable structured output validation.

        Example:
            >>> executor = StreamingExecutor(
            ...     registry,
            ...     max_concurrency=5,
            ...     timeout=60.0,
            ...     enable_structured_output=True
            ... )
        """
        self.registry = registry
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.enable_structured_output = enable_structured_output

        # Initialize components
        self._task_pool = TaskPool(max_concurrency=max_concurrency)
        self._aggregator = StreamingAggregator()
        self._enforcer = (
            StructuredOutputEnforcer()
            if enable_structured_output
            else None
        )

        # Wave executor for DAG execution
        self._wave_executor = WaveExecutor(
            registry=registry,
            timeout=timeout,
            concurrency_limit=max_concurrency
        )

        # Statistics
        self._total_events: int = 0
        self._total_tools_executed: int = 0
        self._start_time: float = 0.0

    async def execute_streaming(
        self,
        dag: ToolCallDAG,
        initial_args: Dict[str, Any] = None
    ) -> AsyncIterator[StreamEvent]:
        """Execute DAG with streaming output.

        Yields StreamEvents as tools execute, providing real-time
        updates about the execution progress.

        Args:
            dag: The ToolCallDAG to execute.
            initial_args: Optional initial arguments for root nodes.

        Yields:
            StreamEvents representing the execution lifecycle.

        Example:
            >>> async for event in executor.execute_streaming(dag):
            ...     if event.type == StreamEventType.TEXT_CHUNK:
            ...         print(f"Text: {event.data['text']}")
            ...     elif event.type == StreamEventType.TOOL_RESULT:
            ...         print(f"Tool done: {event.data['tool_name']}")
        """
        self._start_time = asyncio.get_event_loop().time()
        self._total_events = 0
        self._total_tools_executed = 0

        logger.info(f"Starting streaming execution for DAG with {len(dag)} nodes")

        try:
            # Check for cycles
            cycle = dag.detect_cycle()
            if cycle:
                yield StreamEvent.error(
                    message=f"Cycle detected in DAG: {' -> '.join(cycle)}",
                    error_type="dag_error",
                    recoverable=False
                )
                return

            # Compute waves
            waves = dag.get_waves()
            logger.debug(f"Executing {len(waves)} waves")

            # Results cache for dependency resolution
            results_cache: Dict[str, Any] = (
                initial_args.copy() if initial_args else {}
            )

            # Execute wave by wave
            for wave_idx, wave in enumerate(waves):
                logger.debug(
                    f"Executing wave {wave_idx + 1}/{len(waves)} "
                    f"with {len(wave)} nodes"
                )

                # Process each node in the wave
                for node in wave:
                    # Emit pending event
                    yield await self._emit_tool_events(
                        node,
                        "pending",
                        {"wave": wave_idx + 1, "wave_total": len(waves)}
                    )

                    # Resolve dependencies
                    resolved_args = self._resolve_dependencies(node, results_cache)

                    # Emit ready event
                    yield await self._emit_tool_events(
                        node,
                        "ready",
                        {"resolved_args": resolved_args}
                    )

                    # Execute the tool
                    result = await self._execute_tool(node, resolved_args)

                    # Process result and emit events
                    if isinstance(result, dict) and "error" in result:
                        yield await self._emit_tool_events(
                            node,
                            "result",
                            {
                                "result": result,
                                "success": False,
                                "error": result.get("error")
                            }
                        )
                    else:
                        # Emit executing event
                        yield await self._emit_tool_events(
                            node,
                            "executing",
                            {}
                        )

                        # Aggregate result if structured output enabled
                        processed_result = await self._aggregate_result(result)

                        # Emit result event
                        yield await self._emit_tool_events(
                            node,
                            "result",
                            {
                                "result": processed_result,
                                "success": True
                            }
                        )

                        # Update results cache
                        results_cache[node.id] = processed_result
                        self._total_tools_executed += 1

            # Emit final event
            current_result = self._aggregator.get_current_result()
            yield StreamEvent.final(
                total_chunks=self._total_events,
                total_tools=self._total_tools_executed,
                metadata={
                    "waves_executed": len(waves),
                    "results_cache_keys": list(results_cache.keys()),
                }
            )

            logger.info(
                f"Streaming execution complete. "
                f"Events: {self._total_events}, Tools: {self._total_tools_executed}"
            )

        except Exception as e:
            logger.error(f"Error during streaming execution: {e}")
            yield StreamEvent.error(
                message=str(e),
                error_type="execution_error",
                recoverable=False,
                context={"dag_size": len(dag)}
            )

    def _resolve_dependencies(
        self,
        node: ToolCallNode,
        results_cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve node arguments using completed dependency results.

        Args:
            node: The node whose arguments to resolve.
            results_cache: Cache of completed results.

        Returns:
            Resolved arguments dictionary.
        """
        resolved_args = {}

        for key, value in node.arguments.items():
            if isinstance(value, dict) and "from_dep" in value:
                # Extract dependency reference
                dep_id = value["from_dep"]
                if dep_id in results_cache:
                    resolved_args[key] = results_cache[dep_id]
                else:
                    resolved_args[key] = value
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Support ${node_id} syntax
                dep_id = value[2:-1]
                if dep_id in results_cache:
                    resolved_args[key] = results_cache[dep_id]
                else:
                    resolved_args[key] = value
            else:
                resolved_args[key] = value

        return resolved_args

    async def _execute_tool(
        self,
        node: ToolCallNode,
        resolved_args: Dict[str, Any]
    ) -> Any:
        """Execute a single tool.

        Args:
            node: The tool call node.
            resolved_args: Resolved arguments.

        Returns:
            The tool execution result.
        """
        node.mark_running()

        try:
            # Execute with timeout using task pool
            result = await asyncio.wait_for(
                self.registry.execute(node.name, resolved_args),
                timeout=self.timeout
            )

            # Check for error in result
            if isinstance(result, dict) and "error" in result:
                node.mark_failed(result["error"])
            else:
                node.mark_done(result)

            return result

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {self.timeout}s"
            node.mark_failed(error_msg)
            logger.warning(f"Node {node.id} timed out")
            return {"error": error_msg}

        except ValueError as e:
            node.mark_failed(str(e))
            logger.error(f"Unknown tool in node {node.id}: {e}")
            return {"error": str(e)}

        except Exception as e:
            error_msg = str(e)
            node.mark_failed(error_msg)
            logger.error(f"Error executing node {node.id}: {error_msg}")
            return {"error": error_msg}

    async def _emit_tool_events(
        self,
        node: ToolCallNode,
        phase: str,
        extra_data: Dict = None
    ) -> StreamEvent:
        """Emit stream event for tool lifecycle.

        Args:
            node: The tool call node.
            phase: Current phase ('pending', 'ready', 'executing', 'result').
            extra_data: Additional data to include in the event.

        Returns:
            The generated StreamEvent.
        """
        self._total_events += 1

        event_data = {
            "tool_name": node.name,
            "node_id": node.id,
            "phase": phase,
        }

        if extra_data:
            event_data.update(extra_data)

        if node.execution_key:
            event_data["execution_key"] = node.execution_key

        # Determine event type based on phase
        event_type_map = {
            "pending": StreamEventType.TOOL_CALL_PENDING,
            "ready": StreamEventType.TOOL_CALL_READY,
            "executing": StreamEventType.TOOL_EXECUTING,
            "result": StreamEventType.TOOL_RESULT,
        }

        return StreamEvent(
            type=event_type_map.get(phase, StreamEventType.TOOL_RESULT),
            data=event_data
        )

    async def _aggregate_result(self, result: Any) -> Any:
        """Aggregate and validate result if structured output enabled.

        Args:
            result: The raw result to process.

        Returns:
            The processed/validated result.
        """
        if not self.enable_structured_output or self._enforcer is None:
            return result

        # Create an event for aggregation
        event = StreamEvent(
            type=StreamEventType.TOOL_RESULT,
            data={"result": result, "success": True}
        )

        # Add to aggregator
        await self._aggregator.add_chunk(event)

        # Get current aggregated result
        current = self._aggregator.get_current_result()

        # If we have a schema for this tool, validate
        schema = self._enforcer.get_schema("__last_tool__")
        if schema:
            validation_result = await self._enforcer.validate_and_extract(
                "__last_tool__",
                result,
                schema
            )
            if not validation_result.is_valid:
                logger.warning(f"Validation warnings: {validation_result.warnings}")

        return result

    async def shutdown(self) -> None:
        """Gracefully shutdown executor.

        Shuts down the internal task pool and cleans up resources.

        Example:
            >>> await executor.shutdown()
        """
        logger.info("Shutting down StreamingExecutor")
        await self._task_pool.shutdown(cancel_pending=True)
        logger.info("StreamingExecutor shutdown complete")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get executor statistics.

        Returns:
            Dictionary with execution statistics.
        """
        return {
            "total_events": self._total_events,
            "total_tools_executed": self._total_tools_executed,
            "task_pool": self._task_pool.stats,
            "aggregator": self._aggregator.stats,
            "enable_structured_output": self.enable_structured_output,
        }

    def register_schema(self, name: str, schema: OutputSchema) -> None:
        """Register a schema for structured output validation.

        Args:
            name: Name to associate with the schema.
            schema: The OutputSchema to register.

        Example:
            >>> from petals.client.async import OutputSchema
            >>> executor.register_schema(
            ...     "my_tool",
            ...     OutputSchema(required_fields=["data"])
            ... )
        """
        if self._enforcer:
            self._enforcer.register_schema(name, schema)

    def register_default_schemas(self) -> None:
        """Register default schemas for common tool types."""
        if self._enforcer:
            self._enforcer.register_defaults()

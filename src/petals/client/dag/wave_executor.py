"""
WaveExecutor - Wave-based execution for ToolCall DAG

Executes ToolCallDAG wave-by-wave with async support, handling
parallelism, concurrency limits, and execution key deduplication.
"""
import asyncio
import logging
import re
from typing import Any, Dict, Generator, List, Optional, Set

from petals.client.dag.dag import ToolCallDAG
from petals.client.dag.tool_call_node import ToolCallNode
from petals.client.tool_registry import ToolRegistry
from petals.data_structures import CallStatus

logger = logging.getLogger(__name__)


class WaveExecutor:
    """Executes ToolCallDAG wave-by-wave with async support.

    Execution model:
    1. Compute waves via topological sort
    2. For each wave: execute all nodes in parallel
    3. After wave completes: move to next wave
    4. Results available after each wave for streaming

    Attributes:
        registry: ToolRegistry for executing tools.
        timeout: Maximum time in seconds for a single tool execution.
        max_retries: Maximum number of retry attempts for failed nodes.
        concurrency_limit: Maximum number of concurrent executions.

    Example:
        >>> executor = WaveExecutor(registry, timeout=30.0, max_retries=3)
        >>> dag = ToolCallDAG()
        >>> # ... add nodes to dag ...
        >>> async for completed in executor.execute_dag(dag):
        ...     print(f"Completed: {completed.id}")
    """

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 30.0,
        max_retries: int = 3,
        concurrency_limit: int = 10
    ):
        """Initialize the WaveExecutor.

        Args:
            registry: ToolRegistry containing available tools.
            timeout: Maximum time in seconds for tool execution.
            max_retries: Maximum number of retry attempts for failed nodes.
            concurrency_limit: Maximum concurrent executions.
        """
        self.registry = registry
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrency_limit = concurrency_limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._execution_cache: Dict[str, Any] = {}  # Cache for execution_key deduplication

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the asyncio semaphore.

        Returns:
            The semaphore for concurrency limiting.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    def _resolve_dependencies(
        self,
        node: ToolCallNode,
        results_cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve node arguments using completed dependency results.

        Replaces dependency references in arguments with actual results.
        Supports patterns like {"from_dep": "node_id"} which get replaced
        with the result from that node.

        Args:
            node: The node whose arguments to resolve.
            results_cache: Dictionary mapping node IDs to their results.

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
                    # Keep original if dependency not found
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

    async def _execute_single_node(
        self,
        node: ToolCallNode,
        results_cache: Dict[str, Any]
    ) -> ToolCallNode:
        """Execute a single node with timeout and error handling.

        Args:
            node: The node to execute.
            results_cache: Cache of completed results for dependency resolution.

        Returns:
            The executed node with updated status and result/error.
        """
        node.mark_running()

        try:
            # Check for deduplication via execution_key
            if node.execution_key and node.execution_key in self._execution_cache:
                logger.debug(f"Using cached result for {node.id} (key: {node.execution_key})")
                node.mark_done(self._execution_cache[node.execution_key])
                return node

            # Resolve arguments from dependencies
            resolved_args = self._resolve_dependencies(node, results_cache)

            # Execute with concurrency limiting and timeout
            async with self._get_semaphore():
                result = await asyncio.wait_for(
                    self.registry.execute(node.name, resolved_args),
                    timeout=self.timeout
                )

            # Check for error in result
            if isinstance(result, dict) and "error" in result:
                node.mark_failed(result["error"])
            else:
                node.mark_done(result)

                # Cache for deduplication
                if node.execution_key:
                    self._execution_cache[node.execution_key] = result

        except asyncio.TimeoutError:
            node.mark_failed(f"Timeout after {self.timeout}s")
            logger.warning(f"Node {node.id} timed out after {self.timeout}s")

        except ValueError as e:
            # Unknown tool
            node.mark_failed(str(e))
            logger.error(f"Unknown tool in node {node.id}: {e}")

        except Exception as e:
            error_msg = str(e)
            node.mark_failed(error_msg, error_feedback=self._format_traceback(e))
            logger.error(f"Error executing node {node.id}: {error_msg}")

        return node

    def _format_traceback(self, exc: Exception) -> str:
        """Format exception as traceback string.

        Args:
            exc: The exception to format.

        Returns:
            Formatted traceback string.
        """
        import traceback
        return traceback.format_exception(type(exc), exc, exc.__traceback__)

    async def execute_wave(
        self,
        wave: List[ToolCallNode],
        results_cache: Dict[str, Any]
    ) -> List[ToolCallNode]:
        """Execute a single wave of nodes in parallel.

        Uses semaphore for concurrency limiting. Handles execution_key
        deduplication within the wave.

        Args:
            wave: List of nodes to execute in parallel.
            results_cache: Cache of results from previous waves.

        Returns:
            List of executed nodes with updated status.
        """
        # Create tasks for all nodes in the wave
        tasks = [
            self._execute_single_node(node, results_cache)
            for node in wave
        ]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Update results cache with completed nodes
        for node in results:
            if node.status == CallStatus.DONE:
                results_cache[node.id] = node.result

        return results

    async def execute_dag(
        self,
        dag: ToolCallDAG,
        initial_args: Dict[str, Any] = None
    ) -> Generator[ToolCallNode, None, None]:
        """Execute DAG wave-by-wave, yielding completed nodes.

        Yields ToolCallNode as each wave completes with results.
        This enables streaming behavior where parent nodes can
        start using child results as soon as they're available.

        Args:
            dag: The ToolCallDAG to execute.
            initial_args: Optional initial arguments for root nodes.

        Yields:
            ToolCallNode as each wave completes.

        Raises:
            ValueError: If a cycle is detected in the DAG.
        """
        # Check for cycles
        cycle = dag.detect_cycle()
        if cycle:
            raise ValueError(f"Cannot execute DAG with cycle: {' -> '.join(cycle)}")

        # Initialize results cache
        results_cache: Dict[str, Any] = initial_args.copy() if initial_args else {}

        # Reset execution cache for this DAG
        self._execution_cache.clear()

        # Compute waves
        try:
            waves = dag.get_waves()
        except ValueError as e:
            raise ValueError(f"Invalid DAG structure: {e}") from e

        logger.info(f"Executing DAG with {len(dag)} nodes in {len(waves)} waves")

        for wave_idx, wave in enumerate(waves):
            logger.debug(f"Executing wave {wave_idx + 1}/{len(waves)} with {len(wave)} nodes")

            # Execute wave
            completed = await self.execute_wave(wave, results_cache)

            # Yield each completed node
            for node in completed:
                yield node

            # Check for critical failures that should stop execution
            failed_nodes = [n for n in completed if n.status == CallStatus.FAILED]
            if failed_nodes:
                logger.warning(
                    f"Wave {wave_idx + 1} had {len(failed_nodes)} failures: "
                    f"{[n.id for n in failed_nodes]}"
                )

                # Optionally retry failed nodes with retry capability
                for node in failed_nodes:
                    if node.can_retry(self.max_retries):
                        node.increment_retry()
                        logger.info(f"Retrying node {node.id} (attempt {node.retry_count + 1})")

                        # Re-execute this specific node
                        retried = await self._execute_single_node(node, results_cache)
                        if retried.status == CallStatus.DONE:
                            results_cache[node.id] = retried.result
                        yield retried

        logger.info(f"DAG execution complete. Processed {len(dag)} nodes")

    async def execute_single(
        self,
        node: ToolCallNode,
        results_cache: Dict[str, Any] = None
    ) -> ToolCallNode:
        """Execute a single node without DAG context.

        Useful for testing or executing nodes outside of a full DAG.

        Args:
            node: The node to execute.
            results_cache: Optional cache of results for dependency resolution.

        Returns:
            The executed node.
        """
        if results_cache is None:
            results_cache = {}

        return await self._execute_single_node(node, results_cache)

    def reset(self) -> None:
        """Reset executor state for reuse."""
        self._execution_cache.clear()
        self._semaphore = None
